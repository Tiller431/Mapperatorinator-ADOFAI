from repo_path import ensure_repo_root_on_sys_path

ensure_repo_root_on_sys_path()

import hydra
import os
import re
import torch
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration
from omegaconf import OmegaConf

from osuT5.config import TrainConfig
from osuT5.utils import (
    setup_args,
    train,
    train_profiling,
    load_model,
    get_scheduler,
    get_optimizer,
    get_dataloaders,
    get_shared_training_state,
    assert_model_ready_for_ddp,
    ddp_reducer_named_parameters,
    sync_registered_modules,
)


def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())  # Total parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # Trainable params
    frozen_params = total_params - trainable_params  # Non-trainable (frozen) params
    reducer_tensors = len(ddp_reducer_named_parameters(model))

    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Frozen Parameters: {frozen_params:,}")
    print(f"DDP reducer parameter tensors: {reducer_tensors}")


def get_next_checkpoint_iteration(checkpoint_root: str = "checkpoints") -> int:
    if not os.path.isdir(checkpoint_root):
        return 0

    checkpoint_indices = []
    for entry in os.listdir(checkpoint_root):
        match = re.fullmatch(r"checkpoint_(\d+)", entry)
        if match is not None and os.path.isdir(os.path.join(checkpoint_root, entry)):
            checkpoint_indices.append(int(match.group(1)))

    if not checkpoint_indices:
        return 0

    return max(checkpoint_indices) + 1


@hydra.main(config_path="../configs/train", config_name="v29", version_base="1.1")
def main(args: TrainConfig):
    args: TrainConfig = OmegaConf.to_object(args)
    checkpoint_iteration = get_next_checkpoint_iteration()

    # Fork the Manager *before* Accelerator touches CUDA. Manager() after
    # init_process_group is CUDA+fork UB and can leave rank 0 with a dead
    # context whose reducer param list is empty.
    shared = get_shared_training_state()

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        cpu=args.device == "cpu",
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.optim.grad_acc,
        log_with=args.logging.log_with,
        project_config=ProjectConfiguration(
            project_dir=".",
            logging_dir="tensorboard_logs",
            automatic_checkpoint_naming=True,
            total_limit=args.checkpoint.local_total_limit,
            iteration=checkpoint_iteration,
        ),
        kwargs_handlers=[ddp_kwargs],
    )
    wandb_kwargs = {
        "job_type": "training",
        "sync_tensorboard": args.profile.do_profile,
        "mode": args.logging.mode,
        "settings": wandb.Settings(x_graphql_timeout_seconds=120),
    }
    if args.logging.run_name:
        wandb_kwargs["name"] = args.logging.run_name

    setup_args(args)

    # Build on CPU so every rank materializes the same param set *before*
    # DDP wrap. The pod already printed Embedding(95471, 768), 478,783,248
    # params, and Muon 194 + AdamW 233 = 427; rank 0's module was empty at
    # ``_verify_param_shape_across_processes`` (same 427-vs-0 on ranks 0/1/2).
    with torch.enable_grad():
        model, tokenizer = load_model(
            args.pretrained_path,
            args,
            device="cpu",
            precision=args.precision,
            attn_implementation=args.attn_implementation,
            eval_mode=False,
            gamemode=args.pretrained_gamemode,
        )
    sync_registered_modules(model)
    ddp_signature = assert_model_ready_for_ddp(model)
    print(
        f"[rank {accelerator.process_index}/{accelerator.num_processes}] "
        f"DDP reducer param tensors={len(ddp_signature)} "
        f"(Muon+AdamW split of the full model) before device move"
    )
    train_dataloader, test_dataloader = get_dataloaders(tokenizer, args, shared)

    if args.enable_lora:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(**args.lora)
        model = get_peft_model(model, lora_config)
        # lora_params = {n: p for n, p in model.named_parameters() if "lora" in n}
        # for n, p in lora_params.items():
        #     print(n, p.sum())
        model.print_trainable_parameters()
        sync_registered_modules(model)
        ddp_signature = assert_model_ready_for_ddp(model)

    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args, accelerator)

    if args.model.manual_norm_weights:
        print("Manually normalizing model weights")
        model.transformer.register_step_post_hook(optimizer)
        model.transformer.norm_weights_()

    print(model)
    print_model_parameters(model)
    # print / Muon+AdamW already ran on the pod; the empty tree appeared
    # between those prints and DDP verify. Re-register children, move
    # every rank the same way, and refuse an empty ``_modules`` here.
    sync_registered_modules(model)
    if accelerator.device.type != "cpu":
        model.to(accelerator.device)
        sync_registered_modules(model)
    ddp_signature = assert_model_ready_for_ddp(model)
    print(
        f"[rank {accelerator.process_index}/{accelerator.num_processes}] "
        f"DDP reducer param tensors={len(ddp_signature)} "
        f"immediately before accelerator.prepare()"
    )
    accelerator.wait_for_everyone()

    # noinspection PyTypeChecker
    (
        model,
        optimizer,
        scheduler,
        train_dataloader,
        test_dataloader,
    ) = accelerator.prepare(
        model, optimizer, scheduler, train_dataloader, test_dataloader
    )

    # Rank-0 wandb after DDP wrap so tracker hooks cannot empty/freeze the
    # unwrapped module before ``_verify_param_shape_across_processes``.
    accelerator.init_trackers(
        "osuT5",
        init_kwargs={
            "wandb": wandb_kwargs,
        }
    )

    accelerator.register_for_checkpointing(tokenizer)

    if args.checkpoint_path:
        accelerator.load_state(args.checkpoint_path)
        shared.current_train_step = scheduler.scheduler.last_epoch // accelerator.num_processes + 1

    if args.compile:
        model = torch.compile(model)

    func = train_profiling if args.profile.do_profile else train

    func(
        model,
        train_dataloader,
        test_dataloader,
        accelerator,
        scheduler,
        optimizer,
        tokenizer,
        args,
        shared,
    )


if __name__ == "__main__":
    main()
