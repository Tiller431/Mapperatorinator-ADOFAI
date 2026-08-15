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
    # DDP wrap. Per-rank `.to(accelerator.device)` + rank-0 wandb.watch
    # was the 4×A40 failure: Rank 3 had 427 reducer params, rank 0 had 0.
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
    ddp_signature = assert_model_ready_for_ddp(model)
    print(
        f"[rank {accelerator.process_index}/{accelerator.num_processes}] "
        f"DDP reducer param tensors={len(ddp_signature)} "
        f"before accelerator.prepare()"
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
        ddp_signature = assert_model_ready_for_ddp(model)

    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args, accelerator)

    if args.model.manual_norm_weights:
        print("Manually normalizing model weights")
        model.transformer.register_step_post_hook(optimizer)
        model.transformer.norm_weights_()

    print(model)
    print_model_parameters(model)
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

    # Rank-0 wandb after DDP wrap so tracker hooks cannot freeze the
    # unwrapped module (reducer would then see 0 params on rank 0).
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
