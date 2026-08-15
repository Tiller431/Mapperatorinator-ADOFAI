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
    allgather_ddp_reducer_counts,
    assert_model_ready_for_ddp,
    assert_same_ddp_reducer_counts,
    bind_cuda_device_from_local_rank,
    configure_nccl_for_pcie_multigpu,
    ddp_reducer_named_parameters,
    resolve_mixed_precision,
    strip_fp32_output_conversion,
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

    # NCCL env *before* process-group init. 4×A40 PIX/PXB (no NVLink) hung
    # on the first DDP verify ALLGATHER (NumelIn=1 NumelOut=4); all 4 ranks
    # timed out together. 427-vs-0 after that timeout is not an empty module.
    nccl_defaults = configure_nccl_for_pcie_multigpu()
    if nccl_defaults:
        print(f"NCCL PCIe defaults applied before Accelerator(): {nccl_defaults}")

    # Fork the Manager *before* any CUDA / NCCL init. Manager() after
    # Accelerator() is CUDA+fork UB and breaks the first NCCL collective
    # on every rank (num_workers=0 does not avoid this).
    shared = get_shared_training_state()
    bind_cuda_device_from_local_rank()

    # accelerate launch defaults --mixed_precision to 'no' (launch.py
    # _validate_launch_command) and sets ACCELERATE_MIXED_PRECISION. Honor
    # adofai_v31 mixed_precision: bf16 instead of that default.
    mixed_precision = resolve_mixed_precision(args)
    print(f"mixed_precision={mixed_precision} (train config; not accelerate launch default 'no')")

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        cpu=args.device == "cpu",
        mixed_precision=mixed_precision,
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
    # the NCCL verify allgather. All 4 A40 ranks already printed
    # Embedding(95471, 768) / 478,783,248 / Muon 194 + AdamW 233 = 427.
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

    # Move before creating the optimizer so Muon/AdamW hold the same
    # Parameter objects DDP will wrap (not stale CPU tensors).
    if accelerator.device.type != "cpu":
        model.to(accelerator.device)
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
    sync_registered_modules(model)
    ddp_signature = assert_model_ready_for_ddp(model)
    # Gloo (CPU) gather of the 427-split. Does not use NCCL, so it cannot
    # be the PIX/PXB ALLGATHER hang. 427-vs-0 after an NCCL timeout is
    # not this check.
    reducer_counts = allgather_ddp_reducer_counts(len(ddp_signature))
    assert_same_ddp_reducer_counts(reducer_counts)
    print(
        f"[rank {accelerator.process_index}/{accelerator.num_processes}] "
        f"Gloo reducer counts={reducer_counts} "
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
    # prepare() wraps forward with convert_outputs_to_fp32 (accelerate 1.12.0
    # operations.py ConvertOutputsToFp32 → convert_to_fp32 → tensor.float()).
    # That clones Seq2SeqLMOutput.logits [B, 8192, vocab] to fp32 after
    # ``loss, stats = forward(...)`` and OOMs the same as mixed_precision=no.
    # Pop the wrapper; keep autocast. Loss/backward stay bf16.
    strip_fp32_output_conversion(model)

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
