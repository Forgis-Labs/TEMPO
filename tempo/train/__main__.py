"""Training CLI for TEMPO. Uses HF Accelerate for single/multi-GPU.

Usage:
    # 2-phase pipeline (alignment + training):
    python -m tempo.train pipeline \
        --tokenizer fsq_transformer --fsq_ckpt tokenizer.pt \
        --phase0_data data/pretokenized/stage0_alignment \
        --phase1_data data/pretokenized/stage1_base

    # Multi-GPU with FSDP:
    accelerate launch --config_file tempo/accelerate_fsdp_8gpu.yaml \
        -m tempo.train pipeline --tokenizer fsq_transformer --fsq_ckpt tokenizer.pt

    # Single phase (e.g., skip alignment):
    python -m tempo.train single \
        --data data/pretokenized/stage1_base --checkpoint phase0/best_model.pt

"""

import argparse

from tempo import TEMPO, TEMPOConfig
from .pipeline import run_pipeline, PipelineConfig
from .trainer import train
from .parquet_dataset import load_parquet_splits


def main():
    p = argparse.ArgumentParser(
        prog="python -m tempo.train",
        description="Train a TEMPO time series model.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    pp = sub.add_parser("pipeline", help="Run 2-phase training (alignment + training)")
    _add_model_args(pp)
    pp.add_argument("--phase0_data", default="",
                    help="Phase 0 alignment data dir (default: env TEMPO_PHASE0_DATA)")
    pp.add_argument("--phase1_data", default="",
                    help="Phase 1 training data dir (default: env TEMPO_PHASE1_DATA)")
    pp.add_argument("--skip_phase0", action="store_true",
                    help="Skip alignment phase (resume from phase0 checkpoint)")
    pp.add_argument("--phase0_lr", type=float, default=1e-3)
    pp.add_argument("--phase0_epochs", type=int, default=3)
    pp.add_argument("--phase1_lr", type=float, default=2e-5)
    pp.add_argument("--phase1_epochs", type=int, default=10)
    pp.add_argument("--embed_lr", type=float, default=None,
                    help="Separate LR for TS embeddings in phase 1")
    _add_training_args(pp)

    sp = sub.add_parser("single", help="Run single training phase")
    _add_model_args(sp)
    sp.add_argument("--data", required=True, help="Data directory with train/validation.parquet")
    sp.add_argument("--align_only", action="store_true",
                    help="Alignment mode: only TS embeddings train")
    sp.add_argument("--lr", type=float, default=2e-5)
    sp.add_argument("--epochs", type=int, default=10)
    _add_training_args(sp)

    args = p.parse_args()

    # Build model (on CPU — Accelerate handles device placement)
    model = _build_model(args)

    if args.command == "pipeline":
        config = PipelineConfig(
            phase0_data=args.phase0_data,
            phase1_data=args.phase1_data,
            output_dir=args.output_dir,
            phase0_epochs=args.phase0_epochs,
            phase0_lr=args.phase0_lr,
            phase1_epochs=args.phase1_epochs,
            phase1_lr=args.phase1_lr,
            phase1_embed_lr=args.embed_lr,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            num_workers=args.num_workers,
            checkpoint=args.checkpoint,
            skip_phase0=args.skip_phase0,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
        )
        run_pipeline(model, config)

    elif args.command == "single":
        eos = model.get_eos_token()
        dataset = load_parquet_splits(args.data, eos_token=eos)
        train(
            model, dataset,
            output_dir=args.output_dir,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.lr,
            align_only=args.align_only,
            num_workers=args.num_workers,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
        )


def _add_model_args(parser):
    g = parser.add_argument_group("model")
    g.add_argument("--tokenizer", choices=["totem", "fsq", "fsq_transformer"],
                   default="fsq_transformer")
    g.add_argument("--totem_ckpt", default=None)
    g.add_argument("--fsq_ckpt", default=None)
    g.add_argument("--llm_id", default="Qwen/Qwen3-4B")
    g.add_argument("--lora_r", type=int, default=32)
    g.add_argument("--lora_alpha", type=int, default=64)
    g.add_argument("--use_dora", action="store_true")
    g.add_argument("--checkpoint", default=None,
                   help="Starting checkpoint to resume from")


def _add_training_args(parser):
    g = parser.add_argument_group("training")
    g.add_argument("--batch_size", type=int, default=8)
    g.add_argument("--grad_accum", type=int, default=4)
    g.add_argument("--patience", type=int, default=3)
    g.add_argument("--num_workers", type=int, default=0)
    g.add_argument("--output_dir", default="results")
    g.add_argument("--wandb_project", default=None)
    g.add_argument("--wandb_run_name", default=None)


def _build_model(args):
    print(f"Building TEMPO: {args.llm_id} + {args.tokenizer}")
    config = TEMPOConfig(
        llm_id=args.llm_id,
        tokenizer_type=args.tokenizer,
        totem_ckpt=args.totem_ckpt,
        fsq_ckpt=args.fsq_ckpt,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_dora=args.use_dora,
    )
    model = TEMPO(config)  # CPU — Accelerate handles placement
    summary = model.trainable_summary()
    print(f"  Trainable: {summary['trainable']:,} / {summary['total']:,} "
          f"({summary['pct']:.1f}%)")
    return model


if __name__ == "__main__":
    main()
