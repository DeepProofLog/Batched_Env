
#!/usr/bin/env python3
"""Convenience runner to launch KGE training with sensible presets.

Presets:
- mrr_boost: RotatE with reciprocal relations, warmup + cosine scheduler
- baseline: RotatE baseline (no reciprocal, BCE loss)
- complex: ComplEx bilinear model with standard BCE training
- tucker: TuckER model with moderate dropout and AdamW
"""
from __future__ import annotations

import argparse

from kge_train_torch import TrainConfig, train_model


# PRESET_CHOICES = ("mrr_boost", "baseline", "complex", "tucker")
PRESET_CHOICES = ("rotate")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=PRESET_CHOICES, default="baseline")
    # Common IO
    p.add_argument("--save_dir", default="./kge_pytorch/models")
    p.add_argument("--train", dest="train_path")
    p.add_argument("--valid", dest="valid_path")
    p.add_argument("--test", dest="test_path")
    p.add_argument("--dataset", default="family", help="If provided, loads splits from data_root/dataset/*")
    p.add_argument("--data_root", default="./data")
    p.add_argument("--train_split", default="train.txt")
    p.add_argument("--valid_split", default="valid.txt")
    p.add_argument("--test_split", default="test.txt")
    # Hardware
    p.add_argument("--cpu", default =False, action="store_true")
    p.add_argument("--amp", default =True)
    p.add_argument("--compile", default =True)
    # Overrides
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--seed", type=int, default=3)
    return p

def main(argv=None):
    args = build_parser().parse_args(argv)

    shared_kwargs = dict(
        save_dir=args.save_dir,
        train_path=args.train_path,
        valid_path=args.valid_path,
        test_path=args.test_path,
        dataset=args.dataset,
        data_root=args.data_root,
        train_split=args.train_split,
        valid_split=args.valid_split,
        test_split=args.test_split,
        batch_size=args.batch_size,
        epochs=args.epochs,
        amp=args.amp,
        compile=args.compile,
        cpu=args.cpu,
        seed=args.seed,
    )

    preset = args.preset
    if preset == "baseline":
        preset_kwargs = dict(
            model="RotatE",
            dim=1024,
            gamma=12.0,
            p=1,
            lr=1e-3,
            neg_ratio=1,
            report_train_mrr=False,
            use_reciprocal=False,
            adv_temp=0.0,
            weight_decay=0.0,
            grad_clip=0.0,
            warmup_ratio=0.0,
            scheduler="none",
        )
    elif preset == "complex":
        preset_kwargs = dict(
            model="ComplEx",
            dim=1024,
            lr=5e-4,
            neg_ratio=1,
            report_train_mrr=False,
            use_reciprocal=False,
            adv_temp=0.0,
            weight_decay=1e-6,
            grad_clip=0.0,
            warmup_ratio=0.0,
            scheduler="none",
        )
    elif preset == "tucker":
        preset_kwargs = dict(
            model="TuckER",
            dim=512,
            relation_dim=256,
            dropout=0.3,
            lr=5e-4,
            neg_ratio=1,
            report_train_mrr=False,
            use_reciprocal=False,
            adv_temp=0.0,
            weight_decay=1e-6,
            grad_clip=1.0,
            warmup_ratio=0.0,
            scheduler="none",
        )
    elif preset == "mrr_boost":
        preset_kwargs = dict(
            model="RotatE",
            dim=1024,
            gamma=12.0,
            p=1,
            lr=1e-3,
            neg_ratio=1,
            report_train_mrr=False,
            use_reciprocal=True,
            adv_temp=0.0,
            weight_decay=1e-6,
            grad_clip=2.0,
            warmup_ratio=0.1,
            scheduler="cosine",
        )
    else:
        raise ValueError(f"Unknown preset: {preset}")

    cfg = TrainConfig(**shared_kwargs, **preset_kwargs)

    print("Launching training with config:\n", cfg)
    train_model(cfg)

if __name__ == "__main__":
    main()
