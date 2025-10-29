
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
import os
import sys
from typing import Optional, List


def select_best_gpu_early(min_free_gb: float = 1.0) -> Optional[int]:
    """
    Select GPU with most free memory, run this before ANY torch imports.
    Uses subprocess to query nvidia-smi instead of torch.
    """
    import subprocess
    try:
        # Query GPU memory using nvidia-smi
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )
        
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split(',')
                if len(parts) == 2:
                    gpu_id = int(parts[0].strip())
                    free_mb = float(parts[1].strip())
                    free_gb = free_mb / 1024.0
                    print(f"GPU {gpu_id}: {free_gb:.2f} GB free")
                    if free_gb >= min_free_gb:
                        gpus.append((gpu_id, free_gb))
        
        if gpus:
            # Sort by free memory and pick the best one
            gpus.sort(key=lambda x: x[1], reverse=True)
            best_gpu = gpus[0][0]
            print(f"Selected GPU {best_gpu} with {gpus[0][1]:.2f} GB free memory")
            return best_gpu
        else:
            print(f"No GPU found with at least {min_free_gb:.1f} GB free memory")
            return None
            
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Warning: Could not query GPUs with nvidia-smi: {e}")
        return None


def get_available_gpus_early(min_free_gb: float = 1.0) -> List[int]:
    """
    Get all GPUs with sufficient memory, run this before ANY torch imports.
    Uses subprocess to query nvidia-smi instead of torch.
    """
    import subprocess
    try:
        # Query GPU memory using nvidia-smi
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )
        
        available_gpus = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split(',')
                if len(parts) == 2:
                    gpu_id = int(parts[0].strip())
                    free_mb = float(parts[1].strip())
                    free_gb = free_mb / 1024.0
                    print(f"GPU {gpu_id}: {free_gb:.2f} GB free")
                    if free_gb >= min_free_gb:
                        available_gpus.append(gpu_id)
        
        return available_gpus
            
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Warning: Could not query GPUs with nvidia-smi: {e}")
        return []


def setup_device(device_choice: str, min_memory_gb: float) -> tuple[bool, bool]:
    """
    Setup device selection and return whether to use CPU and multi-GPU.
    Must be called before importing kge_train_torch or torch.
    
    Args:
        device_choice: One of 'cpu', 'cuda:1', 'cuda:all'
        min_memory_gb: Minimum GPU memory required
    
    Returns:
        Tuple of (use_cpu, use_multi_gpu)
    """
    if device_choice == "cpu":
        print("\n=== Using CPU ===")
        print("Training will run on CPU (slower but always available)\n")
        return True, False
    
    elif device_choice == "cuda:1":
        print("\n=== Auto-selecting best GPU ===")
        best_gpu = select_best_gpu_early(min_free_gb=min_memory_gb)
        if best_gpu is not None:
            # Set CUDA_VISIBLE_DEVICES to use only the selected GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
            print(f"Set CUDA_VISIBLE_DEVICES={best_gpu}\n")
            return False, False
        else:
            print(f"No GPU with at least {min_memory_gb} GB free memory found.")
            print("Falling back to CPU\n")
            return True, False
    
    elif device_choice == "cuda:all":
        print("\n=== Using all available GPUs ===")
        available_gpus = get_available_gpus_early(min_free_gb=min_memory_gb)
        
        if len(available_gpus) == 0:
            print(f"No GPUs with at least {min_memory_gb} GB free memory found.")
            print("Falling back to CPU\n")
            return True, False
        elif len(available_gpus) == 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(available_gpus[0])
            print(f"Only 1 GPU available with sufficient memory: GPU {available_gpus[0]}")
            print(f"Set CUDA_VISIBLE_DEVICES={available_gpus[0]}\n")
            return False, False
        else:
            # Multiple GPUs available - enable multi-GPU training
            print(f"Found {len(available_gpus)} GPUs with sufficient memory: {available_gpus}")
            print(f"Multi-GPU training will use DataParallel across all {len(available_gpus)} GPUs.")
            print(f"Note: Each GPU will process a portion of the batch in parallel.")
            print(f"Consider scaling batch_size proportionally (e.g., multiply by {len(available_gpus)}) for optimal throughput.")
            # Set CUDA_VISIBLE_DEVICES to only the available GPUs
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, available_gpus))
            print(f"Set CUDA_VISIBLE_DEVICES={','.join(map(str, available_gpus))}\n")
            return False, True
    
    return False, False


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
    p.add_argument("--device", default="cuda:1", choices=["cpu", "cuda:1", "cuda:all"], 
                   help="Device: 'cpu' (use CPU), 'cuda:1' (auto-select best GPU), 'cuda:all' (use all available GPUs)")
    p.add_argument("--min_gpu_memory_gb", type=float, default=2.0, help="Minimum free GPU memory in GB to consider a GPU available")
    p.add_argument("--amp", default=True)
    p.add_argument("--compile", default=True)
    # Overrides
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--seed", type=int, default=3)
    return p

def main(argv=None):
    args = build_parser().parse_args(argv)

    # Handle device selection BEFORE importing kge_train_torch
    # This ensures CUDA_VISIBLE_DEVICES is set before PyTorch initializes CUDA
    use_cpu, use_multi_gpu = setup_device(args.device, args.min_gpu_memory_gb)
    
    # NOW import the training module after device is configured
    from kge_train_torch import TrainConfig, train_model

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
        cpu=use_cpu,
        multi_gpu=use_multi_gpu,
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
