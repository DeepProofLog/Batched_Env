"""
Profile GPU usage during the full test rollout pipeline
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.profiler import profile, ProfilerActivity
from types import SimpleNamespace

from tests.test_rollout import test_rollout_pipeline


def profile_gpu_test():
    """Profile GPU usage during the full test rollout pipeline"""

    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != 'cuda':
        print("CUDA not available, cannot profile GPU usage")
        return

    print(f"Using device: {device}")

    # Configuration for profiling - longer run for better bottleneck analysis
    args = SimpleNamespace(
        dataset='countries_s3',
        batch_size=50,   # Medium batch size
        n_steps=20,      # More steps per rollout
        n_epochs=10,      # Multiple training epochs
        total_timesteps=500,  # More timesteps for longer profiling
        device=device,
        seed=42,
    )

    print("Starting GPU profiling of test_rollout_pipeline...")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        test_rollout_pipeline(test_mode=None, args=args)

    print("\n" + "="*80)
    print("GPU PROFILING RESULTS - Top CUDA Time")
    print("="*80)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    # Save profiling results to file
    with open('tests/profile_gpu_results.txt', 'w') as f:
        f.write("GPU PROFILING RESULTS - Top CUDA Time\n")
        f.write("="*80 + "\n")
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        f.write("\n\nTop CPU Time:\n")
        f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))


if __name__ == '__main__':
    profile_gpu_test()