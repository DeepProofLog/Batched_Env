
import sys
import unittest
import torch
import numpy as np
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SB3_ROOT = ROOT / "sb3"
if str(SB3_ROOT) not in sys.path:
    sys.path.insert(0, str(SB3_ROOT))

# Import the refactored modules
from train import run_experiment as run_tensor
from train import TrainParityConfig
from sb3.sb3_train_new import run_experiment as run_sb3

class TestSimpleParity(unittest.TestCase):
    def test_end_to_end_parity(self):
        """Run both training scripts and compare final metrics."""
        
        # Shared configuration
        config = TrainParityConfig(
            dataset="countries_s3",
            n_envs=4,
            n_steps=32,
            total_timesteps=512,  # 2 updates: 4*32=128 per update * 2 = 256
            n_corruptions=10,
            seed=42,
            device="cpu",
            verbose=False,
            n_epochs=4,
            batch_size=64,
            learning_rate=3e-4,
            ent_coef=0.0,
            clip_range=0.2,
        )
        
        print("\n" + "="*80)
        print("RUNNING TENSOR IMPLEMENTATION")
        print("="*80)
        tensor_results = run_tensor(config)
        
        print("\n" + "="*80)
        print("RUNNING SB3 IMPLEMENTATION")
        print("="*80)
        sb3_results = run_sb3(config)
        
        print("\n" + "="*80)
        print("COMPARISON RESULTS")
        print("="*80)
        print(f"{'Metric':<15} | {'Tensor':<10} | {'SB3':<10} | {'Diff':<10}")
        print("-" * 55)
        
        mismatches = []
        for key in ["MRR", "Hits@1"]:
            val_tensor = tensor_results.get(key, 0.0)
            val_sb3 = sb3_results.get(key, 0.0)
            diff = abs(val_tensor - val_sb3)
            print(f"{key:<15} | {val_tensor:<10.4f} | {val_sb3:<10.4f} | {diff:<10.4f}")
            
            if diff > 1e-4:
                mismatches.append(f"{key} mismatch: Tensor={val_tensor:.4f}, SB3={val_sb3:.4f}, Diff={diff:.4f}")
        
        if mismatches:
            self.fail("\n".join(mismatches))
        else:
            print("\nSUCCESS: All metrics match within tolerance!")

if __name__ == "__main__":
    unittest.main()
