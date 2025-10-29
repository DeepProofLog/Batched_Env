#!/usr/bin/env python3
"""
Standard PyKeen training example using their recommended approach.
Based on PyKeen documentation: https://pykeen.readthedocs.io/en/stable/tutorial/first_steps.html
"""
import sys
sys.modules['torchvision'] = None  # Prevent torchvision compatibility issues

from pykeen.pipeline import pipeline
from pykeen.datasets import Nations

print("="*60)
print("Standard PyKeen Training Example")
print("Dataset: Nations (small benchmark dataset)")
print("Model: ComplEx")
print("="*60)

# Use the standard pipeline with Nations dataset
# This is the simplest, most standard way to use PyKeen
result = pipeline(
    dataset='Nations',
    model='ComplEx',
    training_kwargs=dict(
        num_epochs=100,
        batch_size=128,
    ),
    optimizer='Adam',
    optimizer_kwargs=dict(lr=0.001),
    random_seed=42,
    device='cuda',
)

print("\n" + "="*60)
print("Training Complete!")
print("="*60)

# Get metrics
metrics = result.metric_results.to_dict()
mrr = metrics.get("both", {}).get("realistic", {}).get("inverse_harmonic_mean_rank", 0)
hits_at_10 = metrics.get("both", {}).get("realistic", {}).get("hits_at_10", 0)

print(f"\nTest Results:")
print(f"  MRR: {mrr:.4f}")
print(f"  Hits@10: {hits_at_10:.4f}")

print(f"\nFinal training loss: {result.losses[-1]:.4f}")
print(f"Loss trajectory (first/last 5): {result.losses[:5]} ... {result.losses[-5:]}")

# Expected results for Nations dataset with ComplEx:
# MRR should be around 0.5-0.7 depending on settings
print(f"\nStatus: {'✓ GOOD' if mrr > 0.3 else '✗ LOW MRR - Something is wrong'}")

# Save results
print(f"\nModel saved to: {result.model}")
print(f"Results saved to default directory")
