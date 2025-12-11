#!/usr/bin/env python3
"""
Debug script for PPO training diagnostics.

This script adds detailed monitoring of:
1. Logit statistics (min, max, mean, std before and after masking)
2. Probability distribution statistics
3. Value function predictions vs actual returns
4. Learning dynamics (gradients, parameter updates)

Usage:
    python debug_training.py [--iterations N] [--save_stats]
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import argparse

# Set environment variables before importing other modules
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
torch.set_float32_matmul_precision('high')


def analyze_logits(logits: torch.Tensor, action_mask: torch.Tensor) -> Dict[str, float]:
    """Analyze logit statistics before and after masking."""
    with torch.no_grad():
        # Get valid logits (before masking with -inf)
        mask = action_mask.bool()
        valid_mask = mask.any(dim=-1)  # Which samples have valid actions
        
        # Global stats (including -inf for invalid)
        stats = {
            "logits_raw_min": logits.min().item(),
            "logits_raw_max": logits.max().item(),
            "logits_raw_mean": logits.mean().item() if not torch.isinf(logits).all() else float('nan'),
        }
        
        # Valid-only stats (exclude -inf)
        valid_logits = logits[mask]
        if len(valid_logits) > 0:
            stats["logits_valid_min"] = valid_logits.min().item()
            stats["logits_valid_max"] = valid_logits.max().item()
            stats["logits_valid_mean"] = valid_logits.mean().item()
            stats["logits_valid_std"] = valid_logits.std().item()
            
            # Softmax probabilities
            probs = torch.softmax(logits, dim=-1)
            valid_probs = probs[mask]
            stats["probs_valid_min"] = valid_probs.min().item()
            stats["probs_valid_max"] = valid_probs.max().item()
            stats["probs_valid_mean"] = valid_probs.mean().item()
            
            # Entropy per sample
            entropy = -(probs * torch.log(probs.clamp(min=1e-10))).masked_fill(~mask, 0).sum(dim=-1)
            valid_entropy = entropy[valid_mask]
            if len(valid_entropy) > 0:
                stats["entropy_min"] = valid_entropy.min().item()
                stats["entropy_max"] = valid_entropy.max().item()
                stats["entropy_mean"] = valid_entropy.mean().item()
                
                # Number of valid actions per sample
                num_valid = mask.sum(dim=-1).float()
                stats["num_valid_actions_min"] = num_valid.min().item()
                stats["num_valid_actions_max"] = num_valid.max().item()
                stats["num_valid_actions_mean"] = num_valid.mean().item()
                
                # Max entropy (uniform distribution)
                max_entropy = torch.log(num_valid.clamp(min=1))
                relative_entropy = (entropy / max_entropy.clamp(min=1e-10))[valid_mask]
                stats["relative_entropy_mean"] = relative_entropy.mean().item()
        else:
            stats["logits_valid_min"] = float('nan')
            stats["logits_valid_max"] = float('nan')
            stats["logits_valid_mean"] = float('nan')
            stats["logits_valid_std"] = float('nan')
            
        return stats


def analyze_values_returns(values: torch.Tensor, returns: torch.Tensor) -> Dict[str, float]:
    """Analyze value predictions vs actual returns."""
    with torch.no_grad():
        diff = values - returns
        return {
            "values_min": values.min().item(),
            "values_max": values.max().item(),
            "values_mean": values.mean().item(),
            "values_std": values.std().item(),
            "returns_min": returns.min().item(),
            "returns_max": returns.max().item(),
            "returns_mean": returns.mean().item(),
            "returns_std": returns.std().item(),
            "value_error_mean": diff.abs().mean().item(),
            "value_error_max": diff.abs().max().item(),
            "value_bias": diff.mean().item(),  # Over/under-estimation
        }


def analyze_advantages(advantages: torch.Tensor) -> Dict[str, float]:
    """Analyze advantage statistics."""
    with torch.no_grad():
        return {
            "advantages_min": advantages.min().item(),
            "advantages_max": advantages.max().item(),
            "advantages_mean": advantages.mean().item(),
            "advantages_std": advantages.std().item(),
            "advantages_pos_frac": (advantages > 0).float().mean().item(),
        }


def analyze_gradients(model: torch.nn.Module) -> Dict[str, float]:
    """Analyze gradient statistics across model parameters."""
    with torch.no_grad():
        grad_norms = []
        param_norms = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())
                param_norms.append(param.norm().item())
                
        if len(grad_norms) > 0:
            return {
                "grad_norm_mean": np.mean(grad_norms),
                "grad_norm_max": np.max(grad_norms),
                "grad_norm_min": np.min(grad_norms),
                "param_norm_mean": np.mean(param_norms),
                "param_norm_max": np.max(param_norms),
            }
        else:
            return {
                "grad_norm_mean": 0.0,
                "grad_norm_max": 0.0,
                "grad_norm_min": 0.0,
                "param_norm_mean": 0.0,
                "param_norm_max": 0.0,
            }


def print_training_health_report(
    logit_stats: Dict[str, float],
    value_stats: Dict[str, float],
    advantage_stats: Dict[str, float],
    loss_stats: Dict[str, float],
    gradient_stats: Optional[Dict[str, float]] = None,
) -> None:
    """Print a formatted training health report."""
    
    print("\n" + "=" * 70)
    print("TRAINING HEALTH REPORT")
    print("=" * 70)
    
    # Logit Analysis
    print("\n📊 LOGIT STATISTICS:")
    print(f"  Valid logits - Min: {logit_stats.get('logits_valid_min', 'N/A'):.3f}, "
          f"Max: {logit_stats.get('logits_valid_max', 'N/A'):.3f}, "
          f"Mean: {logit_stats.get('logits_valid_mean', 'N/A'):.3f}, "
          f"Std: {logit_stats.get('logits_valid_std', 'N/A'):.3f}")
    print(f"  Valid probs - Min: {logit_stats.get('probs_valid_min', 'N/A'):.4f}, "
          f"Max: {logit_stats.get('probs_valid_max', 'N/A'):.4f}")
    print(f"  Valid actions per sample - Mean: {logit_stats.get('num_valid_actions_mean', 'N/A'):.1f}")
    
    # Health check for logits
    logit_range = logit_stats.get('logits_valid_max', 0) - logit_stats.get('logits_valid_min', 0)
    if logit_range > 20:
        print("  ⚠️ WARNING: Large logit range detected - may cause numerical issues")
    elif logit_range < 0.1:
        print("  ⚠️ WARNING: Very small logit range - policy may be too deterministic or uniform")
    else:
        print("  ✅ Logit range looks healthy")
    
    # Entropy Analysis
    print("\n🎲 ENTROPY ANALYSIS:")
    print(f"  Raw entropy - Mean: {logit_stats.get('entropy_mean', 'N/A'):.3f}, "
          f"Min: {logit_stats.get('entropy_min', 'N/A'):.3f}, "
          f"Max: {logit_stats.get('entropy_max', 'N/A'):.3f}")
    print(f"  Relative entropy (vs uniform) - Mean: {logit_stats.get('relative_entropy_mean', 'N/A'):.3f}")
    
    rel_ent = logit_stats.get('relative_entropy_mean', 0.5)
    if rel_ent < 0.2:
        print("  ⚠️ WARNING: Very low relative entropy - policy may be too deterministic")
    elif rel_ent > 0.95:
        print("  ⚠️ WARNING: Very high relative entropy - policy may be too random (not learning)")
    else:
        print("  ✅ Entropy level looks healthy")
    
    # Value Function Analysis
    print("\n💰 VALUE FUNCTION ANALYSIS:")
    print(f"  Values - Min: {value_stats.get('values_min', 'N/A'):.3f}, "
          f"Max: {value_stats.get('values_max', 'N/A'):.3f}, "
          f"Mean: {value_stats.get('values_mean', 'N/A'):.3f}")
    print(f"  Returns - Min: {value_stats.get('returns_min', 'N/A'):.3f}, "
          f"Max: {value_stats.get('returns_max', 'N/A'):.3f}, "
          f"Mean: {value_stats.get('returns_mean', 'N/A'):.3f}")
    print(f"  Value error - Mean: {value_stats.get('value_error_mean', 'N/A'):.3f}, "
          f"Max: {value_stats.get('value_error_max', 'N/A'):.3f}")
    print(f"  Value bias (over/under estimation): {value_stats.get('value_bias', 'N/A'):.3f}")
    
    value_error = value_stats.get('value_error_mean', 0)
    returns_std = value_stats.get('returns_std', 1)
    if value_error > 2 * returns_std:
        print("  ⚠️ WARNING: Value predictions are far from returns - needs more training")
    else:
        print("  ✅ Value function predictions are reasonable")
    
    # Advantage Analysis
    print("\n📈 ADVANTAGE ANALYSIS:")
    print(f"  Advantages - Min: {advantage_stats.get('advantages_min', 'N/A'):.3f}, "
          f"Max: {advantage_stats.get('advantages_max', 'N/A'):.3f}, "
          f"Mean: {advantage_stats.get('advantages_mean', 'N/A'):.3f}")
    print(f"  Positive advantage fraction: {advantage_stats.get('advantages_pos_frac', 'N/A'):.2%}")
    
    pos_frac = advantage_stats.get('advantages_pos_frac', 0.5)
    if pos_frac < 0.3 or pos_frac > 0.7:
        print("  ⚠️ WARNING: Advantage distribution is unbalanced")
    else:
        print("  ✅ Advantage distribution is balanced")
    
    # Loss Analysis
    print("\n📉 LOSS ANALYSIS:")
    print(f"  Policy loss: {loss_stats.get('policy_loss', 'N/A'):.4f}")
    print(f"  Value loss: {loss_stats.get('value_loss', 'N/A'):.4f}")
    print(f"  Entropy: {loss_stats.get('entropy', 'N/A'):.4f}")
    print(f"  Approx KL: {loss_stats.get('approx_kl', 'N/A'):.5f}")
    print(f"  Clip fraction: {loss_stats.get('clip_fraction', 'N/A'):.4f}")
    print(f"  Explained variance: {loss_stats.get('explained_var', 'N/A'):.4f}")
    
    # Policy loss health
    policy_loss = loss_stats.get('policy_loss', 0)
    if policy_loss > 0.1:
        print("  ⚠️ WARNING: Positive policy loss - policy may be getting worse")
    elif policy_loss < -0.1:
        print("  ⚠️ WARNING: Very negative policy loss - updates may be too aggressive")
    else:
        print("  ✅ Policy loss is in healthy range")
    
    # KL divergence health
    approx_kl = loss_stats.get('approx_kl', 0)
    if approx_kl > 0.02:
        print("  ⚠️ WARNING: High KL divergence - policy changes may be too large")
    else:
        print("  ✅ KL divergence is within healthy bounds")
    
    # Explained variance health
    ev = loss_stats.get('explained_var', 0)
    if ev < 0:
        print("  ⚠️ WARNING: Negative explained variance - value function is worse than baseline")
    elif ev < 0.5:
        print("  ℹ️ INFO: Moderate explained variance - value function has room to improve")
    else:
        print("  ✅ Explained variance is good (value function is learning)")
    
    # Gradient Analysis
    if gradient_stats:
        print("\n🔧 GRADIENT ANALYSIS:")
        print(f"  Gradient norm - Mean: {gradient_stats.get('grad_norm_mean', 'N/A'):.4f}, "
              f"Max: {gradient_stats.get('grad_norm_max', 'N/A'):.4f}")
        print(f"  Parameter norm - Mean: {gradient_stats.get('param_norm_mean', 'N/A'):.4f}, "
              f"Max: {gradient_stats.get('param_norm_max', 'N/A'):.4f}")
        
        if gradient_stats.get('grad_norm_max', 0) > 10:
            print("  ⚠️ WARNING: Large gradients detected - may need gradient clipping")
        else:
            print("  ✅ Gradient magnitudes are reasonable")
    
    print("\n" + "=" * 70)


def get_training_recommendations(
    logit_stats: Dict[str, float],
    value_stats: Dict[str, float], 
    loss_stats: Dict[str, float],
) -> List[str]:
    """Generate training recommendations based on current statistics."""
    recommendations = []
    
    # Entropy too low
    rel_ent = logit_stats.get('relative_entropy_mean', 0.5)
    if rel_ent < 0.3:
        recommendations.append(
            "🔧 INCREASE ENTROPY: Consider increasing ent_coef or slowing entropy decay. "
            f"Current relative entropy: {rel_ent:.2f}"
        )
    
    # Explained variance bad
    ev = loss_stats.get('explained_var', 0)
    if ev < 0.3:
        recommendations.append(
            "🔧 IMPROVE VALUE FUNCTION: Consider:\n"
            "   - Increase vf_coef (e.g., 0.5 → 1.0)\n"
            "   - Decrease learning rate for more stable updates\n"
            "   - Increase n_epochs for more value function updates"
        )
    
    # High KL
    approx_kl = loss_stats.get('approx_kl', 0)
    if approx_kl > 0.02:
        recommendations.append(
            "🔧 REDUCE POLICY UPDATE MAGNITUDE: Consider:\n"
            "   - Lower learning rate\n"
            "   - Lower clip_range\n"
            "   - Lower target_kl threshold"
        )
    
    # High value loss
    value_loss = loss_stats.get('value_loss', 0)
    if value_loss > 0.5:
        recommendations.append(
            "🔧 HIGH VALUE LOSS: The value network is struggling. Consider:\n"
            "   - More training epochs (n_epochs)\n"
            "   - Larger batch size for more stable gradients\n"
            "   - Check if reward scale is appropriate"
        )
    
    # Logit range issues
    logit_std = logit_stats.get('logits_valid_std', 1.0)
    if logit_std > 10:
        recommendations.append(
            "🔧 LARGE LOGIT VARIANCE: Logits are too spread out. Consider:\n"
            "   - Enable L2 normalization (use_l2_norm=True)\n"
            "   - Increase temperature\n"
            "   - Enable sqrt scaling (sqrt_scale=True)"
        )
    elif logit_std < 0.1:
        recommendations.append(
            "🔧 SMALL LOGIT VARIANCE: Policy may be too deterministic. Consider:\n"
            "   - Decrease temperature\n"
            "   - Increase ent_coef"
        )
    
    return recommendations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug PPO training")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations to run")
    parser.add_argument("--save_stats", action="store_true", help="Save statistics to file")
    args = parser.parse_args()
    
    print("Running training diagnostics...")
    print("This script is meant to be integrated with the training loop.")
    print("\nTo use:")
    print("1. Import this module in train.py")
    print("2. Call analyze_logits(), analyze_values_returns() etc during training")
    print("3. Call print_training_health_report() periodically")
