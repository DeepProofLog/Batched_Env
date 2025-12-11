"""
Training Stability Improvements for PPO.

This module provides strategies to fix common training issues:
1. Negative explained variance - Value function failing
2. High approx_kl - Policy changes too aggressive  
3. Early stopping at epoch 1-2 - Training not converging

Key Strategies (without reducing LR):
- Return normalization for value function stability
- Adaptive clip range based on KL divergence
- Value function pretraining/warmup period
- Batch size adaptation
- Gradient clipping adjustments
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import numpy as np


class RunningMeanStd:
    """
    Tracks running mean and std for normalization.
    
    Used for return normalization to help the value function learn
    in a stable range regardless of the actual return magnitudes.
    """
    
    def __init__(self, epsilon: float = 1e-4, shape: Tuple = ()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
    
    def update(self, x: np.ndarray) -> None:
        """Update running statistics with new batch of data."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)
    
    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        """Update statistics using Welford's online algorithm."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize tensor using running statistics."""
        mean = torch.tensor(self.mean, dtype=x.dtype, device=x.device)
        std = torch.tensor(np.sqrt(self.var + 1e-8), dtype=x.dtype, device=x.device)
        return (x - mean) / std
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize tensor using running statistics."""
        mean = torch.tensor(self.mean, dtype=x.dtype, device=x.device)
        std = torch.tensor(np.sqrt(self.var + 1e-8), dtype=x.dtype, device=x.device)
        return x * std + mean


class AdaptiveClipRange:
    """
    Adaptive clip range that adjusts based on KL divergence.
    
    Instead of early stopping when KL is high, this reduces clip_range
    to constrain policy updates while still allowing learning.
    
    Strategy:
    - If KL > target_kl * 1.5: reduce clip_range by factor
    - If KL < target_kl * 0.5: increase clip_range slightly
    """
    
    def __init__(
        self,
        initial_clip_range: float = 0.2,
        min_clip_range: float = 0.05,
        max_clip_range: float = 0.3,
        target_kl: float = 0.02,
        adjustment_factor: float = 0.9,
    ):
        self.clip_range = initial_clip_range
        self.min_clip_range = min_clip_range
        self.max_clip_range = max_clip_range
        self.target_kl = target_kl
        self.adjustment_factor = adjustment_factor
        
    def adapt(self, approx_kl: float) -> float:
        """
        Adapt clip range based on current KL divergence.
        
        Returns:
            New clip range value
        """
        if approx_kl > self.target_kl * 1.5:
            # KL too high - reduce clip range
            self.clip_range = max(
                self.min_clip_range,
                self.clip_range * self.adjustment_factor
            )
        elif approx_kl < self.target_kl * 0.5:
            # KL too low - can increase clip range slightly
            self.clip_range = min(
                self.max_clip_range,
                self.clip_range / self.adjustment_factor
            )
        
        return self.clip_range
    
    def get_current(self) -> float:
        """Get current clip range without adaptation."""
        return self.clip_range


class ValueFunctionWarmup:
    """
    Implements value function warmup to improve early training stability.
    
    Strategy:
    - In early iterations, train value function with higher coefficient
    - Gradually reduce vf_coef to standard value
    - Optionally use separate learning rate for value function
    """
    
    def __init__(
        self,
        warmup_iterations: int = 10,
        initial_vf_coef: float = 2.0,
        final_vf_coef: float = 0.5,
        initial_n_epochs: int = 20,
        final_n_epochs: int = 10,
    ):
        self.warmup_iterations = warmup_iterations
        self.initial_vf_coef = initial_vf_coef
        self.final_vf_coef = final_vf_coef
        self.initial_n_epochs = initial_n_epochs
        self.final_n_epochs = final_n_epochs
        
    def get_params(self, iteration: int) -> Dict[str, float]:
        """
        Get training parameters for current iteration.
        
        Returns:
            Dict with 'vf_coef' and 'n_epochs' values
        """
        if iteration >= self.warmup_iterations:
            return {
                'vf_coef': self.final_vf_coef,
                'n_epochs': self.final_n_epochs,
            }
        
        progress = iteration / self.warmup_iterations
        
        # Linear interpolation
        vf_coef = self.initial_vf_coef + progress * (self.final_vf_coef - self.initial_vf_coef)
        n_epochs = int(self.initial_n_epochs + progress * (self.final_n_epochs - self.initial_n_epochs))
        
        return {
            'vf_coef': vf_coef,
            'n_epochs': n_epochs,
        }


class MRRTracker:
    """
    Tracks MRR during training for monitoring purposes.
    
    Provides:
    - Best MRR tracking
    - MRR trend analysis
    - Plateau detection
    """
    
    def __init__(self, patience: int = 10):
        self.history: list = []
        self.best_mrr: float = 0.0
        self.best_iteration: int = 0
        self.patience = patience
        self.no_improvement_count: int = 0
        
    def update(self, mrr: float, iteration: int) -> Dict[str, Any]:
        """
        Update tracker with new MRR value.
        
        Returns:
            Dict with tracking information
        """
        self.history.append({'iteration': iteration, 'mrr': mrr})
        
        is_best = mrr > self.best_mrr
        if is_best:
            self.best_mrr = mrr
            self.best_iteration = iteration
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        
        # Compute trend (last 5 vs previous 5)
        trend = self._compute_trend()
        
        return {
            'current_mrr': mrr,
            'best_mrr': self.best_mrr,
            'best_iteration': self.best_iteration,
            'is_best': is_best,
            'trend': trend,  # 'improving', 'declining', 'stable'
            'no_improvement_count': self.no_improvement_count,
            'should_stop': self.no_improvement_count >= self.patience,
        }
    
    def _compute_trend(self) -> str:
        """Compute MRR trend based on recent history."""
        if len(self.history) < 10:
            return 'insufficient_data'
        
        recent = [h['mrr'] for h in self.history[-5:]]
        previous = [h['mrr'] for h in self.history[-10:-5]]
        
        recent_mean = np.mean(recent)
        previous_mean = np.mean(previous)
        
        diff = recent_mean - previous_mean
        
        if diff > 0.01:
            return 'improving'
        elif diff < -0.01:
            return 'declining'
        else:
            return 'stable'
    
    def get_summary(self) -> str:
        """Get a formatted summary of MRR tracking."""
        if not self.history:
            return "No MRR data recorded"
        
        current = self.history[-1]['mrr']
        return (
            f"MRR: current={current:.3f}, best={self.best_mrr:.3f} "
            f"(iter {self.best_iteration}), trend={self._compute_trend()}"
        )


def compute_explained_variance_detailed(
    values: torch.Tensor,
    returns: torch.Tensor
) -> Dict[str, float]:
    """
    Compute detailed explained variance statistics.
    
    Returns:
        Dict with:
        - explained_variance: The standard EV metric
        - value_bias: Average over/under-estimation
        - value_correlation: Correlation between values and returns
        - prediction_error: Mean absolute error
    """
    values_flat = values.flatten().cpu().numpy()
    returns_flat = returns.flatten().cpu().numpy()
    
    # Standard explained variance
    y_var = np.var(returns_flat)
    if y_var == 0:
        ev = np.nan
    else:
        ev = 1 - np.var(returns_flat - values_flat) / y_var
    
    # Bias (over/under-estimation)
    bias = np.mean(values_flat - returns_flat)
    
    # Correlation
    if np.std(values_flat) > 0 and np.std(returns_flat) > 0:
        correlation = np.corrcoef(values_flat, returns_flat)[0, 1]
    else:
        correlation = 0.0
    
    # MAE
    mae = np.mean(np.abs(values_flat - returns_flat))
    
    return {
        'explained_variance': float(ev),
        'value_bias': float(bias),
        'value_correlation': float(correlation),
        'prediction_error_mae': float(mae),
    }


def diagnose_training_issues(
    metrics: Dict[str, float],
    iteration: int,
) -> list:
    """
    Diagnose training issues from metrics and provide recommendations.
    
    Args:
        metrics: Training metrics dict with keys like 'explained_var', 'approx_kl', etc.
        iteration: Current training iteration
        
    Returns:
        List of (issue, severity, recommendation) tuples
    """
    issues = []
    
    # Check explained variance
    ev = metrics.get('explained_var', 0)
    if ev < 0:
        severity = 'CRITICAL' if ev < -1 else 'WARNING'
        issues.append((
            f'Negative explained variance ({ev:.3f})',
            severity,
            'Increase vf_coef, enable return normalization, or use value warmup'
        ))
    elif ev < 0.5:
        issues.append((
            f'Low explained variance ({ev:.3f})',
            'INFO',
            'Value function is still learning - may improve with more training'
        ))
    
    # Check KL divergence
    kl = metrics.get('approx_kl', 0)
    target_kl = metrics.get('target_kl', 0.03)
    if kl > target_kl * 1.5:
        issues.append((
            f'High KL divergence ({kl:.4f} > {target_kl * 1.5:.4f})',
            'WARNING',
            'Policy updates too large. Use adaptive clip range or reduce n_epochs'
        ))
    
    # Check entropy
    entropy = metrics.get('entropy', 0)
    if entropy < 0.3:
        issues.append((
            f'Low entropy ({entropy:.3f})',
            'WARNING',
            'Policy too deterministic. Increase ent_coef or temperature'
        ))
    elif entropy > 2.0:
        issues.append((
            f'High entropy ({entropy:.3f})',
            'INFO',
            'Policy exploring a lot - may be early training or environment is hard'
        ))
    
    # Check value loss
    value_loss = metrics.get('value_loss', 0)
    if value_loss > 1.0 and iteration > 5:
        issues.append((
            f'High value loss ({value_loss:.3f}) late in training',
            'WARNING',
            'Value function struggling. Check return scaling or increase vf_coef'
        ))
    
    # Check clip fraction
    clip_frac = metrics.get('clip_fraction', 0)
    if clip_frac > 0.3:
        issues.append((
            f'High clip fraction ({clip_frac:.2%})',
            'WARNING',
            'Many updates clipped. Policy gradients might be very different'
        ))
    elif clip_frac < 0.01 and iteration > 3:
        issues.append((
            f'Very low clip fraction ({clip_frac:.2%})',
            'INFO',
            'Policy updates are conservative - may slow learning'
        ))
    
    return issues


def get_alternative_solutions_for_high_kl() -> str:
    """
    Get detailed recommendations for fixing high KL without reducing LR.
    """
    return """
## Solutions for High Approx KL (without reducing LR)

### 1. Reduce n_epochs (RECOMMENDED)
   - Fewer passes over the same data = smaller policy change
   - Example: n_epochs=20 → n_epochs=5
   - Each epoch compounds KL divergence

### 2. Increase batch_size  
   - Larger batches = more stable gradient estimates
   - Reduces variance in policy gradient
   - Example: batch_size=64 → batch_size=256 or higher

### 3. Use Adaptive Clip Range
   - Lower clip_range when KL is high
   - clip_range=0.2 → 0.1 dynamically
   - Constrains magnitude of policy update

### 4. Increase n_steps (rollout length)
   - More data per update = more stable learning
   - n_steps=128 → n_steps=256 or 512
   - Also helps explained variance

### 5. Increase target_kl threshold
   - Allow slightly higher KL before early stopping
   - target_kl=0.03 → target_kl=0.05 or 0.07
   - Gives more epochs to train

### 6. Enable Value Function Clipping (clip_range_vf)
   - Prevents large value updates from destabilizing policy
   - Use same value as clip_range (e.g., 0.2)

### 7. Normalize Advantages More Aggressively
   - Already normalized per-minibatch
   - Could normalize per-rollout instead for stability
"""


def get_alternative_solutions_for_negative_ev() -> str:
    """
    Get detailed recommendations for fixing negative explained variance.
    """
    return """
"""
"""
## Solutions for Negative Explained Variance (without reducing LR)

### 1. Increase vf_coef (RECOMMENDED)
   - Gives more weight to value function loss
   - Example: vf_coef=0.5 → vf_coef=1.0 or 2.0
   - Value function learns faster relative to policy

### 2. Normalize Returns
   - Keeps returns in a stable range for value function
   - Use running mean/std normalization
   - Prevents value predictions from exploding

### 3. Use Value Function Warmup
   - Train with higher vf_coef for first N iterations
   - Gradually decrease to standard value
   - Gives value function head start

### 4. Increase n_epochs
   - More passes = more value function updates
   - Value function often needs more updates than policy
   - BUT: may cause high KL - balance carefully

### 5. Increase Rollout Length (n_steps)
   - Better return estimates with longer trajectories
   - More diverse data for value function
   - n_steps=128 → n_steps=256 or 512

### 6. Check Reward Scale
   - If rewards are very large/small, normalize them
   - Value function learns best in [-1, 1] range

### 7. Use Huber Loss for Value Function
   - More robust to outliers than MSE
   - Less affected by extreme returns
"""

class RewardTracker:
    """
    Tracks training rewards for monitoring purposes.
    """
    
    def __init__(self, patience: int = 10):
        self.history: list = []
        self.best_reward: float = float('-inf')
        self.best_iteration: int = 0
        self.patience = patience
        self.no_improvement_count: int = 0
        
    def update(self, reward: float, iteration: int) -> Dict[str, Any]:
        """
        Update tracker with new reward value.
        
        Returns:
            Dict with tracking information
        """
        self.history.append({'iteration': iteration, 'reward': reward})
        
        is_best = reward > self.best_reward
        if is_best:
            self.best_reward = reward
            self.best_iteration = iteration
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        
        # Compute trend (last 5 vs previous 5)
        trend = self._compute_trend()
        
        return {
            'current_reward': reward,
            'best_reward': self.best_reward,
            'best_iteration': self.best_iteration,
            'is_best': is_best,
            'trend': trend,  # 'improving', 'declining', 'stable'
        }
    
    def _compute_trend(self) -> str:
        """Compute reward trend based on recent history."""
        if len(self.history) < 10:
            return 'insufficient_data'
        
        recent = [h['reward'] for h in self.history[-5:]]
        previous = [h['reward'] for h in self.history[-10:-5]]
        
        recent_mean = np.mean(recent)
        previous_mean = np.mean(previous)
        
        diff = recent_mean - previous_mean
        
        # Use smaller threshold for rewards maybe? or keep 0.01
        if diff > 0.005: 
            return 'improving'
        elif diff < -0.005:
            return 'declining'
        else:
            return 'stable'
    
    def get_summary(self) -> str:
        """Get a formatted summary of reward tracking."""
        if not self.history:
            return "No reward data recorded"
        
        current = self.history[-1]['reward']
        return (
            f"Reward (train): current={current:.3f}, best={self.best_reward:.3f} "
            f"(iter {self.best_iteration}), trend={self._compute_trend()}"
        )


if __name__ == "__main__":
    # Test the components
    print("Testing RunningMeanStd...")
    rms = RunningMeanStd()
    data = np.random.randn(100)
    rms.update(data)
    print(f"  Mean: {rms.mean:.3f}, Var: {rms.var:.3f}, Count: {rms.count}")
    
    print("\nTesting AdaptiveClipRange...")
    acr = AdaptiveClipRange(initial_clip_range=0.2, target_kl=0.02)
    print(f"  Initial: {acr.get_current():.3f}")
    print(f"  After KL=0.05: {acr.adapt(0.05):.3f}")
    print(f"  After KL=0.001: {acr.adapt(0.001):.3f}")
    
    print("\nTesting ValueFunctionWarmup...")
    vfw = ValueFunctionWarmup()
    for i in [0, 5, 10, 15]:
        params = vfw.get_params(i)
        print(f"  Iteration {i}: vf_coef={params['vf_coef']:.2f}, n_epochs={params['n_epochs']}")
    
    print("\nTesting MRRTracker...")
    tracker = MRRTracker()
    for i, mrr in enumerate([0.5, 0.55, 0.58, 0.57, 0.60, 0.62]):
        result = tracker.update(mrr, i)
        print(f"  Iter {i}: {result}")
    
    print("\n" + "="*60)
    print(get_alternative_solutions_for_high_kl())
    print("\n" + "="*60)
    print(get_alternative_solutions_for_negative_ev())
