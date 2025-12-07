"""
Fast Masked Categorical Distribution for PPO training.

This module provides an optimized categorical distribution that:
1. Skips PyTorch's expensive validation in Categorical.__init__
2. Supports masked actions (invalid actions set to -inf logits)
3. Maintains numerical stability with log-softmax normalization

Performance Note:
- PyTorch's torch.distributions.Categorical validates inputs on every init
- This validation takes ~37ms per call, causing 7.7s overhead in training
- This implementation achieves the same functionality with minimal overhead
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class FastMaskedCategorical:
    """
    Fast categorical distribution that skips PyTorch validation.
    
    This is a drop-in replacement for torch.distributions.Categorical
    optimized for RL with masked action spaces.
    
    Args:
        logits: Raw logits tensor [B, num_actions]. Actions with -inf logits
                are treated as invalid/masked.
    """
    
    __slots__ = ('logits', 'probs', '_log_probs', '_batch_shape', '_num_actions')
    
    def __init__(self, logits: Tensor):
        """Initialize with raw logits (skips validation for speed)."""
        self.logits = logits
        self._batch_shape = logits.shape[:-1]
        self._num_actions = logits.shape[-1]
        # Compute log_probs once (log_softmax is numerically stable with -inf)
        self._log_probs = F.log_softmax(logits, dim=-1)
        # Cache probs for entropy (computed lazily for mode/sample)
        self.probs = None
    
    def _get_probs(self) -> Tensor:
        """Lazily compute probabilities."""
        if self.probs is None:
            self.probs = self._log_probs.exp()
        return self.probs
    
    def sample(self) -> Tensor:
        """Sample actions from the distribution."""
        # Use multinomial with probabilities
        probs = self._get_probs()
        # Handle fully-masked rows (all -inf) by setting uniform prob
        probs_safe = probs.clone()
        all_zero = probs_safe.sum(dim=-1) == 0
        if all_zero.any():
            probs_safe[all_zero] = 1.0 / self._num_actions
        # Multinomial sample
        return torch.multinomial(probs_safe, num_samples=1).squeeze(-1)
    
    def mode(self) -> Tensor:
        """Return the most likely action (argmax)."""
        return self.logits.argmax(dim=-1)
    
    def log_prob(self, actions: Tensor) -> Tensor:
        """Compute log probability of given actions."""
        # Gather log probs at action indices
        return self._log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    
    def entropy(self) -> Tensor:
        """Compute entropy of the distribution."""
        probs = self._get_probs()
        # Entropy: -sum(p * log(p)), handle zeros
        log_probs_safe = self._log_probs.clone()
        log_probs_safe[probs == 0] = 0  # 0 * log(0) = 0
        return -(probs * log_probs_safe).sum(dim=-1)


class FastCategoricalDistribution:
    """
    SB3-compatible distribution wrapper using FastMaskedCategorical.
    
    Drop-in replacement for stable_baselines3.common.distributions.CategoricalDistribution.
    
    Key optimization: Does NOT create a new torch.distributions.Categorical each call.
    """
    
    def __init__(self, action_dim: int):
        """Initialize with action dimension (for compatibility)."""
        self.action_dim = action_dim
        self.distribution: Optional[FastMaskedCategorical] = None
    
    def proba_distribution(self, action_logits: Tensor) -> "FastCategoricalDistribution":
        """Create/update distribution from logits - NO expensive validation."""
        # This is the key optimization: we create our fast distribution
        # instead of torch.distributions.Categorical which validates
        self.distribution = FastMaskedCategorical(action_logits)
        return self
    
    def sample(self) -> Tensor:
        """Sample actions."""
        return self.distribution.sample()
    
    def mode(self) -> Tensor:
        """Get most likely actions."""
        return self.distribution.mode()
    
    def log_prob(self, actions: Tensor) -> Tensor:
        """Compute log probability of actions."""
        return self.distribution.log_prob(actions)
    
    def entropy(self) -> Tensor:
        """Compute entropy."""
        return self.distribution.entropy()
    
    def get_actions(self, deterministic: bool = False) -> Tensor:
        """Get actions (SB3 compatibility)."""
        return self.mode() if deterministic else self.sample()
    
    def actions_from_params(self, action_logits: Tensor, deterministic: bool = False) -> Tensor:
        """Get actions from logits (SB3 compatibility)."""
        self.proba_distribution(action_logits)
        return self.mode() if deterministic else self.sample()
    
    def log_prob_from_params(self, action_logits: Tensor) -> Tensor:
        """Get log probs from logits (SB3 compatibility)."""
        raise NotImplementedError("Use sample() then log_prob() instead")

