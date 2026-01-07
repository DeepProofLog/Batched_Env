"""
Neural Bridge Module for KGE + RL Fusion.

This module learns the optimal combination of RL and KGE logprobs for evaluation.
Formula: score = α * rl_logprobs + (1-α) * kge_logprobs

Training:
- Post-training fit on validation set
- Optimizes for MRR (Mean Reciprocal Rank)
- Uses differentiable approximation of ranking loss

Reference: Replaces fixed-weight hybrid scoring in ppo.py
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LinearBridge(nn.Module):
    """Linear combination of RL and KGE logprobs.

    Formula: score = sigmoid(alpha) * rl_logprobs + (1 - sigmoid(alpha)) * kge_logprobs

    Attributes:
        alpha: Learnable parameter controlling RL weight.
    """

    def __init__(
        self,
        init_alpha: float = 0.5,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize linear bridge.

        Args:
            init_alpha: Initial alpha value (will be sigmoid-transformed).
            device: Target device.
        """
        super().__init__()

        # Initialize alpha such that sigmoid(alpha) = init_alpha
        # sigmoid(x) = init_alpha => x = logit(init_alpha)
        init_value = torch.logit(torch.tensor(init_alpha, dtype=torch.float32))
        self.alpha = nn.Parameter(init_value)

        if device is not None:
            self.to(device)

    @property
    def effective_alpha(self) -> float:
        """Return the effective alpha value (after sigmoid)."""
        return torch.sigmoid(self.alpha).item()

    def forward(
        self,
        rl_logprobs: Tensor,  # [B, K] RL log probabilities
        kge_logprobs: Tensor,  # [B, K] KGE log scores
        success_mask: Optional[Tensor] = None,  # [B, K] proof success mask
    ) -> Tensor:
        """Compute combined scores.

        Args:
            rl_logprobs: [B, K] RL log probabilities per candidate.
            kge_logprobs: [B, K] KGE log scores per candidate.
            success_mask: Optional [B, K] boolean mask for proof success.

        Returns:
            [B, K] combined scores.
        """
        alpha = torch.sigmoid(self.alpha)
        scores = alpha * rl_logprobs + (1 - alpha) * kge_logprobs

        return scores

    def __repr__(self) -> str:
        return f"LinearBridge(alpha={self.effective_alpha:.4f})"


class MLPBridge(nn.Module):
    """MLP-based combination of RL and KGE features.

    More expressive than linear but may overfit.
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize MLP bridge.

        Args:
            hidden_dim: Hidden layer dimension.
            device: Target device.
        """
        super().__init__()

        # Input: [rl_logprob, kge_logprob] -> 2 features
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        if device is not None:
            self.to(device)

    def forward(
        self,
        rl_logprobs: Tensor,  # [B, K]
        kge_logprobs: Tensor,  # [B, K]
        success_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute combined scores.

        Args:
            rl_logprobs: [B, K] RL log probabilities.
            kge_logprobs: [B, K] KGE log scores.
            success_mask: Optional [B, K] proof success mask (unused).

        Returns:
            [B, K] combined scores.
        """
        B, K = rl_logprobs.shape

        # Stack features: [B, K, 2]
        features = torch.stack([rl_logprobs, kge_logprobs], dim=-1)

        # Flatten for MLP: [B*K, 2]
        features_flat = features.view(-1, 2)

        # Apply MLP: [B*K, 1]
        scores_flat = self.mlp(features_flat)

        # Reshape: [B, K]
        scores = scores_flat.view(B, K)

        return scores


def differentiable_mrr_loss(
    scores: Tensor,  # [B, K] scores per candidate (higher = better)
    target_idx: Tensor,  # [B] index of positive candidate (usually 0)
    temperature: float = 1.0,
) -> Tensor:
    """Differentiable approximation of MRR loss.

    Uses softmax-based ranking approximation.

    Args:
        scores: [B, K] scores for each candidate.
        target_idx: [B] index of the correct candidate.
        temperature: Softmax temperature (lower = sharper ranking).

    Returns:
        Scalar loss (negative MRR approximation).
    """
    B, K = scores.shape

    # Compute softmax probabilities (as ranking approximation)
    probs = F.softmax(scores / temperature, dim=-1)

    # Get probability of correct answer
    correct_probs = probs.gather(1, target_idx.unsqueeze(1)).squeeze(1)

    # MRR approximation: sum of probabilities at higher ranks
    # This is a smooth approximation of 1/rank
    mrr_approx = correct_probs

    # Negative MRR as loss (we want to maximize MRR)
    return -mrr_approx.mean()


def pairwise_ranking_loss(
    scores: Tensor,  # [B, K] scores
    target_idx: Tensor,  # [B] index of positive
    margin: float = 1.0,
) -> Tensor:
    """Pairwise margin ranking loss.

    Pushes positive score above all negative scores by margin.

    Args:
        scores: [B, K] scores for each candidate.
        target_idx: [B] index of correct candidate.
        margin: Margin for pairwise loss.

    Returns:
        Scalar loss.
    """
    B, K = scores.shape

    # Get positive scores: [B, 1]
    pos_scores = scores.gather(1, target_idx.unsqueeze(1))

    # All negatives: [B, K-1] (assumes positive is at index 0)
    # For general case, mask out the positive
    mask = torch.ones(B, K, dtype=torch.bool, device=scores.device)
    mask.scatter_(1, target_idx.unsqueeze(1), False)
    neg_scores = scores[mask].view(B, K - 1)

    # Pairwise loss: max(0, margin - (pos - neg))
    diff = pos_scores - neg_scores
    loss = F.relu(margin - diff).mean()

    return loss


class NeuralBridgeTrainer:
    """Trainer for neural bridge on validation data.

    Collects (rl_logprobs, kge_logprobs, success_mask) triplets from evaluation
    and trains the bridge to maximize MRR.
    """

    def __init__(
        self,
        bridge: nn.Module,
        lr: float = 0.01,
        epochs: int = 100,
        loss_type: str = 'mrr',  # 'mrr' or 'pairwise'
        verbose: bool = True,
    ) -> None:
        """Initialize trainer.

        Args:
            bridge: Bridge module to train.
            lr: Learning rate.
            epochs: Number of training epochs.
            loss_type: Loss function type.
            verbose: Print training progress.
        """
        self.bridge = bridge
        self.lr = lr
        self.epochs = epochs
        self.loss_type = loss_type
        self.verbose = verbose

        self.optimizer = torch.optim.Adam(bridge.parameters(), lr=lr)
        self.train_data: List[Dict[str, Tensor]] = []

    def add_validation_batch(
        self,
        rl_logprobs: Tensor,  # [B, K]
        kge_logprobs: Tensor,  # [B, K]
        success_mask: Tensor,  # [B, K]
        target_idx: Optional[Tensor] = None,  # [B] (default: 0 for all)
    ) -> None:
        """Add a batch of validation data for training.

        Args:
            rl_logprobs: [B, K] RL log probabilities.
            kge_logprobs: [B, K] KGE log scores.
            success_mask: [B, K] proof success mask.
            target_idx: [B] index of positive (default: 0).
        """
        B, K = rl_logprobs.shape
        if target_idx is None:
            target_idx = torch.zeros(B, dtype=torch.long, device=rl_logprobs.device)

        self.train_data.append({
            'rl_logprobs': rl_logprobs.detach(),
            'kge_logprobs': kge_logprobs.detach(),
            'success_mask': success_mask.detach(),
            'target_idx': target_idx.detach(),
        })

    def train(self) -> Dict[str, float]:
        """Train the bridge on collected validation data.

        Returns:
            Dict with final loss and alpha value.
        """
        if not self.train_data:
            print("[NeuralBridge] No validation data collected, skipping training")
            return {'loss': 0.0, 'alpha': self.bridge.effective_alpha if hasattr(self.bridge, 'effective_alpha') else 0.5}

        # Concatenate all data
        all_rl = torch.cat([d['rl_logprobs'] for d in self.train_data], dim=0)
        all_kge = torch.cat([d['kge_logprobs'] for d in self.train_data], dim=0)
        all_success = torch.cat([d['success_mask'] for d in self.train_data], dim=0)
        all_target = torch.cat([d['target_idx'] for d in self.train_data], dim=0)

        if self.verbose:
            print(f"[NeuralBridge] Training on {all_rl.shape[0]} queries, {all_rl.shape[1]} candidates each")

        self.bridge.train()
        final_loss = 0.0

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()

            # Forward pass
            scores = self.bridge(all_rl, all_kge, all_success)

            # Compute loss
            if self.loss_type == 'mrr':
                loss = differentiable_mrr_loss(scores, all_target)
            else:
                loss = pairwise_ranking_loss(scores, all_target)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            final_loss = loss.item()

            if self.verbose and (epoch + 1) % 10 == 0:
                alpha_str = f", alpha={self.bridge.effective_alpha:.4f}" if hasattr(self.bridge, 'effective_alpha') else ""
                print(f"  Epoch {epoch + 1}/{self.epochs}: loss={final_loss:.4f}{alpha_str}")

        self.bridge.eval()

        result = {'loss': final_loss}
        if hasattr(self.bridge, 'effective_alpha'):
            result['alpha'] = self.bridge.effective_alpha

        return result

    def clear_data(self) -> None:
        """Clear collected validation data."""
        self.train_data.clear()


def create_neural_bridge(
    config: Any,
    device: Optional[torch.device] = None,
) -> Optional[nn.Module]:
    """Factory function to create neural bridge from config.

    Args:
        config: TrainConfig with neural bridge settings.
        device: Target device.

    Returns:
        Bridge module if enabled, None otherwise.
    """
    enabled = getattr(config, 'neural_bridge', False)
    if not enabled:
        return None

    init_alpha = getattr(config, 'neural_bridge_init_alpha', 0.5)

    bridge = LinearBridge(init_alpha=init_alpha, device=device)

    if config.verbose if hasattr(config, 'verbose') else True:
        print(f"[NeuralBridge] Created LinearBridge with init_alpha={init_alpha}")

    return bridge


def create_bridge_trainer(
    bridge: nn.Module,
    config: Any,
) -> NeuralBridgeTrainer:
    """Create trainer for neural bridge.

    Args:
        bridge: Bridge module.
        config: TrainConfig with training settings.

    Returns:
        NeuralBridgeTrainer instance.
    """
    epochs = getattr(config, 'neural_bridge_train_epochs', 100)
    lr = getattr(config, 'neural_bridge_lr', 0.01)
    verbose = getattr(config, 'verbose', True)

    return NeuralBridgeTrainer(
        bridge=bridge,
        lr=lr,
        epochs=epochs,
        verbose=verbose,
    )
