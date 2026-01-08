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


class GatedBridge(nn.Module):
    """Gated bridge with different weights for success vs failure.

    Formula:
    - If success: score = sigmoid(alpha_success) * rl + (1-sigmoid(alpha_success)) * kge
    - Else:       score = sigmoid(alpha_fail) * rl + (1-sigmoid(alpha_fail)) * kge

    Rationale: For proven candidates, RL signal is reliable. For unproven, rely more on KGE.

    Attributes:
        alpha_success: Learnable parameter for successful proofs.
        alpha_fail: Learnable parameter for failed proofs.
    """

    def __init__(
        self,
        init_alpha_success: float = 0.7,
        init_alpha_fail: float = 0.2,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize gated bridge.

        Args:
            init_alpha_success: Initial alpha for successful proofs (higher = more RL weight).
            init_alpha_fail: Initial alpha for failed proofs (lower = more KGE weight).
            device: Target device.
        """
        super().__init__()

        # Initialize alphas using logit transform
        self.alpha_success = nn.Parameter(
            torch.logit(torch.tensor(init_alpha_success, dtype=torch.float32))
        )
        self.alpha_fail = nn.Parameter(
            torch.logit(torch.tensor(init_alpha_fail, dtype=torch.float32))
        )

        if device is not None:
            self.to(device)

    @property
    def effective_alpha_success(self) -> float:
        """Return effective alpha for successful proofs."""
        return torch.sigmoid(self.alpha_success).item()

    @property
    def effective_alpha_fail(self) -> float:
        """Return effective alpha for failed proofs."""
        return torch.sigmoid(self.alpha_fail).item()

    @property
    def effective_alpha(self) -> float:
        """Return average alpha for compatibility."""
        return (self.effective_alpha_success + self.effective_alpha_fail) / 2

    def forward(
        self,
        rl_logprobs: Tensor,  # [B, K]
        kge_logprobs: Tensor,  # [B, K]
        success_mask: Optional[Tensor] = None,  # [B, K]
    ) -> Tensor:
        """Compute combined scores with gated weighting.

        Args:
            rl_logprobs: [B, K] RL log probabilities.
            kge_logprobs: [B, K] KGE log scores.
            success_mask: [B, K] boolean mask for proof success (required).

        Returns:
            [B, K] combined scores.
        """
        alpha_s = torch.sigmoid(self.alpha_success)
        alpha_f = torch.sigmoid(self.alpha_fail)

        scores_success = alpha_s * rl_logprobs + (1 - alpha_s) * kge_logprobs
        scores_fail = alpha_f * rl_logprobs + (1 - alpha_f) * kge_logprobs

        if success_mask is not None:
            scores = torch.where(success_mask, scores_success, scores_fail)
        else:
            # Fallback to average if no mask provided
            scores = (scores_success + scores_fail) / 2

        return scores

    def __repr__(self) -> str:
        return (
            f"GatedBridge(alpha_success={self.effective_alpha_success:.4f}, "
            f"alpha_fail={self.effective_alpha_fail:.4f})"
        )


class PerPredicateBridge(nn.Module):
    """Per-predicate learned weighting of RL and KGE scores.

    Different predicates may benefit from different balances between RL and KGE.
    Learns a separate alpha for each predicate type.

    Formula: score = sigmoid(alpha[pred_idx]) * rl + (1-sigmoid(alpha[pred_idx])) * kge

    Attributes:
        alphas: [n_predicates] learnable parameters, one per predicate.
    """

    def __init__(
        self,
        n_predicates: int,
        init_alpha: float = 0.5,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize per-predicate bridge.

        Args:
            n_predicates: Number of predicates in the dataset.
            init_alpha: Initial alpha value for all predicates.
            device: Target device.
        """
        super().__init__()
        self.n_predicates = n_predicates

        # Initialize all alphas to same value
        init_value = torch.logit(torch.tensor(init_alpha, dtype=torch.float32))
        self.alphas = nn.Parameter(torch.full((n_predicates,), init_value.item()))

        if device is not None:
            self.to(device)

    def get_effective_alphas(self) -> Tensor:
        """Return effective alpha values (after sigmoid)."""
        return torch.sigmoid(self.alphas)

    def forward(
        self,
        rl_logprobs: Tensor,  # [B, K] RL log probabilities
        kge_logprobs: Tensor,  # [B, K] KGE log scores
        pred_indices: Optional[Tensor] = None,  # [B, K] predicate indices
        success_mask: Optional[Tensor] = None,  # [B, K] proof success mask (unused)
    ) -> Tensor:
        """Compute combined scores with per-predicate alphas.

        Args:
            rl_logprobs: [B, K] RL log probabilities per candidate.
            kge_logprobs: [B, K] KGE log scores per candidate.
            pred_indices: [B, K] predicate indices for each candidate.
            success_mask: Optional [B, K] boolean mask (unused, for API compatibility).

        Returns:
            [B, K] combined scores.
        """
        if pred_indices is None:
            # Fallback to global average alpha
            alpha = torch.sigmoid(self.alphas.mean())
            return alpha * rl_logprobs + (1 - alpha) * kge_logprobs

        # Gather alpha for each predicate: [B, K]
        # Clamp indices to valid range for safety
        safe_indices = pred_indices.clamp(0, self.n_predicates - 1)
        alpha = torch.sigmoid(self.alphas[safe_indices])

        return alpha * rl_logprobs + (1 - alpha) * kge_logprobs

    def __repr__(self) -> str:
        alphas = self.get_effective_alphas().detach().cpu().numpy()
        return f"PerPredicateBridge(n_predicates={self.n_predicates}, alphas_range=[{alphas.min():.3f}, {alphas.max():.3f}])"


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
    n_predicates: Optional[int] = None,
) -> Optional[nn.Module]:
    """Factory function to create neural bridge from config.

    Args:
        config: TrainConfig with neural bridge settings.
        device: Target device.
        n_predicates: Number of predicates (required for per_predicate bridge).

    Returns:
        Bridge module if enabled, None otherwise.
    """
    enabled = getattr(config, 'neural_bridge', False)
    if not enabled:
        return None

    bridge_type = getattr(config, 'neural_bridge_type', 'linear')
    init_alpha = getattr(config, 'neural_bridge_init_alpha', 0.5)
    verbose = getattr(config, 'verbose', True)

    if bridge_type == 'gated':
        init_alpha_success = getattr(config, 'neural_bridge_init_alpha_success', 0.7)
        init_alpha_fail = getattr(config, 'neural_bridge_init_alpha_fail', 0.2)
        bridge = GatedBridge(
            init_alpha_success=init_alpha_success,
            init_alpha_fail=init_alpha_fail,
            device=device,
        )
        if verbose:
            print(f"[NeuralBridge] Created GatedBridge with "
                  f"init_alpha_success={init_alpha_success}, init_alpha_fail={init_alpha_fail}")
    elif bridge_type == 'per_predicate':
        if n_predicates is None:
            raise ValueError("n_predicates must be provided for per_predicate bridge")
        bridge = PerPredicateBridge(
            n_predicates=n_predicates,
            init_alpha=init_alpha,
            device=device,
        )
        if verbose:
            print(f"[NeuralBridge] Created PerPredicateBridge with "
                  f"n_predicates={n_predicates}, init_alpha={init_alpha}")
    elif bridge_type == 'mlp':
        hidden_dim = getattr(config, 'neural_bridge_hidden_dim', 32)
        bridge = MLPBridge(hidden_dim=hidden_dim, device=device)
        if verbose:
            print(f"[NeuralBridge] Created MLPBridge with hidden_dim={hidden_dim}")
    else:  # linear (default)
        bridge = LinearBridge(init_alpha=init_alpha, device=device)
        if verbose:
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
