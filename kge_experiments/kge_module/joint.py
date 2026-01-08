"""
Joint KGE-RL Training Module.

Fine-tunes KGE embeddings alongside RL training by adding a KGE
contrastive loss term to the PPO objective.

Purpose: KGE learns from proof structure, RL benefits from KGE gradients.
Shared embeddings get optimized for both objectives.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class KGEContrastiveLoss(nn.Module):
    """Contrastive loss for KGE embedding training.

    Uses margin-based ranking loss to push positive triples above negatives.
    Can be added to PPO loss to jointly train embeddings.

    Formula: L = sum(max(0, margin + f(neg) - f(pos)))
    """

    def __init__(
        self,
        margin: float = 1.0,
        n_negatives: int = 10,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize contrastive loss.

        Args:
            margin: Margin for ranking loss.
            n_negatives: Number of negative samples per positive.
            device: Target device.
        """
        super().__init__()
        self.margin = margin
        self.n_negatives = n_negatives

        if device is not None:
            self.to(device)

    def forward(
        self,
        pos_scores: Tensor,  # [B] scores for positive triples
        neg_scores: Tensor,  # [B, N] scores for negative triples
    ) -> Tensor:
        """Compute margin ranking loss.

        Args:
            pos_scores: [B] positive triple scores.
            neg_scores: [B, N] negative triple scores.

        Returns:
            Scalar loss value.
        """
        # Expand pos_scores for broadcasting: [B, 1]
        pos_expanded = pos_scores.unsqueeze(1)

        # Compute margin loss: max(0, margin + neg - pos)
        loss_per_neg = F.relu(self.margin + neg_scores - pos_expanded)

        # Average over negatives and batch
        return loss_per_neg.mean()


class KGEJointTrainer:
    """Helper for joint KGE-RL training.

    Manages:
    - Collecting positive/negative samples from rollouts
    - Computing KGE contrastive loss
    - Combining with PPO loss

    Attributes:
        loss_fn: KGE contrastive loss function.
        lambda_kge: Weight for KGE loss in total loss.
        embedder: Shared embedder module.
        score_fn: Function to score triples with embedder.
    """

    def __init__(
        self,
        embedder: nn.Module,
        margin: float = 1.0,
        lambda_kge: float = 0.1,
        n_negatives: int = 10,
        verbose: bool = True,
    ) -> None:
        """Initialize joint trainer.

        Args:
            embedder: Shared embedder module (EmbedderLearnable).
            margin: Margin for contrastive loss.
            lambda_kge: Weight for KGE loss in total loss.
            n_negatives: Number of negative samples per positive.
            verbose: Print training statistics.
        """
        self.embedder = embedder
        self.lambda_kge = lambda_kge
        self.n_negatives = n_negatives
        self.verbose = verbose

        self.loss_fn = KGEContrastiveLoss(margin=margin, n_negatives=n_negatives)

        # Accumulated samples for batch training
        self.positive_triples: List[Tensor] = []
        self.negative_triples: List[Tensor] = []

        if verbose:
            print(f"[KGEJoint] Initialized with lambda={lambda_kge}, margin={margin}")

    def add_samples(
        self,
        positive: Tensor,  # [N, 3] positive triples (pred, head, tail)
        negative: Tensor,  # [N, K, 3] negative triples
    ) -> None:
        """Add positive/negative samples for training.

        Args:
            positive: [N, 3] positive triples.
            negative: [N, K, 3] negative triples per positive.
        """
        self.positive_triples.append(positive.detach())
        self.negative_triples.append(negative.detach())

    def compute_loss(self) -> Tensor:
        """Compute KGE contrastive loss from accumulated samples.

        Returns:
            Scalar KGE loss (multiply by lambda_kge before adding to PPO loss).
        """
        if not self.positive_triples:
            return torch.tensor(0.0)

        # Concatenate accumulated samples
        pos = torch.cat(self.positive_triples, dim=0)  # [B, 3]
        neg = torch.cat(self.negative_triples, dim=0)  # [B, K, 3]

        # Score with embedder
        pos_scores = self._score_triples(pos)  # [B]
        neg_scores = self._score_triples_batch(neg)  # [B, K]

        # Compute contrastive loss
        loss = self.loss_fn(pos_scores, neg_scores)

        return loss

    def get_total_loss(self, ppo_loss: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        """Combine PPO loss with KGE loss.

        Args:
            ppo_loss: PPO policy/value loss.

        Returns:
            Tuple of:
            - total_loss: PPO + lambda * KGE loss
            - losses_dict: Individual loss values for logging
        """
        kge_loss = self.compute_loss()
        total_loss = ppo_loss + self.lambda_kge * kge_loss

        losses = {
            'ppo_loss': ppo_loss.item(),
            'kge_loss': kge_loss.item(),
            'total_loss': total_loss.item(),
        }

        return total_loss, losses

    def clear_samples(self) -> None:
        """Clear accumulated samples."""
        self.positive_triples.clear()
        self.negative_triples.clear()

    def _score_triples(self, triples: Tensor) -> Tensor:
        """Score triples using embedder.

        Args:
            triples: [B, 3] triples (pred, head, tail indices).

        Returns:
            [B] scores for each triple.
        """
        # Get embeddings for triples
        # triples shape: [B, 3] -> need to reshape for embedder
        B = triples.shape[0]
        triples_expanded = triples.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, 3]

        embeddings = self.embedder.get_embeddings_batch(triples_expanded)  # [B, 1, D]
        embeddings = embeddings.squeeze(1)  # [B, D]

        # Score = negative L2 norm (higher = better)
        scores = -embeddings.norm(dim=-1)

        return scores

    def _score_triples_batch(self, triples: Tensor) -> Tensor:
        """Score batch of negative triples.

        Args:
            triples: [B, K, 3] negative triples.

        Returns:
            [B, K] scores.
        """
        B, K, _ = triples.shape

        # Reshape for embedder: [B, K, 1, 3]
        triples_expanded = triples.unsqueeze(2)

        # Get embeddings: [B, K, D]
        embeddings = self.embedder.get_embeddings_batch(triples_expanded)

        # Score = negative L2 norm
        scores = -embeddings.norm(dim=-1)

        return scores


def create_negative_samples(
    positive: Tensor,  # [B, 3]
    n_entities: int,
    n_negatives: int = 10,
    corruption_mode: str = 'both',  # 'head', 'tail', or 'both'
) -> Tensor:
    """Create negative samples by corrupting positive triples.

    Args:
        positive: [B, 3] positive triples (pred, head, tail).
        n_entities: Total number of entities for sampling.
        n_negatives: Number of negatives per positive.
        corruption_mode: Which position to corrupt.

    Returns:
        [B, N, 3] negative triples.
    """
    B = positive.shape[0]
    device = positive.device

    # Repeat positives for each negative
    negatives = positive.unsqueeze(1).expand(B, n_negatives, 3).clone()  # [B, N, 3]

    # Sample random entities
    random_entities = torch.randint(0, n_entities, (B, n_negatives), device=device)

    if corruption_mode == 'head':
        negatives[:, :, 1] = random_entities  # Replace head
    elif corruption_mode == 'tail':
        negatives[:, :, 2] = random_entities  # Replace tail
    else:  # both
        # Alternate between head and tail corruption
        head_mask = torch.arange(n_negatives, device=device) % 2 == 0
        head_mask = head_mask.unsqueeze(0).expand(B, -1)

        negatives[:, :, 1] = torch.where(head_mask, random_entities, negatives[:, :, 1])
        negatives[:, :, 2] = torch.where(~head_mask, random_entities, negatives[:, :, 2])

    return negatives


def extract_triples_from_rollout(
    states: Tensor,  # [B, T, S, A, 3] state tensor from rollout
    success_mask: Tensor,  # [B, T] success indicator
) -> Optional[Tensor]:
    """Extract positive triples from successful rollouts.

    Args:
        states: State tensor from rollout buffer.
        success_mask: Boolean mask for successful episodes.

    Returns:
        [N, 3] positive triples from successful proofs, or None if none found.
    """
    # Get first goal of each state (assuming first atom is the goal)
    # states: [B, T, S, A, 3] -> goals: [B, T, 3]
    goals = states[:, :, 0, 0, :]  # First state, first atom

    # Flatten and filter by success
    B, T = success_mask.shape
    goals_flat = goals.view(B * T, 3)
    success_flat = success_mask.view(B * T)

    positives = goals_flat[success_flat]

    if positives.shape[0] == 0:
        return None

    return positives


def create_kge_joint_trainer(
    config: Any,
    embedder: nn.Module,
) -> Optional[KGEJointTrainer]:
    """Factory function to create joint trainer from config.

    Args:
        config: TrainConfig with joint training settings.
        embedder: Shared embedder module.

    Returns:
        KGEJointTrainer if enabled, None otherwise.
    """
    if not getattr(config, 'kge_joint_training', False):
        return None

    lambda_kge = getattr(config, 'kge_joint_lambda', 0.1)
    margin = getattr(config, 'kge_joint_margin', 1.0)
    verbose = getattr(config, 'verbose', True)

    return KGEJointTrainer(
        embedder=embedder,
        margin=margin,
        lambda_kge=lambda_kge,
        verbose=verbose,
    )
