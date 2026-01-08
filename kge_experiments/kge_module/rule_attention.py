"""
KGE Rule Attention Module.

Scores rule conclusions with KGE to provide attention weights for action selection.
When selecting which rule to apply, rules that lead to higher-scoring KGE conclusions
get higher probability.

Purpose: Guide the RL agent towards rules that produce facts likely to be true
according to the KGE model.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class KGERuleAttention(nn.Module):
    """Score rule conclusions with KGE for action selection.

    Computes KGE scores for the first atom of each derived state (the rule conclusion)
    and uses these as attention weights to bias action selection.

    Attributes:
        kge_engine: KGE inference engine for scoring.
        index_manager: IndexManager for converting tensor indices to strings.
        weight: Scaling factor for KGE attention.
        temperature: Temperature for softmax (lower = sharper).
    """

    def __init__(
        self,
        kge_engine: Any,
        index_manager: Any,
        weight: float = 0.5,
        temperature: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize KGE rule attention.

        Args:
            kge_engine: KGE inference engine with predict_batch method.
            index_manager: IndexManager for vocabulary lookups.
            weight: Scaling factor for adding KGE attention to logits.
            temperature: Temperature for softmax normalization.
            device: Target device.
        """
        super().__init__()
        self.kge_engine = kge_engine
        self.index_manager = index_manager
        self.weight = weight
        self.temperature = temperature
        self._device = device

        # Cache for pre-computed scores
        self._score_cache: Dict[Tuple[int, int, int], float] = {}

    def compute_attention_scores(
        self,
        derived_states: Tensor,  # [B, S, A, 3] derived states after unification
        valid_mask: Tensor,  # [B, S] mask for valid derived states
    ) -> Tensor:
        """Compute KGE-based attention scores for derived states.

        Scores the first atom of each derived state using the KGE model.

        Args:
            derived_states: [B, S, A, 3] derived states tensor.
                Each state contains atoms, each atom is (pred, arg1, arg2).
            valid_mask: [B, S] boolean mask indicating valid states.

        Returns:
            [B, S] KGE attention scores (unnormalized log-space).
        """
        B, S, A, _ = derived_states.shape
        device = derived_states.device

        # Extract first atom of each derived state: [B, S, 3]
        first_atoms = derived_states[:, :, 0, :]

        # Score with KGE: [B, S]
        kge_scores = self._batch_score_atoms(first_atoms)

        # Mask invalid states with -inf
        kge_scores = kge_scores.masked_fill(~valid_mask, float('-inf'))

        return kge_scores

    def compute_attention_weights(
        self,
        derived_states: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        """Compute normalized attention weights.

        Args:
            derived_states: [B, S, A, 3] derived states.
            valid_mask: [B, S] validity mask.

        Returns:
            [B, S] normalized attention weights (sum to 1 per batch).
        """
        scores = self.compute_attention_scores(derived_states, valid_mask)
        return F.softmax(scores / self.temperature, dim=-1)

    def augment_logits(
        self,
        action_logits: Tensor,  # [B, num_rules]
        derived_states: Tensor,  # [B, S, A, 3]
        valid_mask: Tensor,  # [B, S]
    ) -> Tensor:
        """Augment action logits with KGE attention.

        Formula: augmented_logits = action_logits + weight * kge_attention

        Args:
            action_logits: [B, num_rules] policy logits.
            derived_states: [B, S, A, 3] derived states.
            valid_mask: [B, S] validity mask.

        Returns:
            [B, num_rules] augmented logits.
        """
        kge_scores = self.compute_attention_scores(derived_states, valid_mask)

        # Handle dimension mismatch: derived_states may have different S than num_rules
        # Truncate or pad to match
        num_rules = action_logits.shape[1]
        if kge_scores.shape[1] > num_rules:
            kge_scores = kge_scores[:, :num_rules]
        elif kge_scores.shape[1] < num_rules:
            padding = torch.full(
                (kge_scores.shape[0], num_rules - kge_scores.shape[1]),
                float('-inf'),
                device=kge_scores.device,
            )
            kge_scores = torch.cat([kge_scores, padding], dim=1)

        return action_logits + self.weight * kge_scores

    def _batch_score_atoms(self, atoms: Tensor) -> Tensor:
        """Score atoms using KGE engine.

        Args:
            atoms: [B, S, 3] tensor of (pred, arg1, arg2) indices.

        Returns:
            [B, S] KGE scores.
        """
        B, S, _ = atoms.shape
        device = atoms.device

        # Convert to atom strings and score
        atom_strs = self._tensor_to_atom_strings(atoms)

        # Score using KGE engine
        scores = self.kge_engine.predict_batch(atom_strs)

        # Convert back to tensor: [B * S] -> [B, S]
        scores_tensor = torch.tensor(scores, dtype=torch.float32, device=device)
        return scores_tensor.view(B, S)

    def _tensor_to_atom_strings(self, atoms: Tensor) -> List[str]:
        """Convert atom tensor to list of atom strings.

        Args:
            atoms: [B, S, 3] tensor of indices.

        Returns:
            List of B*S atom strings in format "pred(arg1,arg2)".
        """
        B, S, _ = atoms.shape
        atoms_flat = atoms.view(-1, 3).cpu().numpy()

        atom_strs = []
        for pred_idx, arg1_idx, arg2_idx in atoms_flat:
            # Skip padding (index 0)
            if pred_idx == 0 or arg1_idx == 0 or arg2_idx == 0:
                atom_strs.append("")
                continue

            pred = self.index_manager.idx2pred.get(int(pred_idx), f"pred{pred_idx}")
            arg1 = self.index_manager.idx2const.get(int(arg1_idx), f"const{arg1_idx}")
            arg2 = self.index_manager.idx2const.get(int(arg2_idx), f"const{arg2_idx}")
            atom_strs.append(f"{pred}({arg1},{arg2})")

        return atom_strs

    def clear_cache(self) -> None:
        """Clear the score cache."""
        self._score_cache.clear()

    def __repr__(self) -> str:
        return f"KGERuleAttention(weight={self.weight}, temperature={self.temperature})"


class PrecomputedRuleAttention(nn.Module):
    """Rule attention using pre-computed KGE scores.

    For faster inference, uses a lookup table instead of runtime KGE scoring.
    Scores are loaded from a pre-computed file.
    """

    def __init__(
        self,
        score_lookup: Dict[str, float],
        weight: float = 0.5,
        temperature: float = 1.0,
        default_score: float = 0.0,
    ) -> None:
        """Initialize with pre-computed scores.

        Args:
            score_lookup: Dict mapping atom strings to scores.
            weight: Scaling factor for attention.
            temperature: Softmax temperature.
            default_score: Default score for atoms not in lookup.
        """
        super().__init__()
        self.score_lookup = score_lookup
        self.weight = weight
        self.temperature = temperature
        self.default_score = default_score

    def get_score(self, atom_str: str) -> float:
        """Get pre-computed score for an atom."""
        return self.score_lookup.get(atom_str, self.default_score)

    @classmethod
    def from_file(
        cls,
        filepath: str,
        weight: float = 0.5,
        temperature: float = 1.0,
    ) -> "PrecomputedRuleAttention":
        """Load pre-computed scores from file.

        Expected format per line: "pred(arg1,arg2) score rank"

        Args:
            filepath: Path to scores file.
            weight: Attention weight.
            temperature: Softmax temperature.

        Returns:
            PrecomputedRuleAttention instance.
        """
        score_lookup = {}
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    atom_str = parts[0]
                    score = float(parts[1])
                    score_lookup[atom_str] = score

        return cls(
            score_lookup=score_lookup,
            weight=weight,
            temperature=temperature,
        )


def create_rule_attention(
    config: Any,
    kge_engine: Optional[Any] = None,
    index_manager: Optional[Any] = None,
) -> Optional[KGERuleAttention]:
    """Factory function to create rule attention from config.

    Args:
        config: TrainConfig with rule attention settings.
        kge_engine: KGE inference engine.
        index_manager: IndexManager for vocabulary.

    Returns:
        KGERuleAttention if enabled, None otherwise.
    """
    enabled = getattr(config, 'kge_rule_attention', False)
    if not enabled:
        return None

    if kge_engine is None:
        print("[KGERuleAttention] KGE engine not provided, skipping")
        return None

    if index_manager is None:
        print("[KGERuleAttention] IndexManager not provided, skipping")
        return None

    weight = getattr(config, 'kge_rule_attention_weight', 0.5)
    temperature = getattr(config, 'kge_rule_attention_temperature', 1.0)
    verbose = getattr(config, 'verbose', True)

    attention = KGERuleAttention(
        kge_engine=kge_engine,
        index_manager=index_manager,
        weight=weight,
        temperature=temperature,
    )

    if verbose:
        print(f"[KGERuleAttention] Created with weight={weight}, temperature={temperature}")

    return attention


__all__ = [
    "KGERuleAttention",
    "PrecomputedRuleAttention",
    "create_rule_attention",
]
