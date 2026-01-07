"""
KGE Unification Scorer Module.

This module provides KGE scoring for derived states from unification,
implemented as a wrapper/post-processor to keep the unification engine clean.

Two modes:
- Offline: Use pre-computed scores from top-k files (fast)
- Online: Score atoms at runtime via KGE engine (slower but handles any state)

Design: Wrapper approach (per user preference) - keeps unification.py clean.
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor


class UnificationScorer:
    """Scores derived states from unification using KGE.

    This module wraps around the unification output and adds KGE-based scoring
    to the derived states. It does NOT modify the unification engine itself.

    Attributes:
        mode: 'offline' (pre-computed) or 'online' (runtime scoring)
        device: Target device for tensors
    """

    def __init__(
        self,
        mode: str = 'offline',
        device: Optional[torch.device] = None,
        kge_engine: Optional[Any] = None,
        index_manager: Optional[Any] = None,
        eps: float = 1e-9,
    ) -> None:
        """Initialize unification scorer.

        Args:
            mode: 'offline' (use pre-computed scores) or 'online' (runtime scoring).
            device: Target device.
            kge_engine: KGE inference engine (required for online mode).
            index_manager: Index manager for atom conversion.
            eps: Small constant for numerical stability.
        """
        self.mode = mode
        self.device = device or torch.device('cpu')
        self.kge_engine = kge_engine
        self.index_manager = index_manager
        self.eps = eps

        # Score cache: atom_str -> score
        self._score_cache: Dict[str, float] = {}

        # Reverse index lookups (built lazily)
        self._idx2pred: Optional[Dict[int, str]] = None
        self._idx2const: Optional[Dict[int, str]] = None
        self._padding_idx: int = 0

    def _build_reverse_lookups(self) -> None:
        """Build reverse index lookups from index manager."""
        if self.index_manager is None:
            return

        self._idx2pred = {v: k for k, v in self.index_manager.predicate_str2idx.items()}
        self._idx2const = {v: k for k, v in self.index_manager.constant_str2idx.items()}
        self._padding_idx = self.index_manager.padding_idx

    def load_precomputed_scores(self, filepath: str) -> None:
        """Load pre-computed scores from a top-k file.

        Args:
            filepath: Path to top-k file (format: "predicate(head,tail) score rank")
        """
        self._score_cache.clear()

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if len(parts) < 2:
                    continue

                atom_repr = parts[0]
                try:
                    score = float(parts[1])
                except ValueError:
                    continue

                self._score_cache[atom_repr] = score

        print(f"[UnificationScorer] Loaded {len(self._score_cache)} precomputed scores")

    def score_derived_states(
        self,
        derived_states: Tensor,  # [B, S, A, 3] derived states
        valid_mask: Tensor,  # [B, S] valid state mask
    ) -> Tensor:
        """Score derived states with KGE.

        Args:
            derived_states: [B, S, A, 3] derived states from unification.
            valid_mask: [B, S] boolean mask for valid states.

        Returns:
            [B, S] tensor of KGE scores for each derived state.
        """
        B, S, A, _ = derived_states.shape
        scores = torch.zeros(B, S, dtype=torch.float32, device=self.device)

        if self._idx2pred is None:
            self._build_reverse_lookups()

        if self._idx2pred is None or self._idx2const is None:
            return scores

        # Score first atom of each derived state
        first_atoms = derived_states[:, :, 0, :]  # [B, S, 3]

        for b in range(B):
            for s in range(S):
                if not valid_mask[b, s]:
                    continue

                pred_idx = first_atoms[b, s, 0].item()
                if pred_idx == self._padding_idx:
                    continue

                pred_name = self._idx2pred.get(int(pred_idx))
                if pred_name is None:
                    continue

                arg0_idx = first_atoms[b, s, 1].item()
                arg1_idx = first_atoms[b, s, 2].item()
                arg0 = self._idx2const.get(int(arg0_idx), str(int(arg0_idx)))
                arg1 = self._idx2const.get(int(arg1_idx), str(int(arg1_idx)))

                atom_str = f"{pred_name}({arg0},{arg1})"

                # Check cache first
                if atom_str in self._score_cache:
                    scores[b, s] = self._score_cache[atom_str]
                elif self.mode == 'online' and self.kge_engine is not None:
                    # Score at runtime
                    try:
                        score_list = self.kge_engine.predict_batch([atom_str])
                        if score_list:
                            score = float(score_list[0])
                            self._score_cache[atom_str] = score
                            scores[b, s] = score
                    except Exception:
                        pass

        return scores

    def filter_top_k(
        self,
        derived_states: Tensor,  # [B, S, A, 3]
        scores: Tensor,  # [B, S] scores from score_derived_states
        k: int,
        valid_counts: Tensor,  # [B] original valid counts
    ) -> Tuple[Tensor, Tensor]:
        """Filter to keep only top-k scored states per batch.

        Args:
            derived_states: [B, S, A, 3] derived states.
            scores: [B, S] KGE scores.
            k: Number of top states to keep.
            valid_counts: [B] original valid state counts.

        Returns:
            Tuple of (filtered_states [B, k, A, 3], filtered_counts [B])
        """
        B, S, A, _ = derived_states.shape
        device = derived_states.device

        # Get top-k indices per batch
        # Use topk on scores (lower indices break ties for equal scores)
        k_clamped = min(k, S)
        _, top_indices = scores.topk(k_clamped, dim=1, largest=True, sorted=True)  # [B, k]

        # Gather top-k states
        top_states = torch.zeros(B, k_clamped, A, 3, dtype=derived_states.dtype, device=device)
        for b in range(B):
            top_states[b] = derived_states[b, top_indices[b]]

        # Update valid counts (min of k and original count)
        new_counts = torch.minimum(valid_counts, torch.full_like(valid_counts, k_clamped))

        return top_states, new_counts

    def score_and_rerank(
        self,
        derived_states: Tensor,  # [B, S, A, 3]
        valid_counts: Tensor,  # [B]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Score derived states and rerank by KGE score.

        Args:
            derived_states: [B, S, A, 3] derived states.
            valid_counts: [B] valid state counts.

        Returns:
            Tuple of (reranked_states, scores, valid_counts)
        """
        B, S = derived_states.shape[0], derived_states.shape[1]

        # Create valid mask
        valid_mask = torch.arange(S, device=self.device).unsqueeze(0) < valid_counts.unsqueeze(1)

        # Score states
        scores = self.score_derived_states(derived_states, valid_mask)

        # Mask out invalid states with low score
        scores = scores.masked_fill(~valid_mask, -1e9)

        # Sort by score (descending)
        sorted_scores, sort_indices = scores.sort(dim=1, descending=True)

        # Gather sorted states
        reranked = torch.zeros_like(derived_states)
        for b in range(B):
            reranked[b] = derived_states[b, sort_indices[b]]

        return reranked, sorted_scores, valid_counts

    def clear_cache(self) -> None:
        """Clear the score cache."""
        self._score_cache.clear()

    @property
    def cache_size(self) -> int:
        """Return number of cached scores."""
        return len(self._score_cache)


class UnificationScorerWrapper:
    """Wrapper that applies KGE scoring to unification results.

    Usage:
        scorer = UnificationScorerWrapper(scorer_module, top_k=5)

        # In evaluation loop:
        derived, counts, _ = engine.get_derived_states(...)
        if scorer:
            derived, counts = scorer.score_and_filter(derived, counts)
    """

    def __init__(
        self,
        scorer: UnificationScorer,
        top_k: Optional[int] = None,
        rerank: bool = True,
    ) -> None:
        """Initialize wrapper.

        Args:
            scorer: UnificationScorer instance.
            top_k: If set, filter to top-k scored states.
            rerank: If True, reorder states by score.
        """
        self.scorer = scorer
        self.top_k = top_k
        self.rerank = rerank

    def score_and_filter(
        self,
        derived_states: Tensor,  # [B, S, A, 3]
        valid_counts: Tensor,  # [B]
    ) -> Tuple[Tensor, Tensor]:
        """Score and optionally filter/rerank derived states.

        Args:
            derived_states: [B, S, A, 3] derived states.
            valid_counts: [B] valid counts.

        Returns:
            Tuple of (processed_states, new_counts)
        """
        B, S = derived_states.shape[0], derived_states.shape[1]

        # Create valid mask
        valid_mask = torch.arange(S, device=self.scorer.device).unsqueeze(0) < valid_counts.unsqueeze(1)

        # Score states
        scores = self.scorer.score_derived_states(derived_states, valid_mask)

        if self.rerank:
            # Sort by score
            scores_masked = scores.masked_fill(~valid_mask, -1e9)
            _, sort_indices = scores_masked.sort(dim=1, descending=True)

            reranked = torch.zeros_like(derived_states)
            for b in range(B):
                reranked[b] = derived_states[b, sort_indices[b]]
            derived_states = reranked

        if self.top_k is not None:
            derived_states, valid_counts = self.scorer.filter_top_k(
                derived_states, scores, self.top_k, valid_counts
            )

        return derived_states, valid_counts


def create_unification_scorer(
    config: Any,
    kge_engine: Optional[Any] = None,
    index_manager: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Optional[UnificationScorer]:
    """Factory function to create unification scorer from config.

    Args:
        config: TrainConfig with unification scoring settings.
        kge_engine: KGE inference engine.
        index_manager: Index manager.
        device: Target device.

    Returns:
        UnificationScorer if enabled, None otherwise.
    """
    enabled = getattr(config, 'unification_scoring', False)
    if not enabled:
        return None

    mode = getattr(config, 'unification_scoring_mode', 'offline')

    scorer = UnificationScorer(
        mode=mode,
        device=device,
        kge_engine=kge_engine,
        index_manager=index_manager,
    )

    # If offline mode, try to load pre-computed scores
    if mode == 'offline':
        dataset = getattr(config, 'dataset', '')
        base_dir = os.path.dirname(os.path.abspath(__file__))
        topk_path = os.path.join(base_dir, 'top_k_scores', 'files', f'kge_top_{dataset}.txt')

        if os.path.exists(topk_path):
            scorer.load_precomputed_scores(topk_path)
        else:
            print(f"[UnificationScorer] Warning: No precomputed scores at {topk_path}, using online mode")
            scorer.mode = 'online'

    return scorer


def create_scorer_wrapper(
    scorer: UnificationScorer,
    config: Any,
) -> UnificationScorerWrapper:
    """Create wrapper for unification scorer.

    Args:
        scorer: UnificationScorer instance.
        config: TrainConfig with settings.

    Returns:
        UnificationScorerWrapper instance.
    """
    top_k = getattr(config, 'unification_top_k', None)

    return UnificationScorerWrapper(
        scorer=scorer,
        top_k=top_k,
        rerank=True,
    )
