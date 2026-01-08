"""
KGE-Filtered Candidates Module.

Pre-filters candidate answers using KGE scores before running proofs.
This is different from Top-K Unification Scorer which filters proof paths.
This filters which answers to even attempt proofs for.

Purpose: Faster evaluation on large datasets (e.g., FB15K-237 with 14k entities).
If KGE already ranks the correct answer in top-100, no need to attempt proofs for all 14k.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor


class KGECandidateFilter:
    """Filter candidates by KGE score before proof attempts.

    Reduces evaluation compute by only attempting proofs for top-k
    KGE-scored candidates per query.

    Attributes:
        top_k: Number of candidates to keep per query.
        kge_inference: KGE inference engine for scoring.
    """

    def __init__(
        self,
        top_k: int = 100,
        kge_inference: Any = None,
        verbose: bool = True,
    ) -> None:
        """Initialize candidate filter.

        Args:
            top_k: Number of candidates to keep per query.
            kge_inference: KGE inference engine (required for scoring).
            verbose: Print filter statistics.
        """
        self.top_k = top_k
        self.kge_inference = kge_inference
        self.verbose = verbose

        if verbose:
            print(f"[KGEFilter] Initialized with top_k={top_k}")

    def filter_candidates(
        self,
        queries: Tensor,  # [B, 3] (predicate, head, tail)
        candidates: Tensor,  # [B, K] candidate entity indices
        is_valid: Tensor,  # [B, K] validity mask
        corruption_mode: str,  # 'head' or 'tail'
        idx2pred: Dict[int, str],
        idx2const: Dict[int, str],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Filter candidates to top-k by KGE score.

        Args:
            queries: [B, 3] query triples (predicate, head, tail).
            candidates: [B, K] candidate entity indices.
            is_valid: [B, K] validity mask for candidates.
            corruption_mode: 'head' or 'tail' corruption.
            idx2pred: Index to predicate string mapping.
            idx2const: Index to constant string mapping.

        Returns:
            Tuple of:
            - filtered_candidates: [B, top_k] filtered candidates
            - filtered_valid: [B, top_k] filtered validity mask
            - original_indices: [B, top_k] indices into original candidates
        """
        if self.kge_inference is None:
            raise ValueError("KGE inference engine required for filtering")

        B, K = candidates.shape
        device = candidates.device

        # If K <= top_k, no filtering needed
        if K <= self.top_k:
            original_indices = torch.arange(K, device=device).unsqueeze(0).expand(B, -1)
            return candidates, is_valid, original_indices

        # Score all candidates with KGE
        kge_scores = self._score_candidates(
            queries, candidates, corruption_mode, idx2pred, idx2const
        )

        # Mask invalid candidates with very low score
        kge_scores = kge_scores.masked_fill(~is_valid, -1e9)

        # Get top-k indices
        _, top_indices = kge_scores.topk(self.top_k, dim=1)  # [B, top_k]

        # Gather filtered candidates
        filtered_candidates = candidates.gather(1, top_indices)
        filtered_valid = is_valid.gather(1, top_indices)

        return filtered_candidates, filtered_valid, top_indices

    def _score_candidates(
        self,
        queries: Tensor,  # [B, 3]
        candidates: Tensor,  # [B, K]
        corruption_mode: str,
        idx2pred: Dict[int, str],
        idx2const: Dict[int, str],
    ) -> Tensor:
        """Score candidates using KGE model.

        Args:
            queries: [B, 3] query triples.
            candidates: [B, K] candidate indices.
            corruption_mode: 'head' or 'tail'.
            idx2pred: Index to predicate mapping.
            idx2const: Index to constant mapping.

        Returns:
            [B, K] KGE scores for each candidate.
        """
        B, K = candidates.shape
        device = candidates.device

        # Build atom strings for KGE scoring
        atoms = []
        for b in range(B):
            pred_idx = queries[b, 0].item()
            head_idx = queries[b, 1].item()
            tail_idx = queries[b, 2].item()

            pred_str = idx2pred.get(pred_idx, f"p{pred_idx}")
            head_str = idx2const.get(head_idx, f"c{head_idx}")
            tail_str = idx2const.get(tail_idx, f"c{tail_idx}")

            for k in range(K):
                cand_idx = candidates[b, k].item()
                cand_str = idx2const.get(cand_idx, f"c{cand_idx}")

                if corruption_mode == 'head':
                    atom_str = f"{pred_str}({cand_str},{tail_str})"
                else:  # tail
                    atom_str = f"{pred_str}({head_str},{cand_str})"

                atoms.append(atom_str)

        # Score with KGE
        scores = self.kge_inference.predict_batch(atoms)  # [B*K]
        scores = scores.view(B, K)

        return scores

    def restore_order(
        self,
        filtered_scores: Tensor,  # [B, top_k]
        original_indices: Tensor,  # [B, top_k]
        original_size: int,
        fill_value: float = -1e9,
    ) -> Tensor:
        """Restore scores to original candidate order.

        Args:
            filtered_scores: [B, top_k] scores for filtered candidates.
            original_indices: [B, top_k] indices into original candidates.
            original_size: Original number of candidates (K).
            fill_value: Value for non-selected candidates.

        Returns:
            [B, original_size] scores in original order.
        """
        B = filtered_scores.shape[0]
        device = filtered_scores.device

        # Create output tensor with fill value
        full_scores = torch.full(
            (B, original_size), fill_value, device=device, dtype=filtered_scores.dtype
        )

        # Scatter filtered scores back to original positions
        full_scores.scatter_(1, original_indices, filtered_scores)

        return full_scores


def create_candidate_filter(
    config: Any,
    kge_inference: Any = None,
) -> Optional[KGECandidateFilter]:
    """Factory function to create candidate filter from config.

    Args:
        config: TrainConfig with filter settings.
        kge_inference: KGE inference engine.

    Returns:
        KGECandidateFilter if enabled, None otherwise.
    """
    enabled = getattr(config, 'kge_filter_candidates', False)
    if not enabled:
        return None

    top_k = getattr(config, 'kge_filter_top_k', 100)
    verbose = getattr(config, 'verbose', True)

    return KGECandidateFilter(
        top_k=top_k,
        kge_inference=kge_inference,
        verbose=verbose,
    )
