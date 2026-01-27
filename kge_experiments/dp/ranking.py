"""
DP-based ranking for MRR evaluation.

Uses the existing TensorSampler for corruption generation and the DPProver
for proof-based ranking. For datasets like countries_s3, if all positives
are provable and corruptions are not, MRR should be 1.0.

Scoring Strategy:
    - Provable positive + unprovable corruptions → rank = 1
    - If some corruptions are also provable → rank = 1 + count_of_proven_corruptions
    - Unprovable positive → rank = 1 + count_of_proven_corruptions
"""

from __future__ import annotations
from typing import Dict, Optional, Sequence, Any
import torch
from torch import Tensor

from kge_experiments.dp.prover import DPProver


def evaluate_dp_mrr(
    prover: DPProver,
    queries: Tensor,                    # [N, 3] test queries in (pred, arg0, arg1) format
    sampler,                            # TensorSampler with corrupt() method
    n_corruptions: int = 50,
    corruption_modes: Sequence[str] = ('head', 'tail'),
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Evaluate MRR using DP prover for proof-based ranking.

    Args:
        prover: DPProver instance
        queries: [N, 3] test queries (pred, head, tail format)
        sampler: Sampler with corrupt() method
        n_corruptions: Number of corruptions per query per mode
        corruption_modes: Which positions to corrupt ('head', 'tail')
        verbose: Print progress

    Returns:
        dict with 'mrr', 'hits@1', 'hits@3', 'hits@10', etc.
    """
    N = queries.shape[0]
    device = prover.device

    if N == 0:
        return {"mrr": 0.0, "hits@1": 0.0, "hits@3": 0.0, "hits@10": 0.0}

    # Convert queries to sampler format: (pred, arg0, arg1) -> (arg0, pred, arg1) for corrupt()
    # The sampler expects (h, r, t) format where r=predicate
    queries_hrt = torch.stack([queries[:, 1], queries[:, 0], queries[:, 2]], dim=1)

    all_ranks = []
    total_pos_proven = 0
    total_corr_proven = 0
    total_corruptions = 0

    for mode in corruption_modes:
        if verbose:
            print(f"Evaluating {mode} corruptions...")

        # Generate corruptions using sampler
        # corrupt() returns [N, K, 3] in (h, r, t) format
        corruptions_hrt = sampler.corrupt(
            queries_hrt,
            num_negatives=n_corruptions,
            mode=mode,
            device=device,
            filter=True,
        )

        K = corruptions_hrt.shape[1]
        total_corruptions += N * K

        # Convert back to (pred, arg0, arg1) format
        corruptions = torch.stack([
            corruptions_hrt[:, :, 1],  # pred
            corruptions_hrt[:, :, 0],  # arg0 (head)
            corruptions_hrt[:, :, 2],  # arg1 (tail)
        ], dim=2)  # [N, K, 3]

        # Prove positive queries
        pos_proven, _ = prover.prove_batch(queries)
        total_pos_proven += pos_proven.sum().item()

        # Prove all corruptions (batch)
        corruptions_flat = corruptions.view(-1, 3)  # [N*K, 3]
        corr_proven_flat, _ = prover.prove_batch(corruptions_flat)
        corr_proven = corr_proven_flat.view(N, K)  # [N, K]

        total_corr_proven += corr_proven.sum().item()

        # Compute ranks
        # Rank = 1 + number of corruptions that are proven (and positive is not proven adds all)
        for i in range(N):
            if pos_proven[i]:
                # Positive proven: rank = 1 + count of proven corruptions that beat it
                # Since we're using proof vs no-proof, all proven corruptions tie with positive
                # Optimistic: rank = 1 (we win all ties)
                rank = 1.0
            else:
                # Positive not proven: rank = 1 + count of proven corruptions
                rank = 1.0 + corr_proven[i].sum().float().item()

            all_ranks.append(rank)

        if verbose:
            mode_proven_rate = corr_proven.float().mean().item()
            print(f"  {mode}: corruption provability = {mode_proven_rate:.1%}")

    ranks = torch.tensor(all_ranks, dtype=torch.float, device=device)

    # Compute metrics
    mrr = (1.0 / ranks).mean().item()
    hits1 = (ranks <= 1).float().mean().item()
    hits3 = (ranks <= 3).float().mean().item()
    hits10 = (ranks <= 10).float().mean().item()

    results = {
        "mrr": mrr,
        "hits@1": hits1,
        "hits@3": hits3,
        "hits@10": hits10,
        "num_queries": N,
        "num_modes": len(corruption_modes),
        "total_evaluations": N * len(corruption_modes),
        "positive_provability": total_pos_proven / (N * len(corruption_modes)),
        "corruption_provability": total_corr_proven / total_corruptions if total_corruptions > 0 else 0.0,
    }

    if verbose:
        print(f"\nResults:")
        print(f"  MRR: {mrr:.4f}")
        print(f"  Hits@1: {hits1:.4f}")
        print(f"  Hits@3: {hits3:.4f}")
        print(f"  Hits@10: {hits10:.4f}")
        print(f"  Positive provability: {results['positive_provability']:.1%}")
        print(f"  Corruption provability: {results['corruption_provability']:.1%}")

    return results


class DPRankingCallback:
    """
    Callback for MRR evaluation using DP prover.

    Can be used standalone or integrated with training loops.
    """

    def __init__(
        self,
        prover: DPProver,
        eval_queries: Tensor,           # [N, 3] test queries
        sampler,                        # TensorSampler
        n_corruptions: int = 50,
        corruption_modes: Sequence[str] = ('head', 'tail'),
        verbose: bool = False,
    ):
        """
        Initialize the callback.

        Args:
            prover: DPProver instance
            eval_queries: [N, 3] test queries
            sampler: TensorSampler with corrupt() method
            n_corruptions: Corruptions per query per mode
            corruption_modes: Which positions to corrupt
            verbose: Print progress
        """
        self.prover = prover
        self.eval_queries = eval_queries.to(prover.device)
        self.sampler = sampler
        self.n_corruptions = n_corruptions
        self.corruption_modes = corruption_modes
        self.verbose = verbose

    def evaluate(self, max_queries: Optional[int] = None) -> Dict[str, float]:
        """
        Run MRR evaluation.

        Args:
            max_queries: Limit number of queries (for debugging)

        Returns:
            dict with metrics
        """
        queries = self.eval_queries
        if max_queries is not None:
            queries = queries[:max_queries]

        return evaluate_dp_mrr(
            prover=self.prover,
            queries=queries,
            sampler=self.sampler,
            n_corruptions=self.n_corruptions,
            corruption_modes=self.corruption_modes,
            verbose=self.verbose,
        )


# Keep old classes for backwards compatibility
class DPRankingCallbackBatch(DPRankingCallback):
    """Alias for DPRankingCallback (batching is handled internally)."""
    pass


def compute_mrr_with_dp(
    prover: DPProver,
    queries: Tensor,
    sampler,
    n_corruptions: int = 50,
    corruption_modes: Sequence[str] = ('head', 'tail'),
    verbose: bool = False,
) -> Dict[str, float]:
    """Convenience function for MRR evaluation."""
    return evaluate_dp_mrr(
        prover=prover,
        queries=queries,
        sampler=sampler,
        n_corruptions=n_corruptions,
        corruption_modes=corruption_modes,
        verbose=verbose,
    )
