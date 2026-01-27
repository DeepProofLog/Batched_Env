"""
Dynamic Programming Prover for KGE link prediction.

This module implements a backward-chaining prover with memoization
for proving queries in knowledge graphs. Designed for torch.compile
compatibility and CUDA graph execution.

Key Components:
    - DPProver: Core prover with backward chaining + memoization
    - ProofTable: GPU-resident hash table for proof caching
    - DPRankingCallback: MRR evaluation using DP proofs
    - BindingEnumerator: Variable query enumeration

Usage:
    from kge_experiments.dp import DPProver, DPRankingCallback

    # Create prover from data handler
    prover = DPProver.from_data_handler(data_handler, device)

    # Prove queries
    proven, depths = prover.prove_batch(queries)

    # Evaluate MRR
    callback = DPRankingCallback(prover, test_queries, known_triples)
    results = callback.evaluate()

Standalone evaluation:
    python -m kge_experiments.dp.runner --dataset countries_s3
"""

from kge_experiments.dp.prover import DPProver, DPProverFast
from kge_experiments.dp.prover_fast_bfs import DPProverFastBFS
from kge_experiments.dp.prover_optimized import DPProverOptimized
from kge_experiments.dp.prover_parallel_bfs import DPProverParallelBFS
from kge_experiments.dp.prover_multiproc import DPProverMultiProc
from kge_experiments.dp.proof_table import ProofTable, ProofTableBatch
from kge_experiments.dp.ranking import (
    DPRankingCallback,
    DPRankingCallbackBatch,
    compute_mrr_with_dp,
)
from kge_experiments.dp.enumerate import (
    BindingEnumerator,
    enumerate_bindings_from_facts,
    enumerate_bindings_batch,
    identify_variables,
)

__all__ = [
    # Core provers
    "DPProver",
    "DPProverFast",
    "DPProverFastBFS",
    "DPProverOptimized",
    "DPProverParallelBFS",
    "DPProverMultiProc",
    # Memoization
    "ProofTable",
    "ProofTableBatch",
    # Ranking / Evaluation
    "DPRankingCallback",
    "DPRankingCallbackBatch",
    "compute_mrr_with_dp",
    # Variable enumeration
    "BindingEnumerator",
    "enumerate_bindings_from_facts",
    "enumerate_bindings_batch",
    "identify_variables",
]
