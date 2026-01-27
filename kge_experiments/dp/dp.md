# DP Prover - Backward Chaining BFS Prover for KGE Link Prediction

## Overview

The DP (Dynamic Programming) Prover implements backward-chaining proof search with BFS exploration for evaluating knowledge graph link prediction. It determines whether a query triple can be **derived** from rules and facts, enabling symbolic MRR evaluation.

## Key Concept: Derivability vs Existence

The prover tests whether a query can be **derived via rules**, not just whether it exists as a fact. When proving query `Q`, `Q` itself is **excluded** from fact lookups to ensure we're testing logical derivability.

```
Query: aunt(alice, bob)
Question: Can this be DERIVED using rules + other facts?
NOT: Does aunt(alice, bob) exist in the fact database?
```

## Quick Start

```bash
# Evaluate MRR on countries_s3
python -m kge_experiments.dp.eval_mrr --dataset countries_s3

# Evaluate on family with custom settings
python -m kge_experiments.dp.eval_mrr --dataset family --n_queries 500 --n_corruptions 100 --max_depth 6
```

## Architecture

```
kge_experiments/dp/
├── eval_mrr.py              # Main evaluation script
├── prover_parallel_bfs.py   # Single-process BFS prover (GPU)
├── prover_multiproc.py      # Multi-process prover (CPU, faster)
├── ranking.py               # MRR computation with corruption ranking
├── proof_table.py           # Memoization table for caching proofs
├── enumerate.py             # Variable binding enumeration
├── prover.py                # Original recursive prover
└── docs/
    ├── dp_prover_results.md # Evaluation results
    └── ppo_engine_analysis.md # Comparison with PPO engine
```

## Provers

### DPProverMultiProc (Recommended)
- Uses Python multiprocessing for parallel query proving
- ~110 queries/second with 16 workers
- Best for large-scale evaluation

```python
from kge_experiments.dp import DPProverMultiProc

prover = DPProverMultiProc.from_index_manager(
    im,
    max_depth=6,
    n_workers=16,
    max_branches=50,
    max_subgoals=8
)
proven, depths = prover.prove_batch(queries)
```

### DPProverParallelBFS
- Single-process GPU-based prover
- ~2 queries/second
- Useful for debugging or small batches

```python
from kge_experiments.dp import DPProverParallelBFS

prover = DPProverParallelBFS.from_index_manager(im, max_depth=6)
proven, depths = prover.prove_batch(queries)
```

## Algorithm: Backward Chaining BFS

```
prove(query):
    1. Exclude query from fact lookups (test derivability, not existence)

    2. For each rule where head predicate matches query:
        a. Unify query with rule head → get substitution θ
        b. Apply θ to rule body → get subgoals
        c. If body is empty → PROVEN (depth 1)
        d. Otherwise, add subgoals to BFS queue

    3. BFS loop:
        a. Pop state (subgoals, depth) from queue
        b. Check each subgoal against facts (excluding original query)
        c. If ALL subgoals are facts → PROVEN
        d. Otherwise, expand first unproven subgoal with rules
        e. Add new states to queue (with depth + 1)

    4. If queue exhausted or limits reached → NOT PROVEN
```

## Handling Shared Variables

Rules often have shared variables across body atoms:

```prolog
aunt(X, Y) :- aunt(X, Z), brother(Y, Z).
```

The prover correctly handles this by:
1. Finding bindings that satisfy ALL body atoms simultaneously
2. Pre-filtering facts by known substitutions
3. Validating bindings against all constraining atoms

**Bug fixed**: Previously, body atoms were grounded independently, breaking shared variable constraints.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_depth` | 6 | Maximum proof depth |
| `max_branches` | 50 | Maximum BFS states explored |
| `max_subgoals` | 8 | Maximum subgoals per state |
| `n_workers` | CPU count | Worker processes (multiproc only) |

## MRR Evaluation

The evaluation workflow:

```
1. Load test queries
2. For each query:
   a. Prove the positive query
   b. Generate corruptions (replace head or tail entity)
   c. Prove all corruptions
   d. Rank: positive provable + corruption not provable → rank 1
3. Compute MRR = mean(1/rank)
```

## Results

| Dataset | MRR | Hits@1 | Positive Provability |
|---------|-----|--------|---------------------|
| countries_s3 | 1.000 | 1.000 | 100% |
| family | 0.999 | 0.999 | 99.6% |

## Comparison with PPO Engine

| Aspect | DP Prover | PPO Engine |
|--------|-----------|------------|
| Completeness | Higher (~98%) | Limited (76%) |
| Speed | ~110 q/s (multiproc) | ~85 states/s |
| Device | CPU (multiproc) | GPU |
| Limits | Configurable | Hard-coded (K_max=120) |

The DP prover explores more proof paths but uses CPU. The PPO engine is GPU-optimized but has hard limits on exploration.

## Usage in Training

The prover can be used for:
1. **Evaluation**: Compute MRR during/after training
2. **Reward shaping**: Provide proof-based rewards
3. **Filtering**: Identify provable vs unprovable queries

```python
from kge_experiments.dp import DPProverMultiProc
from kge_experiments.dp.ranking import evaluate_dp_mrr

# Create prover
prover = DPProverMultiProc.from_index_manager(im, max_depth=6)

# Evaluate
results = evaluate_dp_mrr(
    prover=prover,
    queries=test_queries,
    sampler=sampler,
    n_corruptions=50,
    corruption_modes=["head", "tail"]
)
print(f"MRR: {results['mrr']:.4f}")
```

## Files

| File | Purpose |
|------|---------|
| `eval_mrr.py` | CLI for MRR evaluation |
| `prover_multiproc.py` | Fast multiprocessing prover |
| `prover_parallel_bfs.py` | Single-process GPU prover |
| `ranking.py` | MRR computation utilities |
| `proof_table.py` | Tensor-based proof caching |
| `enumerate.py` | Variable binding enumeration |
| `__init__.py` | Public API exports |
