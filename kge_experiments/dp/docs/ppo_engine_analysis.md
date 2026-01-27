# PPO Unification Engine vs DP Prover Analysis

## Executive Summary

The PPO `UnificationEngineVectorized` and the DP BFS provers have different semantics, leading to different provability results. The PPO engine prioritizes **speed** (batched GPU operations) at the cost of **completeness** (hard limits on exploration). The DP prover prioritizes **completeness** (explores more paths) at the cost of **speed**.

## PPO Engine Limitations

### Hard Limits

| Parameter | Value | Impact |
|-----------|-------|--------|
| `K_max` | 120 | Max derived states per expansion step |
| `max_fact_pairs` | 2543 | Max facts matched per predicate |
| `max_rule_pairs` | 100 | Max rules matched per predicate |
| `M_max` | 30 | Max atoms per state |

### Key Constraints

1. **Fixed Tensor Shapes**: All operations use fixed `[B, K_max, M_max, 3]` tensors. Derivations beyond K_max are silently dropped.

2. **Truncated Fact Matching**: For predicates with >2543 facts (e.g., `hypernym` in wn18rr has 34,796), only first 2543 are considered.

3. **No Backtracking**: BFS explores all K_max candidates in parallel, but doesn't backtrack if all fail. A query is unprovable if no path succeeds within the limits.

4. **Deterministic Ordering**: Facts/rules are processed in index order, not by relevance. This can miss proofs if the "right" fact is beyond the limit.

### Implications for Provability

On family dataset:
- **Depth file (PPO semantics)**: 75.9% provable (15,057/19,845)
- **DP BFS prover**: ~98% provable
- **Gap**: ~22% of queries are provable via BFS but not within PPO's limits

The "unprovable" queries in PPO are not logically unprovable—they just can't be found within the engine's constraints.

## Speed Comparison

### Family Dataset (50 queries, depth 6)

| Prover | Time | Speed | Provability |
|--------|------|-------|-------------|
| DPProverParallelBFS (1 proc) | 26.2s | 1.9 q/s | 98% |
| DPProverMultiProc (16 proc) | 0.64s | 78 q/s | 98% |
| PPO engine (1 step) | 1.18s | 85 states/s | - |

### Analysis

1. **Single-process BFS**: ~2 q/s - too slow for large-scale evaluation
2. **Multiprocessing BFS**: ~78 q/s - 40x faster, usable for evaluation
3. **PPO engine**: 85 states/s per expansion step, but requires multiple steps for full proof

### Bottlenecks

**DP BFS Prover:**
- Python `for` loops in BFS
- `deque` operations not GPU-accelerated
- Binding enumeration creates many candidates

**PPO Engine:**
- Large tensor allocations (`[B, 120, 30, 3]`)
- Memory bandwidth for fact/rule gathering
- Compilation overhead for first call

## Semantic Differences

### Query Exclusion

Both exclude the query being proved from fact lookups (to test derivability, not existence):

```python
# PPO engine
excluded_queries = query.unsqueeze(0).unsqueeze(0)  # [B, 1, 3]
derived = engine.get_derived_states_compiled(states, vars, excluded_queries)

# DP BFS prover
excluded = (pred, arg0, arg1)
is_fact = self._check_fact(p, a0, a1, excluded=excluded)
```

### Exploration Strategy

| Aspect | PPO Engine | DP BFS Prover |
|--------|------------|---------------|
| Strategy | Parallel BFS with fixed width | Sequential BFS with variable width |
| Max branches | K_max = 120 | max_branches = 50 |
| Binding limit | Implicit (tensor truncation) | Explicit ([:3] or [:8]) |
| Rule limit | max_rule_pairs = 100 | All rules tried |

## Recommendations

### For MRR Evaluation

Use **DPProverMultiProc** with these settings:
- `max_depth=6` (matches PPO training)
- `n_workers=16` (or CPU count)
- Accept ~98% provability (more complete than PPO)

### For Matching PPO Semantics

If exact match with depth file is needed:
1. Use PPO engine directly via `prover_engine.py` (needs stability fixes)
2. Or add hard limits to BFS prover matching PPO's K_max

### For Scaling to Larger Datasets (wn18rr)

The bottleneck is binding enumeration with large fact counts:
- wn18rr has 34,796 facts for `hypernym` predicate
- BFS enumerates too many candidates
- Options:
  - Use depth 1-2 only (limited derivation)
  - Implement fact sampling (random subset)
  - Use PPO engine (has built-in truncation)

## Conclusion

The DP BFS prover is **more complete** (proves more queries) but **slower** than PPO's engine. For evaluation purposes, the multiprocessing version at 78 q/s is practical for datasets like family. For wn18rr, the current approach doesn't scale—either use PPO engine or accept very limited depth.
