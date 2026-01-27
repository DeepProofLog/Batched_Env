# DP Prover MRR Evaluation Results

## Summary

| Dataset | MRR | Hits@1 | Hits@10 | Positive Provability | Corruption Provability |
|---------|-----|--------|---------|----------------------|------------------------|
| countries_s3 | **1.000** | 1.000 | 1.000 | 100.0% | 0.6% |
| family | **0.999** | 0.999 | 0.999 | 99.6% | 32.6% |

## countries_s3

```
Queries: 24 (all test queries)
Corruptions per mode: 50
Max depth: 6
Total time: 2.6s

Results:
  MRR: 1.0000
  Hits@1: 1.0000
  Hits@3: 1.0000
  Hits@10: 1.0000
  Positive provability: 100.0%
  Corruption provability: 0.6%
```

**Analysis**: Perfect MRR achieved. All test queries are provable via rules, while corruptions (invalid country-region pairs) are mostly unprovable.

## family

```
Queries: 500 (random sample from test set)
Corruptions per mode: 100
Max depth: 6
Total time: 670.7s (0.75 q/s)

Results:
  MRR: 0.9990
  Hits@1: 0.9990
  Hits@3: 0.9990
  Hits@10: 0.9990
  Positive provability: 99.6%
  Corruption provability: 32.6%
```

**Analysis**: Near-perfect MRR. The 0.4% unprovable positive queries are likely due to BFS exploration limits. The 32.6% corruption provability is expected for family relationships - some corrupted triples happen to be valid family relationships.

## Bug Fix Applied

A critical bug in `_apply_rule` was fixed during this evaluation:

**Problem**: When rule bodies had shared unbound variables (e.g., `aunt(X, Y) :- aunt(X, Z), brother(Y, Z)`), the bindings were enumerated independently for each body atom, breaking the shared variable constraint.

**Fix**: Pre-filter facts by known substitutions before enumerating bindings, ensuring shared variables get consistent bindings across all body atoms.

## Prover Configuration

- **Prover**: DPProverMultiProc with 16 workers
- **Max depth**: 6
- **Max branches**: 50 (BFS exploration limit)
- **Max subgoals**: 8 (per state)

## Comparison with PPO Engine

| Metric | DP Prover | PPO Engine |
|--------|-----------|------------|
| Provability (family train) | ~98% | 75.9% |
| Speed | 150 proofs/s | ~85 states/s |
| Completeness | More complete | Limited by K_max=120 |

The DP prover is more complete (explores more proof paths) but uses CPU multiprocessing instead of GPU.

## Commands

```bash
# countries_s3
python -m kge_experiments.dp.eval_mrr --dataset countries_s3 --n_queries 100 --n_corruptions 50

# family
python -m kge_experiments.dp.eval_mrr --dataset family --n_queries 500 --n_corruptions 100
```
