# KGE Integration Methods (Detailed)

This document explains every Knowledge Graph Embedding (KGE) integration path in this repo, how the data flows, and which knobs control each method. The primary implementation lives under `kge_experiments/kge_module/`, with call sites in `kge_experiments/ppo.py`, `kge_experiments/train.py`, and `kge_experiments/data_handler.py`.

## 0) Common KGE Inference Layer

**Where:** `kge_experiments/kge_module/inference.py`

**What it does**
- Provides a single wrapper (`KGEInference`) that hides backend differences and exposes a `predict_batch(List[str]) -> scores` API.
- Supports two backends: PyTorch models in `kge_experiments/kge_module/pytorch/` and PyKEEN models in `kge_experiments/kge_pykeen/` (if present).
- Resolves checkpoints via `kge_checkpoint_dir` + `kge_run_signature`, or auto-picks the latest run.

**Key settings**
- `kge_inference` (bool): enable KGE inference.
- `kge_engine` (str): `pytorch` or `pykeen`.
- `kge_checkpoint_dir` (str): location of KGE checkpoints.
- `kge_run_signature` (str): specific run to load.
- `kge_scores_file` (str): optional precomputed scores file for some backends.

**Data format**
- All KGE-scored items are string atoms in the form `pred(arg1,arg2)`.
- Conversions from tensor indices to strings are handled by `IndexManager` in callers (see `ppo.py`, `rule_attention.py`, `filter.py`).

## 1) Probabilistic Facts (Dataset Augmentation)

**Where:** `kge_experiments/data_handler.py`

**What it does**
- Loads extra facts from a precomputed KGE top-k file and merges them into the fact base.
- Uses file: `kge_experiments/kge_module/top_k_scores/files/kge_top_<dataset>.txt`.
- Each line is `predicate(head,tail) score rank`.
- Filters by `topk_facts` (rank limit) and/or `topk_facts_threshold` (score threshold).

**How it changes behavior**
- Adds additional “soft” facts to the training KB before indexing, which can expand proof coverage.

**Key settings**
- `prob_facts` (bool): enable probabilistic facts.
- `topk_facts` (int): rank cutoff (None = no cutoff).
- `topk_facts_threshold` (float): minimum score threshold.

## 2) PBRS (Potential-Based Reward Shaping)

**Where:** `kge_experiments/kge_module/pbrs.py`

**What it does**
- Shapes RL rewards using the potential of the *first goal* in the state:
  - `r' = r + gamma * Phi(s') - Phi(s)`
  - `Phi(s) = beta * log(kge_score(first_goal))`
- Supports two modes:
  - `precompute`: loads scores from top-k files once.
  - `runtime`: queries `kge_engine.predict_batch` on-demand.

**How it changes behavior**
- Encourages exploration toward states whose first goal has high KGE confidence.

**Key settings**
- `pbrs_beta` (float): scaling factor for the potential.
- `pbrs_gamma` (float): discount factor for shaping.
- `pbrs_precompute` (bool): use precomputed top-k file.

**Important details**
- Terminal predicates (`True`, `False`, `End`, `Endf`, `Endt`) always have potential 0.
- Cache is maintained per `PBRSModule` instance.

## 3) KGE Rule Attention (Action Selection Bias)

**Where:** `kge_experiments/kge_module/rule_attention.py`

**What it does**
- Scores the **first atom of each derived state** using KGE.
- Adds those scores to action logits:
  - `augmented_logits = action_logits + weight * kge_attention`
- The KGE attention is in log-space and masked with `-inf` for invalid candidates.

**How it changes behavior**
- Biases the policy to pick rules whose conclusions look plausible to the KGE model.

**Key settings**
- `kge_rule_attention` (bool): enable rule attention.
- `kge_rule_attention_weight` (float): scale applied to KGE attention.
- `kge_rule_attention_temperature` (float): temperature for softmax normalization.

**Variants**
- `PrecomputedRuleAttention` can be built from a static score file (same top-k format).

## 4) KGE-Filtered Candidates (Pre-Proof Filtering)

**Where:** `kge_experiments/kge_module/filter.py`

**What it does**
- Before running proofs, it scores all candidate entities for a query with KGE.
- Keeps only the top-k candidates and drops the rest.

**How it changes behavior**
- Reduces evaluation cost when the candidate space is large (e.g., FB15K-237).

**Key settings**
- `kge_filter_candidates` (bool): enable filtering.
- `kge_filter_top_k` (int): how many candidates to keep.

**Notes**
- Filtering is applied before proof attempts. It is not the same as “top-k unification”.

## 5) Neural Bridge (Learned RL + KGE Fusion)

**Where:** `kge_experiments/kge_module/neural_bridge.py`, used in `kge_experiments/ppo.py`

**What it does**
- Learns how to combine RL log-probs and KGE log-scores into a single ranking score.
- Several bridge architectures:
  - **LinearBridge**: single alpha scalar.
  - **GatedBridge**: separate alpha for success/failure.
  - **PerPredicateBridge**: alpha per predicate.
  - **MLPBridge**: learned non-linear fusion.

**Training pipeline**
1) `ppo.train_neural_bridge()` runs a lightweight evaluation pass.
2) During `evaluate()`, RL log-probs, KGE log-scores, and success mask are collected via `NeuralBridgeTrainer.add_validation_batch(...)`.
3) The trainer runs for `neural_bridge_train_epochs` and optimizes MRR (default) or pairwise ranking loss.

**Scoring use**
- During evaluation, if `neural_bridge` is enabled, scores are computed as:
  - `scores = neural_bridge(rl_logprobs, kge_log_scores, success_mask)`

**Key settings**
- `neural_bridge` (bool)
- `neural_bridge_type` (str): `linear`, `gated`, `per_predicate`, `mlp`
- `neural_bridge_init_alpha`, `neural_bridge_init_alpha_success`, `neural_bridge_init_alpha_fail`
- `neural_bridge_train_epochs`
- `neural_bridge_lr`

## 6) Hybrid Scoring (Fixed-Weight Fusion in Evaluation)

**Where:** `kge_experiments/ppo.py` (evaluation path)

**What it does**
- If KGE is enabled but neural bridge is not, a fixed-weight hybrid is used:
  - `score = kge_eval_kge_weight * kge_log_scores`
  - `score += kge_eval_rl_weight` for successful proofs
  - `score -= kge_fail_penalty` for failed proofs

**Key settings**
- `kge_eval_kge_weight` (float)
- `kge_eval_rl_weight` (float)
- `kge_fail_penalty` (float)
- `kge_only_eval` (bool): bypass RL, rank by KGE only.

## 7) KGE-Initialized Embeddings (Weight Transfer)

**Where:** `kge_experiments/kge_module/embed_init.py`

**What it does**
- Copies entity and relation embeddings from a trained KGE model into the policy’s embedder.
- Handles dimension mismatch by truncation or zero-padding.

**Initialization flow (exact steps)**
- The policy is constructed normally with fresh embedding tables.
- `apply_kge_init(config, embedder, kge_inference, data_handler)` checks `kge_init_embeddings`.
- It pulls the KGE model’s entity/relation embeddings from `kge_inference.model`.
- For each policy constant/predicate index, it looks up the matching KGE ID by string.
- It overwrites the policy’s `constant_embedder.embedder.weight` and `predicate_embedder.embedder.weight` rows for matched items.
- If dimensions differ, it truncates or zero-pads the KGE vectors to fit the policy embedding size.

**How it changes behavior**
- Gives the RL policy a better initialization, often improving convergence and generalization.

**Key settings**
- `kge_init_embeddings` (bool)

**Requirements**
- The policy embedder must expose `constant_embedder` and `predicate_embedder`.
- KGE inference must have a loaded model with accessible embeddings.
 - This is not wired into `kge_experiments/train.py` by default; you must call `apply_kge_init(...)` after policy creation.

## 8) KGE Ensembles (Multiple KGE Models)

**Where:** `kge_experiments/kge_module/ensemble.py`

**What it does**
- Loads multiple KGE models and combines their scores.
- Supported combination modes:
  - `mean`
  - `max`
  - `learned` (softmax weights learned during training)

**Optional bridge**
- `EnsembleBridge` combines RL with multiple KGE scores:
  - `score = alpha_rl * rl + sum(alpha_kge[i] * kge[i])`

**Key settings**
- `kge_ensemble` (bool)
- `kge_ensemble_signatures` (str): comma-separated KGE run signatures.
- `kge_ensemble_method` (str): `mean`, `max`, `learned`
- `neural_bridge_init_alpha` (float): initial RL weight for `EnsembleBridge`

## 9) Joint KGE-RL Training (Shared Embeddings)

**Where:** `kge_experiments/kge_module/joint.py`

**What it does**
- Adds a KGE contrastive loss to the PPO objective, training embeddings jointly.
- Builds negative samples by corrupting head/tail entities.
- Uses margin ranking loss:
  - `L = mean(max(0, margin + f(neg) - f(pos)))`

**How it changes behavior**
- Embeddings are tuned for both proof success and KGE plausibility.

**Key settings**
- `kge_joint_training` (bool)
- `kge_joint_lambda` (float): weight of the KGE loss.
- `kge_joint_margin` (float): margin for the contrastive loss.

## 10) Benchmarking (Overhead Tracking)

**Where:** `kge_experiments/kge_module/benchmark.py`

**What it does**
- Collects timing statistics per module or operation.
- Can be used as a context manager or decorator.

**Key settings**
- `kge_benchmark` (bool)

**Typical usage**
```python
from kge_module.benchmark import get_benchmark
bench = get_benchmark()
with bench.time("kge_inference"):
    scores = kge_engine.predict_batch(atoms)
```

## 11) Evaluation Data Flow (How KGE Scores Enter Ranking)

**Where:** `kge_experiments/ppo.py` (`evaluate`)

**Flow summary**
1) Build candidate triples for each query.
2) Score candidates with KGE (`_score_kge_candidates`).
3) Convert to log scores and combine with RL log-probs via:
   - KGE-only (if `kge_only_eval`), or
   - Neural Bridge (if enabled), or
   - Fixed-weight hybrid (otherwise).
4) Compute ranks from the final scores.

This is the central place where KGE affects evaluation metrics (MRR, Hits@K).

## 12) Where Each Method Is Wired In

- **KGE inference build**: `kge_experiments/kge_module/inference.py` (`build_kge_inference`)
- **Probabilistic facts**: `kge_experiments/data_handler.py` (`_load_probabilistic_facts`)
- **PBRS**: `kge_experiments/kge_module/pbrs.py`
- **Rule attention**: `kge_experiments/kge_module/rule_attention.py`
- **Candidate filter**: `kge_experiments/kge_module/filter.py`
- **Neural bridge**: `kge_experiments/kge_module/neural_bridge.py`, training in `kge_experiments/ppo.py`
- **Hybrid scoring / KGE-only eval**: `kge_experiments/ppo.py`
- **Embedding init**: `kge_experiments/kge_module/embed_init.py`
- **Ensemble**: `kge_experiments/kge_module/ensemble.py`
- **Joint training**: `kge_experiments/kge_module/joint.py`
- **Benchmarking**: `kge_experiments/kge_module/benchmark.py`
