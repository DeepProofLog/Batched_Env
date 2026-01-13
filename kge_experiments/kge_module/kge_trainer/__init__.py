"""KGE Integration Module.

Contains all KGE (Knowledge Graph Embedding) integration components:
- inference: KGE model loading and scoring
- pbrs: Potential-Based Reward Shaping
- neural_bridge: Learned RL+KGE fusion
- embed_init: KGE-initialized embeddings
- ensemble: Multi-KGE model ensemble
- joint: Joint KGE-RL training
- filter: KGE-filtered candidates
- pytorch: PyTorch KGE model implementations
"""
from __future__ import annotations

# Core inference
from kge_module.inference import (
    KGEInference,
    build_kge_inference,
    current_backend,
    default_checkpoint_dir,
    find_latest_run,
    normalize_backend,
)

# PBRS (Potential-Based Reward Shaping)
from kge_module.pbrs import (
    PBRSModule,
    create_pbrs_module,
)

# Neural Bridge (Learned RL+KGE fusion)
from kge_module.neural_bridge import (
    LinearBridge,
    GatedBridge,
    PerPredicateBridge,
    PredicateTypeBridge,
    MLPBridge,
    NeuralBridgeTrainer,
    create_neural_bridge,
    create_bridge_trainer,
    create_predicate_type_bridge,
    identify_symmetric_predicates,
)

# Embedding initialization
from kge_module.embed_init import (
    initialize_embeddings_from_kge,
    apply_kge_init,
)

# Ensemble
from kge_module.ensemble import (
    KGEEnsemble,
    EnsembleBridge,
    load_ensemble_models,
    create_kge_ensemble,
    create_ensemble_bridge,
)

# Joint training
from kge_module.joint import (
    KGEContrastiveLoss,
    KGEJointTrainer,
    create_negative_samples,
    extract_triples_from_rollout,
    create_kge_joint_trainer,
)

# Candidate filtering
from kge_module.filter import (
    KGECandidateFilter,
    create_candidate_filter,
)

# Rule attention (KGE-guided action selection)
from kge_module.rule_attention import (
    KGERuleAttention,
    PrecomputedRuleAttention,
    create_rule_attention,
)

# Benchmarking
from kge_module.benchmark import (
    KGEBenchmark,
    get_benchmark,
    set_benchmark,
    create_benchmark,
)

__all__ = [
    # Inference
    "KGEInference",
    "build_kge_inference",
    "current_backend",
    "default_checkpoint_dir",
    "find_latest_run",
    "normalize_backend",
    # PBRS
    "PBRSModule",
    "create_pbrs_module",
    # Neural Bridge
    "LinearBridge",
    "GatedBridge",
    "PerPredicateBridge",
    "PredicateTypeBridge",
    "MLPBridge",
    "NeuralBridgeTrainer",
    "create_neural_bridge",
    "create_bridge_trainer",
    "create_predicate_type_bridge",
    "identify_symmetric_predicates",
    # Embedding init
    "initialize_embeddings_from_kge",
    "apply_kge_init",
    # Ensemble
    "KGEEnsemble",
    "EnsembleBridge",
    "load_ensemble_models",
    "create_kge_ensemble",
    "create_ensemble_bridge",
    # Joint
    "KGEContrastiveLoss",
    "KGEJointTrainer",
    "create_negative_samples",
    "extract_triples_from_rollout",
    "create_kge_joint_trainer",
    # Filter
    "KGECandidateFilter",
    "create_candidate_filter",
    # Rule Attention
    "KGERuleAttention",
    "PrecomputedRuleAttention",
    "create_rule_attention",
    # Benchmarking
    "KGEBenchmark",
    "get_benchmark",
    "set_benchmark",
    "create_benchmark",
]
