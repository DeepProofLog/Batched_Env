import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Optional, Set

@dataclass
class TrainConfig:
    """Configuration for training (unified for runner and compiled scripts)."""
    
    # Dataset / Paths
    dataset: str = "countries_s3"
    data_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    train_file: str = "train.txt"
    valid_file: str = "valid.txt"
    test_file: str = "test.txt"
    rules_file: str = "rules.txt"
    facts_file: str = "train.txt"
    
    # Depths (can be specific integers or None)
    train_depth: Any = None
    valid_depth: Any = None
    test_depth: Any = None
    
    # Sample counts (None means use all)
    n_train_queries: Optional[int] = None
    n_eval_queries: Optional[int] = 100
    n_test_queries: Optional[int] = 100
    
    # Environment / Logic
    padding_atoms: int = 6
    padding_states: int = 64
    eval_padding_states: int = 120  # Optimized: 16 gives ~0.87 ms/candidate for WN18RR
    eval_max_depth: int = 14  # Optimized: 14 steps sufficient for most proofs
    max_steps: int = 20 # max_depth in runner
    use_exact_memory: bool = True
    memory_pruning: bool = True
    skip_unary_actions: bool = True
    end_proof_action: bool = True
    reward_type: int = 0
    max_total_vars: int = 1000
    max_fact_pairs_cap: Optional[int] = None  # Cap for large predicates (auto-set to eval_padding_states if None)
    eval_batch_size: int = 75  # Optimized batch size for evaluation (75 for best speed)
    sample_deterministic_per_env: bool = False  # False for fullgraph compilation compatibility
    
    # Model Architecture
    algorithm_type: str = "ppo"  # Algorithm to use (ppo, etc.)
    model_name: str = "PPO"
    atom_embedding_size: int = 64
    state_embedding_size: int = 64 # derived
    hidden_dim: int = 256
    num_layers: int = 8
    dropout_prob: float = 0.0
    use_l2_norm: bool = True
    sqrt_scale: bool = False
    temperature: float = 1.0
    atom_embedder: str = 'transe'
    state_embedder: str = 'mean'
    
    # PPO / Training Hyperparams
    n_envs: int = 3
    n_steps: int = 20
    n_epochs: int = 4
    batch_size: int = 20
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    ent_coef: float = 0.2
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    total_timesteps: int = 120
    
    # Sampling / Corruption
    negative_ratio: float = 1.0 # train_neg_ratio
    eval_neg_samples: Optional[int] = 4
    test_neg_samples: Optional[int] = 10  # Default to non-exhaustive evaluation
    n_corruptions: Optional[int] = 10 # test_neg_samples alias
    corruption_scheme: List[str] = field(default_factory=lambda: ['head', 'tail'])
    sampler_default_mode: str = "both"
    
    # LR Decay
    lr_decay: bool = False
    lr_init_value: float = 3e-4
    lr_final_value: float = 1e-6
    lr_start: float = 0.0
    lr_end: float = 1.0
    lr_transform: str = 'linear'
    
    # Entropy Decay
    ent_coef_decay: bool = False
    ent_coef_init_value: float = 0.01
    ent_coef_final_value: float = 0.01
    ent_coef_start: float = 0.0
    ent_coef_end: float = 1.0
    ent_coef_transform: str = 'linear'
    
    # Model Saving / Logging
    save_model: bool = True
    load_model: Any = False # False or 'last_epoch' or path
    restore_best: bool = True # restore_best_val_model
    load_best_metric: str = 'eval'
    models_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    eval_freq: int = 0
    eval_best_metric: str = 'mrr'
    ranking_tie_seed: int = 0
    use_logger: bool = False
    logger_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
    run_signature: str = "compiled_run"

    # KGE inference (evaluation-time fusion)
    kge_inference: bool = False
    kge_inference_success: bool = True
    kge_engine: Optional[str] = None
    kge_checkpoint_dir: Optional[str] = None
    kge_run_signature: Optional[str] = None
    kge_scores_file: Optional[str] = None
    kge_eval_kge_weight: float = 2.0
    kge_eval_rl_weight: float = 1.0
    kge_fail_penalty: float = 100.0  # Penalty for failed proofs in hybrid mode
    kge_only_eval: bool = False  # If True, use KGE-only scoring at test time (matches paper)

    # KGE Integration: Probabilistic Facts
    prob_facts: bool = False  # Enable loading probabilistic facts from KGE top-k file
    prob_facts_topk: Optional[int] = None  # Top-K facts per (predicate, role, anchor)
    prob_facts_threshold: Optional[float] = None  # Minimum score threshold for facts

    # KGE Integration: PBRS (Potential-Based Reward Shaping)
    pbrs_beta: float = 0.0  # PBRS weight (0 = disabled). Phi(s) = beta * log(KGE_score)
    pbrs_gamma: float = 0.99  # Discount factor for PBRS
    pbrs_precompute: bool = True  # Pre-compute potentials (faster) vs runtime scoring

    # KGE Integration: Neural Bridge (learned RL+KGE fusion)
    neural_bridge: bool = False  # Enable learned combination of RL and KGE logprobs
    neural_bridge_type: str = 'linear'  # 'linear', 'gated', or 'mlp'
    neural_bridge_init_alpha: float = 0.5  # Initial alpha value (α*RL + (1-α)*KGE)
    neural_bridge_init_alpha_success: float = 0.7  # Gated bridge: alpha for successful proofs
    neural_bridge_init_alpha_fail: float = 0.2  # Gated bridge: alpha for failed proofs
    neural_bridge_train_epochs: int = 100  # Epochs to train bridge on validation
    neural_bridge_lr: float = 0.01  # Learning rate for bridge training
    neural_bridge_hidden_dim: int = 32  # MLP bridge hidden dimension

    # KGE Integration: Predicate-Aware Scoring
    predicate_aware_scoring: bool = False  # Use different weights for symmetric vs chain predicates
    predicate_aware_symmetric_weight: float = 0.7  # RL weight for symmetric predicates (high = trust RL)
    predicate_aware_chain_weight: float = 0.0  # RL weight for chain-only predicates (0 = pure KGE)

    # KGE Integration: KGE-Filtered Candidates (query-level filtering)
    kge_filter_candidates: bool = False  # Pre-filter candidates by KGE score before proofs
    kge_filter_top_k: int = 100  # Keep top-k candidates per query by KGE score

    # KGE Integration: KGE-Initialized Embeddings
    kge_init_embeddings: bool = False  # Initialize policy embeddings from KGE model

    # KGE Integration: Ensemble KGE Models
    kge_ensemble: bool = False  # Use ensemble of KGE models
    kge_ensemble_signatures: Optional[str] = None  # Comma-separated run signatures
    kge_ensemble_method: str = 'mean'  # 'mean', 'max', or 'learned'

    # KGE Integration: Joint KGE-RL Training
    kge_joint_training: bool = False  # Fine-tune KGE embeddings alongside RL
    kge_joint_lambda: float = 0.1  # Weight for KGE contrastive loss in total loss
    kge_joint_margin: float = 1.0  # Margin for contrastive loss

    # KGE Integration: Unification Scoring
    unification_scoring: bool = False  # Enable KGE scoring of derived states
    unification_scoring_mode: str = 'offline'  # 'offline' (pre-computed) or 'online' (runtime)
    unification_top_k: Optional[int] = None  # Filter to top-k scored states

    # KGE Integration: Rule Attention
    kge_rule_attention: bool = False  # Use KGE scores to weight rule selection
    kge_rule_attention_weight: float = 0.5  # Weight for adding KGE attention to logits
    kge_rule_attention_temperature: float = 1.0  # Temperature for softmax (lower = sharper)

    # KGE Benchmarking
    kge_benchmark: bool = False  # Enable timing collection for KGE modules

    # Misc
    seed: int = 42
    seed_run_i: int = 42 # specific run seed
    device: str = "cpu"
    verbose: bool = True
    parity: bool = False
    profile: bool = False
    use_callbacks: bool = True  # Enable callbacks in run_experiment

    # Callback control (individual toggles)
    use_metrics_callback: bool = True
    use_ranking_callback: bool = True
    use_checkpoint_callback: bool = True
    use_annealing_callback: bool = True
    
    # Query Filtering
    filter_queries_by_rules: bool = True

    def __post_init__(self):
        # Default corruption scheme logic
        if self.corruption_scheme is None:
             self.corruption_scheme = ['head', 'tail']
             
        if 'countries' in self.dataset or 'ablation' in self.dataset:
             self.corruption_scheme = ['tail']
             
        # Resolve depths if needed (simple check)
        if self.train_depth and self.train_file == "train.txt":
            self.train_file = "train_depths.txt"
        if self.valid_depth and self.valid_file == "valid.txt":
            self.valid_file = "valid_depths.txt"
        if self.test_depth and self.test_file == "test.txt":
             self.test_file = "test_depths.txt"

        # Derived embedding sizes
        if self.state_embedding_size is None or self.state_embedding_size == 64: # if default
             if self.state_embedder != "concat":
                 self.state_embedding_size = self.atom_embedding_size
             else:
                 self.state_embedding_size = self.atom_embedding_size * self.padding_atoms

        # Aliases / Compatibility
        if self.n_corruptions is None:
            self.n_corruptions = 10 # default

        # Dynamic Run Signature
        if self.run_signature in [None, "run_v1", "compiled_run"]:
            import datetime
            date_str = datetime.datetime.now().strftime("%Y%m%d")
            self.run_signature = f"{self.dataset}-{self.seed}-{date_str}"
