import os
from dataclasses import dataclass, field
from typing import List, Any, Optional

# Default KGE run signatures per dataset (for kge_inference)
KGE_RUN_SIGNATURES = {
    'wn18rr': 'torch_wn18rr_RotatE_1024_20260107_125531_s42',
    'family': 'torch_family_RotatE_1024_20260107_124531_s42',
    'fb15k237': 'torch_fb15k237_TuckER_512_20260111_002222_s42',
    'pharmkg_full': 'torch_pharmkg_full_ComplEx_1024_20260111_054518_s42',
    'umls': 'torch_umls_ComplEx_1024_20260110_223751_s42',
    'nations': 'torch_nations_TuckER_512_20260110_224506_s42',
}

# Default padding_states per dataset
PADDING_STATES_MAP = {
    'countries_s3': 20, 'countries_s2': 20, 'countries_s1': 20,
    'family': 130, 'wn18rr': 262, 'fb15k237': 358,
    'nations': 64, 'umls': 64, 'pharmkg_full': 358,
}

# Default max_fact_pairs_cap per dataset (for large predicates)
MAX_FACT_PAIRS_CAP_MAP = {
    'wn18rr': 8000,      # hypernym has 35k facts, cap to 1000 for 7x speedup, but mind the limitations
    'fb15k237': 8000,    # similar issue expected
    'pharmkg_full': 8000,
    # family, countries: no cap needed (max ~2.5k facts per predicate)
}

@dataclass
class TrainConfig:
    """Configuration for training (unified for runner and compiled scripts)."""
    
    # Dataset / Paths
    dataset: str = "family"
    data_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    train_file: str = "train.txt"
    valid_file: str = "valid.txt"
    test_file: str = "test.txt"
    rules_file: str = "rules.txt"
    facts_file: str = "train.txt"
    
    # Depths (can be specific integers or None)
    train_depth: Any = field(default_factory=lambda: {1,2,3,4,5,6,7,8,9,10,11,12,13,14})
    valid_depth: Any = None
    test_depth: Any = None
    
    # Sample counts (None means use all)
    n_train_queries: Optional[int] = None
    n_eval_queries: Optional[int] = 200  # Increased from 20 to reduce val-test gap
    n_test_queries: Optional[int] = 500
    
    # Environment / Logic
    padding_atoms: int = 6
    padding_states: int = 120
    eval_padding_states: int = 120
    eval_max_depth: int = 20
    max_steps: int = 20 # max_depth in runner
    use_exact_memory: bool = False
    memory_pruning: bool = True
    skip_unary_actions: bool = False
    end_proof_action: bool = True
    reward_type: int = 4
    max_total_vars: int = 100
    max_fact_pairs_cap: Optional[int] = None  # Cap for large predicates (auto-set to eval_padding_states if None)
    eval_batch_size: int = 75  # Optimized batch size for evaluation (75 for best speed)
    fixed_batch_size: Optional[int] = None  # Fixed batch size for evaluation (defaults to n_envs if None)
    ranking_compile_mode: str = 'reduce-overhead'  # torch.compile mode for ranking_step
    sample_deterministic_per_env: bool = False  # False for fullgraph compilation compatibility
    
    # Model Architecture
    algorithm_type: str = "ppo"  # Algorithm to use (ppo, etc.)
    model_name: str = "PPO"
    atom_embedding_size: int = 250
    state_embedding_size: int = 64 # derived
    hidden_dim: int = 512
    num_layers: int = 8
    dropout_prob: float = 0.05  # Set to 0.0 to avoid train/eval mode issues with PPO value function learning
    separate_value_network: bool = True  # Use separate backbone for value network (independent from policy)
    use_l2_norm: bool = True
    sqrt_scale: bool = False
    temperature: float = 0.1
    atom_embedder: str = 'transe'
    state_embedder: str = 'mean'
    
    # PPO / Training Hyperparams
    n_envs: int = 128
    n_steps: int = 512  # Larger batches improve stability (exp4D)
    n_epochs: int = 5
    batch_size: int = 1024  # Match n_steps increase (exp4D)
    learning_rate: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = 0.2  # Stabilize value updates (None to disable)
    ent_coef: float = 0.1
    vf_coef: float = 1
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.07  # Stop epoch when policy diverges too much (0.07 optimal for family)

    # Value Function Learning Enhancements
    separate_value_lr: Optional[float] = None  # Separate LR for value network (None = use learning_rate)
    normalize_advantage: bool = False  # Normalize advantages per batch
    normalize_returns: bool = False  # Normalize returns for value targets
    value_head_scale: float = 1.0  # Scale factor for value head hidden dim (2.0 = 2x larger)
    total_timesteps: int = 2000000  # Optimal for family dataset (longer training hurts)
    
    # Sampling / Corruption
    negative_ratio: float = 1.0 # train_neg_ratio
    eval_neg_samples: Optional[int] = 100
    test_neg_samples: Optional[int] = 100
    corruption_scheme: List[str] = field(default_factory=lambda: ['head', 'tail'])
    sampler_default_mode: str = "both"
    
    # LR Warmup
    lr_warmup: bool = True  # Enable LR warmup to prevent early KL divergence
    lr_warmup_steps: float = 0.1  # Warmup for first 10% of training

    # LR Decay
    lr_decay: bool = True  # Reduce step size over training
    lr_init_value: float = 1e-4
    lr_final_value: float = 1e-6  # 100x reduction at end
    lr_start: float = 0.0
    lr_end: float = 1.0
    lr_transform: str = 'cos'  # Smooth cosine decay
    
    # Entropy Decay
    ent_coef_decay: bool = True  # Prevent entropy collapse
    ent_coef_init_value: float = 0.15  # Start with more exploration
    ent_coef_final_value: float = 0.02  # Maintain some exploration
    ent_coef_start: float = 0.0
    ent_coef_end: float = 0.8  # Finish decay at 80% training
    ent_coef_transform: str = 'cos'  # Smooth cosine decay
    
    # Model Saving / Logging
    save_model: bool = True
    load_model: Any = False # False or 'last_epoch' or path
    restore_best: bool = True # restore_best_val_model
    load_best_metric: str = 'eval'
    models_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    eval_freq: int = 4
    eval_best_metric: str = 'mrr'
    ranking_tie_seed: int = 0
    use_logger: bool = True
    logger_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
    run_signature: str = "compiled_run"

    # KGE inference (evaluation-time fusion)
    kge_inference: bool = False
    kge_inference_success: bool = False
    kge_engine: Optional[str] = "pytorch"
    kge_checkpoint_dir: Optional[str] = None
    kge_run_signature: Optional[str] = None
    kge_scores_file: Optional[str] = None
    kge_eval_kge_weight: float = 2.0
    kge_eval_rl_weight: float = 1.0
    kge_fail_penalty: float = 100  # Penalty for failed proofs in hybrid mode
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
    seed: int = 0
    seed_run_i: int = 0 # specific run seed
    device: str = "cuda"
    verbose: bool = True
    parity: bool = False
    profile: bool = False
    use_callbacks: bool = True  # Enable callbacks in run_experiment
    augment_train: bool = True  # For countries dataset

    # Logging control
    log_per_depth: bool = True  # Enable per-depth metrics in logs (e.g., proven_d_2_pos, proven_d_3_pos)
    log_per_predicate: bool = True  # Enable per-predicate metrics in logs (e.g., proven_d_2_pos_brother)

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

        # Dynamic Run Signature
        if self.run_signature in [None, "run_v1", "compiled_run"]:
            import datetime
            date_str = datetime.datetime.now().strftime("%Y%m%d")
            self.run_signature = f"{self.dataset}-{self.seed}-{date_str}"

        # Auto-configure padding_states per dataset
        if self.padding_states == 120:  # default value, auto-configure
            self.padding_states = PADDING_STATES_MAP.get(self.dataset, 64)

        # Auto-configure max_fact_pairs_cap for large datasets
        if self.max_fact_pairs_cap is None:
            self.max_fact_pairs_cap = MAX_FACT_PAIRS_CAP_MAP.get(self.dataset, None)

        # KGE inference: auto-set kge_run_signature if not provided
        if self.kge_inference and self.kge_run_signature is None:
            self.kge_run_signature = KGE_RUN_SIGNATURES.get(self.dataset, None)
