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
    n_eval_queries: Optional[int] = None
    n_test_queries: Optional[int] = None
    
    # Environment / Logic
    padding_atoms: int = 6
    padding_states: int = 64
    max_steps: int = 20 # max_depth in runner
    use_exact_memory: bool = True
    memory_pruning: bool = True
    skip_unary_actions: bool = True
    end_proof_action: bool = True
    reward_type: int = 0
    max_total_vars: int = 1000
    sample_deterministic_per_env: bool = True
    
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
    state_embedder: str = 'sum'
    
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
    eval_neg_samples: int = 4
    n_corruptions: int = 10 # test_neg_samples alias
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
    use_logger: bool = False
    logger_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
    run_signature: str = "compiled_run"
    
    # Misc
    seed: int = 42
    seed_run_i: int = 42 # specific run seed
    device: str = "cpu"
    verbose: bool = True
    parity: bool = False
    profile: bool = False
    use_callbacks: bool = True  # Enable callbacks in run_experiment

    # Compilation settings
    compile: bool = True  # Auto-compile environment functions
    compile_mode: str = 'reduce-overhead'
    compile_fullgraph: bool = True
    
    # Callback control (individual toggles)
    use_metrics_callback: bool = True
    use_ranking_callback: bool = True
    use_checkpoint_callback: bool = True
    use_annealing_callback: bool = True
    
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
