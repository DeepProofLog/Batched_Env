"""
Shared Parity Configuration.

Unified configuration for all parity tests ensuring consistent parameters
and tolerances across test_learn_parity, test_train_parity, test_runner_parity,
test_eval_parity, and test_script_parity.

Usage:
    from parity_config import ParityConfig, TOLERANCE, create_parser, config_from_args
    
    # Use defaults
    config = ParityConfig()
    
    # Override via CLI
    parser = create_parser()
    args = parser.parse_args()
    config = config_from_args(args)
"""
import argparse
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Any
from types import SimpleNamespace

# ==============================================================================
# Strict Tolerance - DO NOT RELAX
# ==============================================================================
TOLERANCE = 0.001  # Strict parity tolerance for all comparisons


# ==============================================================================
# Unified Configuration
# ==============================================================================

@dataclass
class ParityConfig:
    """
    Unified configuration for all parity tests.
    
    This replaces the separate configurations from each test file:
    - test_learn_parity.py: create_default_config()
    - test_train_parity.py: TrainParityConfig
    - test_runner_parity.py: RunnerParityConfig
    - test_eval_parity.py: create_default_config()
    """
    # Dataset / data files
    dataset: str = "countries_s3"
    data_path: str = "./data/"
    train_file: str = "train.txt"
    valid_file: str = "valid.txt"
    test_file: str = "test.txt"
    rules_file: str = "rules.txt"
    facts_file: str = "train.txt"
    train_depth: Optional[int] = None
    max_total_vars: int = 1000
    
    # Environment / padding
    padding_atoms: int = 6
    padding_states: int = 100
    max_depth: int = 20
    use_exact_memory: bool = True
    memory_pruning: bool = True
    skip_unary_actions: bool = True
    end_proof_action: bool = True
    reward_type: int = 0
    sample_deterministic_per_env: bool = True  # For parity testing
    
    # PPO / training
    n_envs: int = 4
    n_steps: int = 20
    n_epochs: int = 3
    batch_size: int = 512
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.2
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    total_timesteps: int = 120  # n_steps * n_envs * 2 default
    
    # Embedding / model
    embed_dim: int = 64
    n_vars_for_embedder: int = 1000
    atom_embedder: str = "transe"
    state_embedder: str = "sum"
    hidden_dim: int = 256
    num_layers: int = 8
    dropout_prob: float = 0.0
    
    # Evaluation
    n_eval_episodes: int = 24
    n_corruptions: int = 10
    corruption_mode: str = "tail"  # 'head', 'tail', or 'both'
    mode: str = "valid"  # which split to evaluate: 'train', 'valid', 'test'
    
    # Misc
    seed: int = 42
    device: str = "cpu"
    verbose: bool = True
    skip_training: bool = False
    skip_eval: bool = False
    parity: bool = True
    
    # For runner parity
    timesteps_train: int = 120
    train_neg_ratio: int = 4
    
    def clone(self) -> "ParityConfig":
        """Create a copy of this config."""
        return ParityConfig(**asdict(self))
    
    def to_namespace(self) -> SimpleNamespace:
        """Convert to SimpleNamespace for compatibility with existing code."""
        return SimpleNamespace(**asdict(self))
    
    def to_argparse_namespace(self) -> argparse.Namespace:
        """Convert to argparse.Namespace for compatibility."""
        return argparse.Namespace(**asdict(self))
    
    def update(self, **kwargs) -> "ParityConfig":
        """Return a new config with updated values."""
        new_dict = asdict(self)
        new_dict.update(kwargs)
        return ParityConfig(**new_dict)


# ==============================================================================
# CLI Parser
# ==============================================================================

def create_parser(description: str = "Parity Test") -> argparse.ArgumentParser:
    """
    Create argument parser with all config options.
    
    Each test file can use this parser and override as needed.
    """
    parser = argparse.ArgumentParser(description=description)
    
    # Dataset
    parser.add_argument("--dataset", type=str, default="countries_s3",
                        help="Dataset name (default: countries_s3)")
    parser.add_argument("--data-path", type=str, default="./data/",
                        help="Path to data directory")
    
    # Environment
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Number of parallel environments")
    parser.add_argument("--n-steps", type=int, default=20,
                        help="Number of rollout steps")
    parser.add_argument("--padding-atoms", type=int, default=6)
    parser.add_argument("--padding-states", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=20)
    
    # PPO
    parser.add_argument("--n-epochs", type=int, default=3,
                        help="Number of PPO epochs per update")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--total-timesteps", type=int, default=120)
    
    # Embedding
    parser.add_argument("--embed-dim", type=int, default=64)
    
    # Evaluation
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument("--n-corruptions", type=int, default=10)
    parser.add_argument("--corruption-mode", type=str, default="tail",
                        choices=["head", "tail", "both"])
    parser.add_argument("--mode", type=str, default="valid",
                        choices=["train", "valid", "test"])
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true", default=False,
                        help="Disable verbose output")
    parser.add_argument("--skip-training", action="store_true", default=False)
    parser.add_argument("--skip-eval", action="store_true", default=False)
    parser.add_argument("--parity-mode", action="store_true", default=True,
                       help="Enable strict parity mode (slower)")
    
    # Runner-specific
    parser.add_argument("--timesteps-train", type=int, default=256)
    parser.add_argument("--run-training", action="store_true", default=False,
                        help="Run actual training comparison (slow)")
    
    return parser


def config_from_args(args: argparse.Namespace, base: ParityConfig = None) -> ParityConfig:
    """
    Create ParityConfig from parsed arguments.
    
    Args:
        args: Parsed argparse.Namespace
        base: Optional base config to use as defaults
        
    Returns:
        ParityConfig with values from args
    """
    base = base or ParityConfig()
    base_dict = asdict(base)
    
    # Map CLI args to config fields (handle different naming conventions)
    arg_to_field = {
        'data_path': 'data_path',
        'n_envs': 'n_envs',
        'n_steps': 'n_steps',
        'n_epochs': 'n_epochs',
        'batch_size': 'batch_size',
        'learning_rate': 'learning_rate',
        'embed_dim': 'embed_dim',
        'n_eval_episodes': 'n_eval_episodes',
        'n_corruptions': 'n_corruptions',
        'corruption_mode': 'corruption_mode',
        'timesteps_train': 'timesteps_train',
        'total_timesteps': 'total_timesteps',
        'padding_atoms': 'padding_atoms',
        'padding_states': 'padding_states',
        'max_depth': 'max_depth',
        'gamma': 'gamma',
        'clip_range': 'clip_range',
        'ent_coef': 'ent_coef',
        'skip_training': 'skip_training',
        'skip_eval': 'skip_eval',
        'parity_mode': 'parity',
    }
    
    # Update base_dict with args
    args_dict = vars(args)
    for arg_name, field_name in arg_to_field.items():
        if arg_name in args_dict and args_dict[arg_name] is not None:
            base_dict[field_name] = args_dict[arg_name]
    
    # Direct mappings
    for field in ['dataset', 'seed', 'device', 'mode']:
        if field in args_dict and args_dict[field] is not None:
            base_dict[field] = args_dict[field]
    
    # Handle verbose/quiet
    if args_dict.get('quiet', False):
        base_dict['verbose'] = False
    elif 'verbose' in args_dict:
        base_dict['verbose'] = args_dict['verbose']
    
    return ParityConfig(**base_dict)


# ==============================================================================
# Trace Comparison Utilities
# ==============================================================================

def traces_match(val1: float, val2: float, tol: float = TOLERANCE) -> bool:
    """Check if two values match within tolerance."""
    return abs(val1 - val2) <= tol


def relative_close(val1: float, val2: float, rtol: float = TOLERANCE) -> bool:
    """Check if two values are relatively close."""
    max_val = max(abs(val1), abs(val2))
    if max_val < 1e-8:
        return True
    return abs(val1 - val2) / max_val < rtol


def assert_parity(name: str, val1: float, val2: float, tol: float = TOLERANCE) -> bool:
    """
    Assert parity between two values with informative message.
    
    Returns True if parity holds, False otherwise.
    Prints diagnostic information if parity fails.
    """
    diff = abs(val1 - val2)
    match = diff <= tol
    if not match:
        print(f"  ✗ {name}: SB3={val1:.6f}, Tensor={val2:.6f}, diff={diff:.6f} > tol={tol}")
    return match


if __name__ == "__main__":
    # Test the config
    config = ParityConfig()
    print(f"Default config: {config}")
    print(f"TOLERANCE: {TOLERANCE}")
    
    # Test CLI parsing
    parser = create_parser()
    args = parser.parse_args(["--dataset", "family", "--n-envs", "8"])
    config = config_from_args(args)
    print(f"Parsed config: dataset={config.dataset}, n_envs={config.n_envs}")
