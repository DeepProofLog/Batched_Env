"""
SB3-based training script aligned with tests/parity/test_train_parity.py.
"""
import gc
import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import numpy as np

# Add root to path
ROOT = Path(__file__).resolve().parents[1]
SB3_ROOT = ROOT / "sb3"
TEST_ENVS_ROOT = ROOT / "tests" / "other" / "test_envs"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SB3_ROOT) not in sys.path:
    sys.path.insert(1, str(SB3_ROOT))
if str(TEST_ENVS_ROOT) not in sys.path:
    sys.path.insert(2, str(TEST_ENVS_ROOT))

# SB3 imports
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.callbacks import BaseCallback

# sb3 imports
from sb3.sb3_custom_dummy_env import CustomDummyVecEnv
from sb3.sb3_dataset import DataHandler as SB3DataHandler
from sb3.sb3_index_manager import IndexManager as SB3IndexManager
from sb3.sb3_env import LogicEnv_gym as SB3Env
from sb3.sb3_model import PPO_custom as SB3PPO, CustomActorCriticPolicy, CustomCombinedExtractor
from sb3.sb3_embeddings import EmbedderLearnable as SB3Embedder
from sb3.sb3_model_eval import eval_corruptions as sb3_eval_corruptions
from sb3.sb3_neg_sampling import get_sampler as get_sb3_sampler

@dataclass
class TrainParityConfig:
    """Configuration for train parity tests."""
    # Dataset / data files
    dataset: str = "countries_s3"
    data_path: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    train_file: str = "train.txt"
    valid_file: str = "valid.txt"
    test_file: str = "test.txt"
    rules_file: str = "rules.txt"
    facts_file: str = "train.txt"
    train_depth: Any = None
    
    # Environment / padding
    padding_atoms: int = 6
    padding_states: int = 64
    max_steps: int = 20
    use_exact_memory: bool = True
    memory_pruning: bool = True
    skip_unary_actions: bool = True
    end_proof_action: bool = True
    reward_type: int = 0
    max_total_vars: int = 1000
    
    # PPO / training
    n_envs: int = 3
    n_steps: int = 20
    n_epochs: int = 4
    batch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.2
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None #0.03  # KL divergence threshold for early stopping
    total_timesteps: int = 120
    n_corruptions: int = 10
    corruption_scheme: List[str] = None  # ['head'], ['tail'], or ['head', 'tail']
    sampler_default_mode: str = "both" # this allows the sampler to corrupt both head and tail by default. Can be overridden per-eval.

    def __post_init__(self):
        # Set default corruption_scheme based on dataset if not specified
        if self.corruption_scheme is None:
            if 'countries' in self.dataset or 'ablation' in self.dataset:
                self.corruption_scheme = ['tail']
            else:
                self.corruption_scheme = ['head', 'tail']
    
    # Embedding / model
    atom_embedding_size: int = 64
    
    # Model saving / evaluation
    eval_freq: int = 0  # Evaluate every N timesteps (0 = only at end)
    save_model: bool = False  # Save model checkpoints
    model_path: str = "./models/"  # Path to save models
    restore_best: bool = True  # Restore best model after training
    
    # Misc
    seed: int = 42
    device: str = "cpu"
    verbose: bool = True
    parity: bool = False  # Enable deterministic mode for parity testing
    sample_deterministic_per_env: bool = True  # For parity testing

def seed_all(seed: int, deterministic: bool = False):
    """Set all random seeds. If deterministic=True, enables torch.use_deterministic_algorithms."""
    import random
    import os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        # Set CUBLAS_WORKSPACE_CONFIG for deterministic CUDA matmul operations
        if torch.cuda.is_available():
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True, warn_only=False)
        print('ensuring determinism in the torch algorithm')

def create_sb3_components(config: TrainParityConfig) -> Dict[str, Any]:
    """Create SB3 training components (data handler, index manager, env, model)."""
    device = torch.device(config.device)
    
    # Data handler
    dh = SB3DataHandler(
        dataset_name=config.dataset,
        base_path=config.data_path,
        train_file=config.train_file,
        valid_file=config.valid_file,
        test_file=config.test_file,
        rules_file=config.rules_file,
        facts_file=config.facts_file,
        train_depth=config.train_depth,
        corruption_mode=True,
    )
    
    # Index manager
    im = SB3IndexManager(
        constants=dh.constants,
        predicates=dh.predicates,
        max_total_vars=config.max_total_vars,
        rules=dh.rules,
        padding_atoms=config.padding_atoms,
        max_arity=dh.max_arity,
        device=device,
    )
    facts_set = set(dh.facts)
    im.build_fact_index(list(facts_set), deterministic=True)
    
    # Sampler
    # Convert sampler_default_mode string to list format expected by corruption_scheme
    if config.sampler_default_mode == 'both':
        sampler_corruption_scheme = ['head', 'tail']
    elif config.sampler_default_mode == 'head':
        sampler_corruption_scheme = ['head']
    elif config.sampler_default_mode == 'tail':
        sampler_corruption_scheme = ['tail']
    else:
        sampler_corruption_scheme = ['head', 'tail']  # Default fallback
    
    sampler = get_sb3_sampler(
        data_handler=dh,
        index_manager=im,
        corruption_scheme=sampler_corruption_scheme,
        device=device,
        corruption_mode=True,
    )
    
    # Embedder - use reasonable n_vars to avoid massive memory usage
    n_vars_for_embedder = 1000
    torch.manual_seed(config.seed)
    embedder = SB3Embedder(
        n_constants=im.constant_no,
        n_predicates=im.predicate_no,
        n_vars=n_vars_for_embedder,
        max_arity=dh.max_arity,
        padding_atoms=config.padding_atoms,
        atom_embedder='transe',
        state_embedder='sum',
        constant_embedding_size=config.atom_embedding_size,
        predicate_embedding_size=config.atom_embedding_size,
        atom_embedding_size=config.atom_embedding_size,
        device=device,
    )
    embedder.embed_dim = config.atom_embedding_size
    
    # Create environments with Monitor wrapper like working tests
    def make_env(idx: int, mode: str = "train"):
        def _init():
            queries = dh.train_queries if mode == "train" else dh.test_queries
            labels = [1] * len(queries)  # All queries are positive examples
            env = SB3Env(
                index_manager=im,
                data_handler=dh,
                queries=queries,
                labels=labels,
                query_depths=None,
                facts=facts_set,
                mode=mode,
                sample_deterministic=True,  # Round-robin for parity
                seed=config.seed,
                max_depth=config.max_steps,
                memory_pruning=config.memory_pruning,
                padding_atoms=config.padding_atoms,
                padding_states=config.padding_states,
                verbose=0,
                prover_verbose=0,
                device=device,
                engine='python',
                engine_strategy='complete',
                skip_unary_actions=config.skip_unary_actions,
                endf_action=config.end_proof_action,
                reward_type=config.reward_type,
            )
            env._train_ptr = idx  # Set train pointer for deterministic query selection
            return Monitor(env)  # Wrap with Monitor like working tests
        return _init
    
    # Use DummyVecEnv for training, CustomDummyVecEnv for evaluation
    train_env = DummyVecEnv([make_env(i, "train") for i in range(config.n_envs)])
    eval_env = CustomDummyVecEnv([make_env(i, "eval") for i in range(config.n_envs)])
    
    # Create model
    torch.manual_seed(config.seed)
    policy_kwargs = {
        'features_extractor_class': CustomCombinedExtractor,
        'features_extractor_kwargs': {'features_dim': embedder.embed_dim, 'embedder': embedder},
    }
    
    model = SB3PPO(
        CustomActorCriticPolicy,
        train_env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        verbose=1,
        device=device,
        ent_coef=config.ent_coef,
        clip_range=config.clip_range,
        gamma=config.gamma,
        target_kl=config.target_kl,
        policy_kwargs=policy_kwargs,
        seed=config.seed,
    )
    
    return {
        'dh': dh,
        'im': im,
        'sampler': sampler,
        'embedder': embedder,
        'train_env': train_env,
        'eval_env': eval_env,
        'model': model,
        'device': device,
    }

def run_experiment(config: TrainParityConfig) -> Dict[str, float]:
    """Run full training experiment and return evaluation metrics."""
    from pathlib import Path
    from stable_baselines3.common.callbacks import BaseCallback
    
    # Custom callback for MRR evaluation during training
    class MRREvalCallback(BaseCallback):
        """Callback that evaluates MRR and saves best model."""
        def __init__(self, eval_env, sampler, data_handler, config, model_path=None):
            super().__init__()
            self.eval_env = eval_env
            self.sampler = sampler
            self.data_handler = data_handler
            self.config = config
            self.model_path = Path(model_path) if model_path else None
            self.best_mrr = -1.0
            self.best_weights = None
            self.eval_count = 0
            
        def _on_step(self) -> bool:
            # Check if it's time to evaluate
            if self.config.eval_freq <= 0:
                return True
            if self.num_timesteps > 0 and self.num_timesteps % self.config.eval_freq == 0:
                self.eval_count += 1
                
                # Run evaluation
                test_queries = self.data_handler.valid_queries[:self.config.n_envs * 4]
                eval_results = sb3_eval_corruptions(
                    model=self.model,
                    env=self.eval_env,
                    data=test_queries,
                    sampler=self.sampler,
                    n_corruptions=self.config.n_corruptions,
                    corruption_scheme=self.config.corruption_scheme,
                    verbose=0,
                )
                
                current_mrr = eval_results.get('mrr_mean', 0.0)
                print(f"[Eval {self.eval_count}] timesteps={self.num_timesteps}, MRR={current_mrr:.4f}", end="")
                
                # Check if new best
                if current_mrr > self.best_mrr:
                    self.best_mrr = current_mrr
                    self.best_weights = {k: v.cpu().clone() for k, v in self.model.policy.state_dict().items()}
                    print(f" ★ New best!")
                    
                    if self.model_path and self.config.save_model:
                        self.model_path.mkdir(parents=True, exist_ok=True)
                        self.model.save(self.model_path / "best_model")
                        print(f"    Saved to {self.model_path / 'best_model'}")
                else:
                    print()
            return True
    
    print("=" * 70)
    print("SB3 TRAINING")
    print(f"Dataset: {config.dataset}")
    print(f"N envs: {config.n_envs}, N steps: {config.n_steps}")
    print(f"Total timesteps: {config.total_timesteps}")
    print(f"Seed: {config.seed}")
    print("=" * 70)
    
    # Create SB3 components
    print("\n[1/3] Creating SB3 components...")
    seed_all(config.seed, deterministic=config.parity)
    sb3_comp = create_sb3_components(config)
    
    # [PARITY] Output IndexManager info
    im = sb3_comp['im']
    print(f"[PARITY] IndexManager: constants={im.constant_no}, predicates={im.predicate_no}")
    
    # [PARITY] Output Embedder checksum
    embedder = sb3_comp['embedder']
    embedder_checksum = sum(p.sum().item() for p in embedder.parameters())
    print(f"[PARITY] Embedder checksum: {embedder_checksum:.6f}")
    
    # [PARITY] Output Policy init checksum
    policy = sb3_comp['model'].policy
    policy_checksum_init = sum(p.sum().item() for p in policy.parameters())
    print(f"[PARITY] Policy checksum after creation: {policy_checksum_init:.6f}")
    
    # [PARITY] Output RNG state before sampler
    print(f"[PARITY] RNG state before sampler: {torch.get_rng_state().sum().item():.0f}")
    
    # Create callback if eval_freq is set
    callback = None
    if config.eval_freq > 0:
        callback = MRREvalCallback(
            eval_env=sb3_comp['eval_env'],
            sampler=sb3_comp['sampler'],
            data_handler=sb3_comp['dh'],
            config=config,
            model_path=config.model_path if config.save_model else None,
        )
        print(f"[Callback] Evaluating every {config.eval_freq} timesteps")
    
    # SB3 training
    print("\n[2/3] Running training...")
    seed_all(config.seed, deterministic=config.parity)
    sb3_comp['model'].learn(total_timesteps=config.total_timesteps, progress_bar=False, callback=callback)
    
    # Restore best model if we tracked it
    if callback is not None and callback.best_weights is not None and config.restore_best:
        print(f"\n[Best Model] Restoring best model (MRR={callback.best_mrr:.4f})")
        sb3_comp['model'].policy.load_state_dict(callback.best_weights)
    
    # [PARITY] Output Policy trained checksum
    policy_checksum_trained = sum(p.sum().item() for p in policy.parameters())
    print(f"[PARITY] Policy checksum after training: {policy_checksum_trained:.6f}")
    
    # SB3 evaluation
    print("\n[3/3] Running evaluation...")
    seed_all(config.seed + 1000, deterministic=config.parity)
    
    # [PARITY] Output RNG state before eval
    print(f"[PARITY] RNG before eval: {torch.get_rng_state().sum().item():.0f}")
    
    test_queries = sb3_comp['dh'].test_queries[:config.n_envs * 4]
    
    # [DEBUG] Log evaluation setup
    print(f"\n[SB3 EVAL DEBUG]")
    print(f"  corruption_scheme: {config.corruption_scheme if getattr(config, 'corruption_scheme', None) else 'default'}")
    print(f"  n_corruptions: {config.n_corruptions}")
    print(f"  num test queries: {len(test_queries)}")
    print(f"  sampler default_mode: {config.sampler_default_mode}")
    print(f"  first query: {test_queries[0]}")
    
    sb3_eval_results = sb3_eval_corruptions(
        model=sb3_comp['model'],
        env=sb3_comp['eval_env'],
        data=test_queries,
        sampler=sb3_comp['sampler'],
        n_corruptions=config.n_corruptions,
        corruption_scheme=config.corruption_scheme,  # Use config instead of hardcoded
        verbose=0,
    )
    
    # Extract results
    mrr = sb3_eval_results.get('mrr_mean', 0.0)
    hits1 = sb3_eval_results.get('hits1_mean', 0.0)
    hits3 = sb3_eval_results.get('hits3_mean', 0.0)
    hits10 = sb3_eval_results.get('hits10_mean', 0.0)
    
    # [PARITY] Output metrics in format parse_metrics expects
    print(f"\n[PARITY] Evaluation Results:")
    print(f"[PARITY] SB3 MRR: {mrr:.4f}")
    print(f"[PARITY] SB3 Hits@1: {hits1:.4f}")
    print(f"[PARITY] SB3 Hits@3: {hits3:.4f}")
    print(f"[PARITY] SB3 Hits@10: {hits10:.4f}")
    
    # Get training stats from SB3 model
    model = sb3_comp['model']
    train_stats = getattr(model, 'last_train_metrics', {})
    
    # Comprehensive results dict
    results = {
        # Evaluation metrics
        "MRR": mrr,
        "Hits@1": hits1,
        "Hits@3": hits3,
        "Hits@10": hits10,
        # Checksums
        "index_manager_constants": im.constant_no,
        "index_manager_predicates": im.predicate_no,
        "embedder_checksum": embedder_checksum,
        "policy_checksum_init": policy_checksum_init,
        "policy_checksum_trained": policy_checksum_trained,
        # Training losses (from last epoch)
        "policy_loss": train_stats.get('policy_loss', 0.0),
        "value_loss": train_stats.get('value_loss', 0.0),
        "entropy": train_stats.get('entropy', 0.0),
        "approx_kl": train_stats.get('approx_kl', 0.0),
        "clip_fraction": train_stats.get('clip_fraction', 0.0),
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Train New (SB3)")
    parser.add_argument("--dataset", type=str, default="countries_s3")
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--total-timesteps", type=int, default=2000)
    parser.add_argument("--n-corruptions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    # Callback / model saving options
    parser.add_argument("--eval-freq", type=int, default=0,
                        help="Evaluate every N timesteps (0=only at end)")
    parser.add_argument("--save-model", action="store_true", default=False,
                        help="Save model checkpoints")
    parser.add_argument("--model-path", type=str, default="./models/",
                        help="Path to save models")
    parser.add_argument("--no-restore-best", action="store_true", default=False,
                        help="Don't restore best model after training")
    
    args = parser.parse_args()
    
    config = TrainParityConfig(
        dataset=args.dataset,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        total_timesteps=args.total_timesteps,
        n_corruptions=args.n_corruptions,
        seed=args.seed,
        device=args.device,
        verbose=args.verbose,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        eval_freq=args.eval_freq,
        save_model=args.save_model,
        model_path=args.model_path,
        restore_best=not args.no_restore_best,
    )
    
    run_experiment(config)

if __name__ == "__main__":
    main()
