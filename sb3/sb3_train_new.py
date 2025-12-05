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
TEST_ENVS_ROOT = ROOT / "test_envs"

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
from sb3_custom_dummy_env import CustomDummyVecEnv
from sb3_dataset import DataHandler as SB3DataHandler
from sb3_index_manager import IndexManager as SB3IndexManager
from sb3_env import LogicEnv_gym as SB3Env
from sb3_model import PPO_custom as SB3PPO, CustomActorCriticPolicy, CustomCombinedExtractor
from sb3_embeddings import EmbedderLearnable as SB3Embedder
from sb3_model_eval import eval_corruptions as sb3_eval_corruptions
from sb3_neg_sampling import get_sampler as get_sb3_sampler

@dataclass
class TrainParityConfig:
    """Configuration for train parity tests."""
    # Dataset / data files
    dataset: str = "countries_s3"
    data_path: str = "./data/"
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
    total_timesteps: int = 120
    n_corruptions: int = 10
    
    # Embedding / model
    atom_embedding_size: int = 64
    
    # Misc
    seed: int = 42
    device: str = "cpu"
    verbose: bool = True

def seed_all(seed: int):
    """Set all random seeds."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
    sampler = get_sb3_sampler(
        data_handler=dh,
        index_manager=im,
        corruption_scheme=['head', 'tail'],
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
    print("=" * 70)
    print("SB3 TRAINING")
    print(f"Dataset: {config.dataset}")
    print(f"N envs: {config.n_envs}, N steps: {config.n_steps}")
    print(f"Total timesteps: {config.total_timesteps}")
    print(f"Seed: {config.seed}")
    print("=" * 70)
    
    # Create SB3 components
    print("\n[1/3] Creating SB3 components...")
    seed_all(config.seed)
    sb3_comp = create_sb3_components(config)
    
    # SB3 training
    print("\n[2/3] Running training...")
    seed_all(config.seed)
    sb3_comp['model'].learn(total_timesteps=config.total_timesteps, progress_bar=False)
    
    # SB3 evaluation
    print("\n[3/3] Running evaluation...")
    seed_all(config.seed + 1000)
    test_queries = sb3_comp['dh'].test_queries[:config.n_envs * 4]
    
    sb3_eval_results = sb3_eval_corruptions(
        model=sb3_comp['model'],
        env=sb3_comp['eval_env'],
        data=test_queries,
        sampler=sb3_comp['sampler'],
        n_corruptions=config.n_corruptions,
        corruption_scheme=['tail'],
        verbose=0,
    )
    
    results = {
        "MRR": sb3_eval_results.get('mrr_mean', 0.0),
        "Hits@1": sb3_eval_results.get('hits1_mean', 0.0)
    }
    
    print("\nEvaluation Results:")
    print(f"  SB3 MRR: {results['MRR']:.4f}")
    print(f"  SB3 Hits@1: {results['Hits@1']:.4f}")
    
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
    )
    
    run_experiment(config)

if __name__ == "__main__":
    main()
