"""
Tensor-based training script aligned with tests/parity/test_train_parity.py.
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
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_handler import DataHandler
from index_manager import IndexManager
from unification import UnificationEngine
from env import BatchedEnv
from embeddings import EmbedderLearnable as TensorEmbedder
from model import ActorCriticPolicy as TensorPolicy
from ppo import PPO as TensorPPO
from model_eval import eval_corruptions as tensor_eval_corruptions
from sampler import Sampler

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

def create_tensor_components(config: TrainParityConfig) -> Dict[str, Any]:
    """Create tensor training components (data handler, index manager, env, model)."""
    device = torch.device(config.device)
    
    # Data handler
    dh = DataHandler(
        dataset_name=config.dataset,
        base_path=config.data_path,
        train_file=config.train_file,
        valid_file=config.valid_file, 
        test_file=config.test_file,
        rules_file=config.rules_file,
        facts_file=config.facts_file,
        train_depth=config.train_depth,
        corruption_mode="dynamic",
    )
    
    # Index manager
    im = IndexManager(
        constants=dh.constants,
        predicates=dh.predicates,
        max_total_runtime_vars=config.max_total_vars,
        max_arity=dh.max_arity,
        padding_atoms=config.padding_atoms,
        device=device,
        rules=dh.rules,
    )
    
    # Materialize indices
    dh.materialize_indices(im=im, device=device)
    
    # Sampler
    domain2idx, entity2domain = dh.get_sampler_domain_info()
    sampler = Sampler.from_data(
        all_known_triples_idx=dh.all_known_triples_idx,
        num_entities=im.constant_no,
        num_relations=im.predicate_no,
        device=device,
        default_mode="both",
        seed=config.seed,
        domain2idx=domain2idx,
        entity2domain=entity2domain,
    )
    
    # Create stringifier params
    stringifier_params = {
        'verbose': 0,
        'idx2predicate': im.idx2predicate,
        'idx2constant': im.idx2constant,
        'idx2template_var': im.idx2template_var,
        'padding_idx': im.padding_idx,
        'n_constants': im.constant_no
    }
    
    # Create unification engine
    engine = UnificationEngine.from_index_manager(
        im, take_ownership=True,
        stringifier_params=stringifier_params,
        end_pred_idx=im.end_pred_idx,
        end_proof_action=config.end_proof_action,
        max_derived_per_state=config.padding_states,
    )
    engine.index_manager = im
    
    # Convert queries to tensor format like working tests
    train_queries = dh.train_queries
    test_queries = dh.test_queries
    
    def convert_queries_to_tensor(queries):
        query_tensors = []
        for q in queries:
            query_atom = im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
            query_padded = torch.full((config.padding_atoms, 3), im.padding_idx, dtype=torch.long, device=device)
            query_padded[0] = query_atom
            query_tensors.append(query_padded)
        return torch.stack(query_tensors, dim=0)
    
    train_queries_tensor = convert_queries_to_tensor(train_queries)
    test_queries_tensor = convert_queries_to_tensor(test_queries)
    
    # Create environments - match working tests exactly
    train_env = BatchedEnv(
        batch_size=config.n_envs,
        queries=train_queries_tensor,
        labels=torch.ones(len(train_queries), dtype=torch.long, device=device),
        query_depths=torch.ones(len(train_queries), dtype=torch.long, device=device),
        unification_engine=engine,
        mode='train',
        max_depth=config.max_steps,
        memory_pruning=config.memory_pruning,
        use_exact_memory=config.use_exact_memory,
        skip_unary_actions=config.skip_unary_actions,
        end_proof_action=config.end_proof_action,
        reward_type=config.reward_type,
        padding_atoms=config.padding_atoms,
        padding_states=config.padding_states,
        true_pred_idx=im.predicate_str2idx.get('True'),
        false_pred_idx=im.predicate_str2idx.get('False'),
        end_pred_idx=im.predicate_str2idx.get('Endf'),
        verbose=0,
        prover_verbose=0,
        device=device,
        runtime_var_start_index=im.constant_no + 1,
        total_vocab_size=im.constant_no + config.max_total_vars,
        sample_deterministic_per_env=True,
    )
    
    eval_env = BatchedEnv(
        batch_size=config.n_envs,
        queries=test_queries_tensor,
        labels=torch.ones(len(test_queries), dtype=torch.long, device=device),
        query_depths=torch.ones(len(test_queries), dtype=torch.long, device=device),
        unification_engine=engine,
        mode='eval',
        max_depth=config.max_steps,
        memory_pruning=config.memory_pruning,
        use_exact_memory=True,
        skip_unary_actions=config.skip_unary_actions,
        end_proof_action=config.end_proof_action,
        reward_type=config.reward_type,
        padding_atoms=config.padding_atoms,
        padding_states=config.padding_states,
        true_pred_idx=im.predicate_str2idx.get('True'),
        false_pred_idx=im.predicate_str2idx.get('False'),
        end_pred_idx=im.predicate_str2idx.get('Endf'),
        verbose=0,
        prover_verbose=0,
        device=device,
        runtime_var_start_index=im.constant_no + 1,
        total_vocab_size=im.constant_no + config.max_total_vars,
        sample_deterministic_per_env=True,
    )
    
    # Create embedder with fixed seed - match SB3 exactly
    n_vars_for_embedder = 1000
    torch.manual_seed(config.seed)
    embedder = TensorEmbedder(
        n_constants=im.constant_no,
        n_predicates=im.predicate_no,
        n_vars=n_vars_for_embedder,
        max_arity=dh.max_arity,
        padding_atoms=config.padding_atoms,
        atom_embedder='transe',
        state_embedder='sum',  # Must match SB3
        constant_embedding_size=config.atom_embedding_size,
        predicate_embedding_size=config.atom_embedding_size,
        atom_embedding_size=config.atom_embedding_size,
        device=str(device),
    )
    embedder.embed_dim = config.atom_embedding_size
    
    # Create policy with fixed seed
    action_size = config.padding_states
    torch.manual_seed(config.seed)
    policy = TensorPolicy(
        embedder=embedder,
        embed_dim=config.atom_embedding_size,
        action_dim=action_size,
        hidden_dim=256,
        num_layers=8,
        dropout_prob=0.0,
        device=device,
    ).to(device)
    
    return {
        'dh': dh,
        'im': im,
        'sampler': sampler,
        'embedder': embedder,
        'engine': engine,
        'train_env': train_env,
        'eval_env': eval_env,
        'policy': policy,
        'device': device,
    }

def run_experiment(config: TrainParityConfig) -> Dict[str, float]:
    """Run full training experiment and return evaluation metrics."""
    print("=" * 70)
    print("TENSOR TRAINING")
    print(f"Dataset: {config.dataset}")
    print(f"N envs: {config.n_envs}, N steps: {config.n_steps}")
    print(f"Total timesteps: {config.total_timesteps}")
    print(f"Seed: {config.seed}")
    print("=" * 70)
    
    # Create tensor components
    print("\n[1/3] Creating tensor components...")
    seed_all(config.seed)
    tensor_comp = create_tensor_components(config)
    
    # Tensor training
    print("\n[2/3] Running training...")
    seed_all(config.seed)
    tensor_ppo = TensorPPO(
        policy=tensor_comp['policy'],
        env=tensor_comp['train_env'],
        n_steps=config.n_steps,
        learning_rate=config.learning_rate,
        n_epochs=config.n_epochs,
        batch_size=config.batch_size,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        gamma=config.gamma,
        device=tensor_comp['device'],
        verbose=True,
    )
    tensor_ppo.learn(total_timesteps=config.total_timesteps)
    
    # Tensor evaluation
    print("\n[3/3] Running evaluation...")
    seed_all(config.seed + 1000)
    tensor_comp['policy'].eval()
    
    # Get test queries and convert to tensor
    tensor_im = tensor_comp['im']
    test_query_objs = tensor_comp['dh'].test_queries[:config.n_envs * 4]
    tensor_query_atoms = []
    for q in test_query_objs:
        query_atom = tensor_im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
        tensor_query_atoms.append(query_atom)
    tensor_queries = torch.stack(tensor_query_atoms, dim=0)
    
    tensor_eval_results = tensor_eval_corruptions(
        actor=tensor_comp['policy'],
        env=tensor_comp['eval_env'],
        queries=tensor_queries,
        sampler=tensor_comp['sampler'],
        n_corruptions=config.n_corruptions,
        corruption_modes=('tail',),
        verbose=False,
    )
    
    results = {
        "MRR": tensor_eval_results.get('MRR', 0.0),
        "Hits@1": tensor_eval_results.get('Hits@1', 0.0)
    }
    
    print("\nEvaluation Results:")
    print(f"  Tensor MRR: {results['MRR']:.4f}")
    print(f"  Tensor Hits@1: {results['Hits@1']:.4f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Train New (Tensor)")
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
