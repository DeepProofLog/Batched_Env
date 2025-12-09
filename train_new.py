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
    sample_deterministic_per_env: bool = True  # For parity testing
    
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
    
    # Model saving / evaluation
    eval_freq: int = 0  # Evaluate every N timesteps (0 = only at end)
    save_model: bool = False  # Save model checkpoints
    model_path: str = "./models/"  # Path to save models
    restore_best: bool = True  # Restore best model after training
    
    # Misc
    seed: int = 42
    device: str = "cpu"
    verbose: bool = True
    parity: bool = False

def seed_all(seed: int):
    """Set all random seeds."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_eval_callback(
    eval_env,
    eval_queries,
    sampler,
    policy,
    config: TrainParityConfig,
    model_path: str = None,
):
    """
    Create an evaluation callback that:
    1. Evaluates MRR periodically during training
    2. Saves model when a new best MRR is achieved
    3. Returns the callback function and a state dict for tracking best model
    
    Args:
        eval_env: Evaluation environment
        eval_queries: Tensor of evaluation queries
        sampler: Sampler for corruption generation
        policy: Policy network to save
        config: Training config
        model_path: Path to save best model (None = no saving)
    
    Returns:
        Tuple of (callback_fn, state_dict)
    """
    from pathlib import Path
    
    # State dict to track best model across callback invocations
    state = {
        'best_mrr': -1.0,
        'best_weights': None,
        'eval_count': 0,
        'total_timesteps_at_eval': [],
    }
    
    def callback(locals_dict, globals_dict):
        """Callback invoked at end of each learn iteration."""
        iteration = locals_dict.get('iteration', 0)
        total_steps_done = locals_dict.get('total_steps_done', 0)
        
        # Only evaluate if eval_freq is set and it's time
        if config.eval_freq <= 0:
            return True  # Continue training
        
        if total_steps_done > 0 and total_steps_done % config.eval_freq == 0:
            state['eval_count'] += 1
            state['total_timesteps_at_eval'].append(total_steps_done)
            
            # Run evaluation
            policy.eval()
            with torch.no_grad():
                eval_results = tensor_eval_corruptions(
                    actor=policy,
                    env=eval_env,
                    queries=eval_queries,
                    sampler=sampler,
                    n_corruptions=config.n_corruptions,
                    corruption_modes=('tail',),
                    verbose=False,
                )
            policy.train()
            
            current_mrr = eval_results.get('MRR', 0.0)
            
            print(f"[Eval {state['eval_count']}] timesteps={total_steps_done}, MRR={current_mrr:.4f}", end="")
            
            # Check if new best
            if current_mrr > state['best_mrr']:
                state['best_mrr'] = current_mrr
                # Save current weights as best
                state['best_weights'] = {k: v.clone() for k, v in policy.state_dict().items()}
                print(f" ★ New best!")
                
                # Save to disk if path specified
                if model_path and config.save_model:
                    save_path = Path(model_path)
                    save_path.mkdir(parents=True, exist_ok=True)
                    torch.save(policy.state_dict(), save_path / "best_model.pt")
                    print(f"    Saved to {save_path / 'best_model.pt'}")
            else:
                print()
        
        return True  # Continue training
    
    return callback, state

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
        sample_deterministic_per_env=config.sample_deterministic_per_env,
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
        sample_deterministic_per_env=config.sample_deterministic_per_env,
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
    
    # [PARITY] Output IndexManager info
    im = tensor_comp['im']
    print(f"[PARITY] IndexManager: constants={im.constant_no}, predicates={im.predicate_no}")
    
    # [PARITY] Output Embedder checksum
    embedder = tensor_comp['embedder']
    embedder_checksum = sum(p.sum().item() for p in embedder.parameters())
    print(f"[PARITY] Embedder checksum: {embedder_checksum:.6f}")
    
    # [PARITY] Output Policy init checksum
    policy = tensor_comp['policy']
    policy_checksum_init = sum(p.sum().item() for p in policy.parameters())
    print(f"[PARITY] Policy checksum after creation: {policy_checksum_init:.6f}")
    
    # [PARITY] Output RNG state before sampler
    print(f"[PARITY] RNG state before sampler: {torch.get_rng_state().sum().item():.0f}")
    
    # Prepare eval queries for callback (do this before training to avoid RNG drift)
    tensor_im = tensor_comp['im']
    eval_query_objs = tensor_comp['dh'].valid_queries[:config.n_envs * 4]  # Use valid for eval during training
    eval_query_atoms = []
    for q in eval_query_objs:
        query_atom = tensor_im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
        eval_query_atoms.append(query_atom)
    eval_queries = torch.stack(eval_query_atoms, dim=0)
    
    # Create evaluation callback if eval_freq is set
    callback = None
    callback_state = None
    if config.eval_freq > 0:
        callback, callback_state = make_eval_callback(
            eval_env=tensor_comp['eval_env'],
            eval_queries=eval_queries,
            sampler=tensor_comp['sampler'],
            policy=tensor_comp['policy'],
            config=config,
            model_path=config.model_path if config.save_model else None,
        )
        print(f"[Callback] Evaluating every {config.eval_freq} timesteps")
    
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
        parity=config.parity,
    )
    tensor_ppo.learn(total_timesteps=config.total_timesteps, callback=callback)
    
    # Restore best model if we tracked it
    if callback_state is not None and callback_state['best_weights'] is not None and config.restore_best:
        print(f"\n[Best Model] Restoring best model (MRR={callback_state['best_mrr']:.4f})")
        policy.load_state_dict(callback_state['best_weights'])
    
    # [PARITY] Output Policy trained checksum
    policy_checksum_trained = sum(p.sum().item() for p in policy.parameters())
    print(f"[PARITY] Policy checksum after training: {policy_checksum_trained:.6f}")
    
    # Tensor evaluation
    print("\n[3/3] Running evaluation...")
    seed_all(config.seed + 1000)
    
    # [PARITY] Output RNG state before eval
    print(f"[PARITY] RNG before eval: {torch.get_rng_state().sum().item():.0f}")
    
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
    
    # Extract results
    mrr = tensor_eval_results.get('MRR', 0.0)
    hits1 = tensor_eval_results.get('Hits@1', 0.0)
    hits3 = tensor_eval_results.get('Hits@3', 0.0)
    hits10 = tensor_eval_results.get('Hits@10', 0.0)
    
    # [PARITY] Output metrics in format parse_metrics expects
    print(f"\n[PARITY] Evaluation Results:")
    print(f"[PARITY] Tensor MRR: {mrr:.4f}")
    print(f"[PARITY] Tensor Hits@1: {hits1:.4f}")
    print(f"[PARITY] Tensor Hits@3: {hits3:.4f}")
    print(f"[PARITY] Tensor Hits@10: {hits10:.4f}")
    
    # Get training stats from PPO
    train_stats = getattr(tensor_ppo, 'last_train_metrics', {})
    
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


def main(args, log_filename=None, use_logger=False, use_WB=False, WB_path=None, date=None, external_components=None):
    """
    Main entry point for runner.py compatibility.
    
    Accepts runner.py arguments and converts them to TrainParityConfig
    for use with train_new's run_experiment() function.
    
    Args:
        args: Namespace with config from runner.py
        log_filename: Path to log file (for compatibility, not used)
        use_logger: Whether to enable logging (for compatibility)  
        use_WB: Whether to use Weights & Biases (for compatibility)
        WB_path: W&B run path (for compatibility)
        date: Timestamp string (for compatibility)
        external_components: Pre-initialized components for testing
    
    Returns:
        Tuple[Dict, Dict, Dict]: Metrics (train, valid, test)
    """
    # Build config from runner args namespace
    config = TrainParityConfig(
        # Dataset / data files
        dataset=getattr(args, 'dataset_name', 'countries_s3'),
        data_path=getattr(args, 'data_path', './data/'),
        train_file=getattr(args, 'train_file', 'train.txt'),
        valid_file=getattr(args, 'valid_file', 'valid.txt'),
        test_file=getattr(args, 'test_file', 'test.txt'),
        rules_file=getattr(args, 'rules_file', 'rules.txt'),
        facts_file=getattr(args, 'facts_file', 'train.txt'),
        train_depth=getattr(args, 'train_depth', None),
        
        # Environment / padding
        padding_atoms=getattr(args, 'padding_atoms', 6),
        padding_states=getattr(args, 'padding_states', 64),
        max_steps=getattr(args, 'max_depth', 20),
        use_exact_memory=getattr(args, 'use_exact_memory', True),
        memory_pruning=getattr(args, 'memory_pruning', True),
        skip_unary_actions=getattr(args, 'skip_unary_actions', True),
        end_proof_action=getattr(args, 'end_proof_action', True),
        reward_type=getattr(args, 'reward_type', 0),
        max_total_vars=getattr(args, 'max_total_vars', 1000),
        
        # PPO / training
        n_envs=getattr(args, 'batch_size_env', 3),
        n_steps=getattr(args, 'n_steps', 20),
        n_epochs=getattr(args, 'n_epochs', 4),
        batch_size=getattr(args, 'batch_size', 64),
        learning_rate=getattr(args, 'lr', 3e-4),
        gamma=getattr(args, 'gamma', 0.99),
        gae_lambda=getattr(args, 'gae_lambda', 0.95),
        clip_range=getattr(args, 'clip_range', 0.2),
        ent_coef=getattr(args, 'ent_coef', 0.2),
        vf_coef=getattr(args, 'vf_coef', 0.5),
        max_grad_norm=getattr(args, 'max_grad_norm', 0.5),
        total_timesteps=getattr(args, 'timesteps_train', 120),
        n_corruptions=getattr(args, 'eval_neg_samples', 10) or 10,
        
        # Embedding / model
        atom_embedding_size=getattr(args, 'atom_embedding_size', 64),
        
        # Model saving / evaluation
        eval_freq=getattr(args, 'eval_freq', 0),
        save_model=getattr(args, 'save_model', False),
        model_path=getattr(args, 'models_path', './models/'),
        restore_best=getattr(args, 'restore_best_val_model', True),
        
        # Misc
        seed=getattr(args, 'seed_run_i', 42),
        device=getattr(args, 'device', 'cpu'),
        verbose=getattr(args, 'verbose', True),
    )
    
    # Run training
    results = run_experiment(config)
    
    # Convert results to expected format for runner.py
    # runner.py expects (train_metrics, valid_metrics, test_metrics)
    test_metrics = {
        'mrr_mean': results.get('MRR', 0.0),
        'hits1_mean': results.get('Hits@1', 0.0),
        'hits3_mean': results.get('Hits@3', 0.0),
        'hits10_mean': results.get('Hits@10', 0.0),
        'rewards_pos_mean': 0.0,  # Not tracked in simple experiment
        'rewards_neg_mean': 0.0,
        'success_rate': 0.0,
    }
    train_metrics = {k: 0 for k in test_metrics.keys()}
    valid_metrics = {k: 0 for k in test_metrics.keys()}
    
    return train_metrics, valid_metrics, test_metrics


def main_cli():
    """Command-line interface for train_new.py (standalone use)."""
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
    main_cli()