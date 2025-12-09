"""
Profile the PPO learn() function in the tensor version.

This script profiles the training loop to identify bottlenecks in the
tensor-based implementation. Uses both cProfile and torch.profiler for
comprehensive CPU and GPU profiling.

Usage:
    python tests/profile_learn.py
    python tests/profile_learn.py --use-gpu-profiler
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import cProfile
import pstats
import io
from types import SimpleNamespace
from time import time

import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity

from utils.seeding import seed_all


def setup_components(device: torch.device, config: SimpleNamespace):
    """
    Initialize all components needed for training.
    
    Args:
        device: Target device (cuda or cpu)
        config: Configuration namespace
        
    Returns:
        Tuple of (ppo, policy, env, eval_env, sampler, dh, im)
    """
    # Enable TF32 for maximum performance on Ampere+ GPUs
    torch.set_float32_matmul_precision('high')
    
    from data_handler import DataHandler
    from index_manager import IndexManager
    from unification import UnificationEngine, set_compile_mode
    from env import BatchedEnv
    from embeddings import EmbedderLearnable as TensorEmbedder
    from model import ActorCriticPolicy as TensorPolicy
    from ppo import PPO as TensorPPO
    from sampler import Sampler
    
    # Enable compile mode for maximum performance
    set_compile_mode(True)
    
    # Set seeds for reproducibility
    seed_all(config.seed, deterministic=False, warn=False)
    
    # Load data
    dh = DataHandler(
        dataset_name=config.dataset,
        base_path=config.data_path,
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="train.txt",
        corruption_mode='dynamic',
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
    
    # Reseed before embedder
    torch.manual_seed(config.seed)
    
    # Embedder
    embedder = TensorEmbedder(
        n_constants=im.constant_no,
        n_predicates=im.predicate_no,
        n_vars=1000,
        max_arity=dh.max_arity,
        padding_atoms=config.padding_atoms,
        atom_embedder='transe',
        state_embedder='mean',
        constant_embedding_size=config.atom_embedding_size,
        predicate_embedding_size=config.atom_embedding_size,
        atom_embedding_size=config.atom_embedding_size,
        device=str(device),
    )
    embedder.embed_dim = config.atom_embedding_size
    
    # Create stringifier params
    stringifier_params = {
        'verbose': 1,
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
    
    # Convert queries to tensor format
    def convert_queries(queries):
        tensors = []
        for q in queries:
            atom = im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
            padded = torch.full((config.padding_atoms, 3), im.padding_idx, dtype=torch.long, device=device)
            padded[0] = atom
            tensors.append(padded)
        return torch.stack(tensors, dim=0)
    
    train_queries_tensor = convert_queries(dh.train_queries)
    test_queries_tensor = convert_queries(dh.test_queries)
    
    # Train environment
    train_env = BatchedEnv(
        batch_size=config.batch_size_env,
        queries=train_queries_tensor,
        labels=torch.ones(len(dh.train_queries), dtype=torch.long, device=device),
        query_depths=torch.ones(len(dh.train_queries), dtype=torch.long, device=device),
        unification_engine=engine,
        mode='train',
        max_depth=config.max_depth,
        memory_pruning=config.memory_pruning,
        use_exact_memory=False,  # Use BloomFilter (production default)
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
        sample_deterministic_per_env=False,
    )
    
    # Eval environment
    eval_env = BatchedEnv(
        batch_size=config.batch_size_env,
        queries=test_queries_tensor,
        labels=torch.ones(len(dh.test_queries), dtype=torch.long, device=device),
        query_depths=torch.ones(len(dh.test_queries), dtype=torch.long, device=device),
        unification_engine=engine,
        mode='eval',
        max_depth=config.max_depth,
        memory_pruning=config.memory_pruning,
        use_exact_memory=False,  # Use BloomFilter (production default)
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
        sample_deterministic_per_env=False,
    )
    
    # Reseed before model creation
    seed_all(config.seed, deterministic=False, warn=False)
    
    # Policy
    action_size = config.padding_states
    policy = TensorPolicy(
        embedder=embedder,
        embed_dim=config.atom_embedding_size,
        action_dim=action_size,
        hidden_dim=256,
        num_layers=8,
        dropout_prob=0.0,
        device=device,
        compile_model=config.compile,
        use_amp=config.use_amp,
    ).to(device)
    
    # PPO
    ppo = TensorPPO(
        policy=policy,
        env=train_env,
        n_steps=config.n_steps,
        learning_rate=config.lr,
        n_epochs=config.n_epochs,
        batch_size=config.batch_size,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        gamma=config.gamma,
        device=device,
        verbose=0,  # Quiet for profiling
        max_grad_norm=None,
        use_amp=config.use_amp,
    )
    return ppo, policy, train_env, eval_env, sampler, dh, im


def profile_learn_cprofile(config: SimpleNamespace):
    """Profile the PPO learn() function with cProfile."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("\nSetting up components...")
    ppo, policy, train_env, eval_env, sampler, dh, im = setup_components(device, config)
    
    print(f"\nProfiling PPO.learn() for {config.total_timesteps} timesteps...")
    profiler = cProfile.Profile()
    profiler.enable()
    
    start_time = time()
    ppo.learn(total_timesteps=config.total_timesteps)
    elapsed = time() - start_time
    
    profiler.disable()
    
    print(f"\nTraining completed in {elapsed:.2f}s")
    
    # Print profiling results
    n_functions = 30
    
    print("\n" + "="*80)
    print("PROFILING RESULTS - Top by Cumulative Time")
    print("="*80)
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats('cumulative')
    ps.print_stats(n_functions)
    print(s.getvalue())
    
    print("\n" + "="*80)
    print("PROFILING RESULTS - Top by Total Time")
    print("="*80)
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats('tottime')
    ps.print_stats(n_functions)
    print(s.getvalue())
    
    # Save results to file
    output_path = 'tests/profile_learn_results.txt'
    with open(output_path, 'w') as f:
        f.write(f"Device: {device}\n")
        f.write(f"Total timesteps: {config.total_timesteps}\n")
        f.write(f"Training time: {elapsed:.2f}s\n\n")
        
        ps = pstats.Stats(profiler, stream=f)
        ps.strip_dirs()
        f.write("="*80 + "\n")
        f.write("Top by Cumulative Time\n")
        f.write("="*80 + "\n")
        ps.sort_stats('cumulative')
        ps.print_stats(n_functions)
    
        f.write("\n" + "="*80 + "\n")
        f.write("Callers of .item()\n")
        f.write("="*80 + "\n")
        ps.print_callers('item', 20)
    
        f.write("\n\n" + "="*80 + "\n")
        f.write("Top by Total Time\n")
        f.write("="*80 + "\n")
        ps.sort_stats('tottime')
        ps.print_stats(n_functions)
    
    print(f"\nResults saved to {output_path}")


def profile_learn_gpu(config: SimpleNamespace):
    """Profile the PPO learn() function with torch.profiler for GPU analysis."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != 'cuda':
        print("CUDA not available, cannot profile GPU usage")
        return
    
    print(f"Using device: {device}")
    
    print("\nSetting up components...")
    ppo, policy, train_env, eval_env, sampler, dh, im = setup_components(device, config)
    
    print(f"\nGPU Profiling PPO.learn() for {config.total_timesteps} timesteps...")
    
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        start_time = time()
        ppo.learn(total_timesteps=config.total_timesteps)
        elapsed = time() - start_time
    
    print(f"\nTraining completed in {elapsed:.2f}s")
    
    print("\n" + "="*80)
    print("GPU PROFILING RESULTS - Top CUDA Time")
    print("="*80)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
    
    print("\n" + "="*80)
    print("GPU PROFILING RESULTS - Top CPU Time")
    print("="*80)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))
    
    # Save results to file
    output_path = 'tests/profile_learn_gpu_results.txt'
    with open(output_path, 'w') as f:
        f.write(f"Device: {device}\n")
        f.write(f"Total timesteps: {config.total_timesteps}\n")
        f.write(f"Training time: {elapsed:.2f}s\n\n")
        
        f.write("="*80 + "\n")
        f.write("Top CUDA Time\n")
        f.write("="*80 + "\n")
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
        
        f.write("\n\n" + "="*80 + "\n")
        f.write("Top CPU Time\n")
        f.write("="*80 + "\n")
        f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))
    
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Profile PPO learn() function')
    parser.add_argument('--use-gpu-profiler', action='store_true',
                        help='Use torch.profiler for GPU profiling instead of cProfile')
    parser.add_argument('--dataset', type=str, default='family',
                        help='Dataset name')
    parser.add_argument('--total-timesteps', type=int, default=90,
                        help='Total training timesteps')
    parser.add_argument('--batch-size-env', type=int, default=128,
                        help='Environment batch size')
    parser.add_argument('--n-steps', type=int, default=128,
                        help='Steps per rollout')
    parser.add_argument('--n-epochs', type=int, default=5,
                        help='PPO epochs per update')
    parser.add_argument('--batch-size', type=int, default=2048,
                        help='PPO minibatch size')
    parser.add_argument('--compile', default=True,
                        help='Enable torch.compile')
    parser.add_argument('--amp', default=True,
                        help='Enable Automatic Mixed Precision (AMP)')
    args = parser.parse_args()
    
    # Configuration matching runner.py defaults
    config = SimpleNamespace(
        dataset=args.dataset,
        data_path='./data/',
        total_timesteps=args.total_timesteps,
        batch_size_env=args.batch_size_env,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=5e-5,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.2,
        padding_atoms=6,
        padding_states=150,
        max_depth=20,
        memory_pruning=True,
        skip_unary_actions=False,  # Disabled for max FPS - unary actions have log_prob=0 so don't affect learning
        end_proof_action=True,
        reward_type=0,
        max_total_vars=100,
        atom_embedding_size=250,
        seed=0,
        compile=args.compile,
        use_amp=args.amp,
    )
    
    if args.use_gpu_profiler:
        profile_learn_gpu(config)
    else:
        profile_learn_cprofile(config)


if __name__ == '__main__':
    main()