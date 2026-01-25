"""
Profile the PPO.learn() function.

This script profiles the optimized training loop to verify performance
matches or exceeds the original implementation.

Targets:
- Runtime: ≤16s (matching profile_learn.py baseline)
- Steps/second: ≥ baseline

Usage:
    conda activate rl
    python tests/profile_learn_.py
    python tests/profile_learn_.py --use-gpu-profiler
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import cProfile
import pstats
import io
from datetime import datetime
from types import SimpleNamespace
from time import time

import torch

try:
    from torch.profiler import profile, ProfilerActivity
except (ImportError, ModuleNotFoundError):
    profile = None
    ProfilerActivity = None


class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

    def isatty(self):
        return any(getattr(f, 'isatty', lambda: False)() for f in self.files)


def setup_components(device: torch.device, config: SimpleNamespace):
    """Initialize all components for  training."""
    torch.set_float32_matmul_precision('high')
    
    from data_handler import DataHandler
    from index_manager import IndexManager
    from unification import UnificationEngineVectorized
    from nn.embeddings import EmbedderLearnable as TensorEmbedder
    from policy import ActorCriticPolicy as TensorPolicy
    from nn.sampler import Sampler
    from env import EnvVec
    from ppo import PPO
    from train import build_callbacks


    dh = DataHandler(
        dataset_name=config.dataset,
        base_path=config.data_path,
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="train.txt",
    )
    
    im = IndexManager(
        constants=dh.constants,
        predicates=dh.predicates,
        max_total_runtime_vars=config.max_total_vars,
        max_arity=dh.max_arity,
        padding_atoms=config.padding_atoms,
        device=device,
        rules=dh.rules,
    )
    dh.materialize_indices(im=im, device=device)
    
    default_mode = 'tail' if 'countries' in config.dataset else 'both'
    domain2idx, entity2domain = dh.get_sampler_domain_info()
    sampler = Sampler.from_data(
        all_known_triples_idx=dh.all_known_triples_idx,
        num_entities=im.constant_no,
        num_relations=im.predicate_no,
        device=device,
        default_mode=default_mode,
        seed=config.seed,
        domain2idx=domain2idx,
        entity2domain=entity2domain,
    )
    
    torch.manual_seed(config.seed)
    
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
    
    vec_engine = UnificationEngineVectorized.from_index_manager(
        im,
        padding_atoms=config.padding_atoms,
        max_derived_per_state=config.padding_states,
        end_proof_action=config.end_proof_action,
    )
    
    def convert_queries_unpadded(queries):
        tensors = []
        for q in queries:
            atom = im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
            tensors.append(atom)
        return torch.stack(tensors, dim=0)
    
    train_queries = convert_queries_unpadded(dh.train_queries)
    test_queries = convert_queries_unpadded(dh.test_queries)
    
    # Use Env instead of EnvVec
    train_env = EnvVec(
        vec_engine=vec_engine,
        batch_size=config.batch_size_env,
        padding_atoms=config.padding_atoms,
        padding_states=config.padding_states,
        max_depth=config.max_depth,
        end_proof_action=config.end_proof_action,
        runtime_var_start_index=im.constant_no + 1,
        device=device,
        memory_pruning=True,
        sampler=sampler,
        train_queries=train_queries,
        valid_queries=test_queries,
        skip_unary_actions=True,
    )
    
    action_size = config.padding_states
    policy = TensorPolicy(
        embedder=embedder,
        embed_dim=config.atom_embedding_size,
        action_dim=action_size,
        hidden_dim=256,
        num_layers=8,
        dropout_prob=0.0,
        device=device,
        compile_policy=config.compile,
        use_amp=config.use_amp,
    ).to(device)
    
    # Use PPO instead of PPO
    ppo = PPO(policy, train_env, config, device=device)

    # Build callbacks if enabled
    if getattr(config, 'use_callbacks', False):
        callback_manager, _, _ = build_callbacks(
            config, ppo, policy, sampler, dh,
            eval_env=train_env, date=None, im=im
        )
        ppo.callback = callback_manager

    return {
        'ppo': ppo,
        'policy': policy,
        'train_env': train_env,
        'sampler': sampler,
        'dh': dh,
        'im': im,
        'vec_engine': vec_engine,
        'train_queries': train_queries,
        'test_queries': test_queries,
    }


def compile_and_warmup(components, config) -> float:
    """Run warmup to compile graphs. PPO handles compilation internally."""
    ppo = components['ppo']

    warmup_start = time()

    # Warmup: run learn() iteration to compile all graphs
    # PPO compiles in its __init__, this just ensures graphs are ready
    print("Warmup: Running learn() iteration to compile all graphs...")
    ppo.learn(total_timesteps=config.n_steps)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    warmup_time = time() - warmup_start
    return warmup_time


def run_training(components, config):
    """Run training steps."""
    ppo = components['ppo']
    
    ppo.learn(
        total_timesteps=config.total_timesteps,
        reset_num_timesteps=False,
    )
    
    return ppo.last_train_metrics if hasattr(ppo, 'last_train_metrics') else {}, ppo.num_timesteps


def profile_cprofile(config: SimpleNamespace):
    """Profile with cProfile for CPU bottlenecks."""
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'profile_learn.txt')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    log_file = open(output_path, 'w')
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = sys.stdout
    
    try:
        initial_wallclock = time()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Profile  Learn Results")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Device: {device}")
        print(f"Dataset: {config.dataset}")
        print(f"\nConfiguration:")
        print(f"  Total timesteps (profiled): {config.total_timesteps}")
        print(f"  Batch size env: {config.batch_size_env}")
        print(f"  N steps per rollout: {config.n_steps}")
        print(f"  N epochs: {config.n_epochs}")
        print(f"  Minibatch size: {config.batch_size}")
        
        print("\nSetting up components...")
        init_start = time()
        components = setup_components(device, config)
        init_time = time() - init_start
        print(f"Initialization time: {init_time:.2f}s")
        
        print("\nRunning warmup (compilation + first rollout)...")
        warmup_time = compile_and_warmup(components, config)
        print(f"Warmup time: {warmup_time:.2f}s")
        
        print(f"\nProfiling training for {config.total_timesteps} timesteps...")
        profiler = cProfile.Profile()
        profiler.enable()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time()
        
        metrics, total_steps = run_training(components, config)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        runtime = time() - start_time
        
        profiler.disable()
        
        total_time = warmup_time + runtime
        warmup_steps = config.batch_size_env * config.n_steps
        profile_steps = total_steps - warmup_steps
        steps_per_sec = profile_steps / runtime if runtime > 0 else 0
        
        print(f"\n{'='*80}")
        print("TIMING SUMMARY")
        print(f"{'='*80}")
        print(f"Init time:         {init_time:.4f}s")
        print(f"Warmup time:       {warmup_time:.4f}s")
        print(f"Runtime:           {runtime:.4f}s")
        print(f"Total time:        {total_time:.4f}s")
        print(f"Steps/second:      {steps_per_sec:.1f}")
        print(f"Total timesteps:   {total_steps} (Warmup: {warmup_steps}, Profiled: {profile_steps})")
        print(f"\nTarget: Runtime ≤ 16s")
        print(f"Status: {'PASS ✓' if runtime <= 16 else 'FAIL ✗'}")
        
        if metrics:
            print(f"\nLast training metrics:")
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
        
        n_functions = 40
        
        print(f"\n{'='*80}")
        print("PROFILING RESULTS - Top by Cumulative Time")
        print(f"{'='*80}")
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.strip_dirs()
        ps.sort_stats('cumulative')
        ps.print_stats(n_functions)
        print(s.getvalue())
        
        print(f"\n{'='*80}")
        print("PROFILING RESULTS - Top by Total Time")
        print(f"{'='*80}")
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.strip_dirs()
        ps.sort_stats('tottime')
        ps.print_stats(n_functions)
        print(s.getvalue())
        
        print(f"\nResults saved to {output_path}")

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


def profile_gpu(config: SimpleNamespace):
    """Profile with torch.profiler for GPU analysis."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != 'cuda':
        print("CUDA not available, cannot profile GPU")
        return
    
    if profile is None:
        print("torch.profiler not available")
        return
        
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'profile_learn_gpu.txt')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    log_file = open(output_path, 'w')
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = sys.stdout
    
    try:
        print(f"Profile  Learn GPU Results")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Device: {device}")
        
        print("\nSetting up components...")
        init_start = time()
        components = setup_components(device, config)
        init_time = time() - init_start
        print(f"Initialization time: {init_time:.2f}s")
        
        print("\nRunning warmup...")
        warmup_time = compile_and_warmup(components, config)
        print(f"Warmup time: {warmup_time:.2f}s")
        
        print(f"\nGPU Profiling training...")
        
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            torch.cuda.synchronize()
            start_time = time()
            metrics, total_steps = run_training(components, config)
            torch.cuda.synchronize()
            runtime = time() - start_time
        
        warmup_steps = config.batch_size_env * config.n_steps
        profile_steps = total_steps - warmup_steps
        steps_per_sec = profile_steps / runtime if runtime > 0 else 0
        
        print(f"\n{'='*80}")
        print("TIMING SUMMARY")
        print(f"{'='*80}")
        print(f"Warmup time:       {warmup_time:.4f}s")
        print(f"Runtime:           {runtime:.4f}s")
        print(f"Steps/second:      {steps_per_sec:.1f}")
        print(f"\nTarget: Runtime ≤ 16s")
        print(f"Status: {'PASS ✓' if runtime <= 16 else 'FAIL ✗'}")
        
        print(f"\n{'='*80}")
        print("GPU PROFILING - Top CUDA Time")
        print(f"{'='*80}")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=40))
        
        print(f"\n{'='*80}")
        print("GPU PROFILING - Top CPU Time")
        print(f"{'='*80}")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=40))
        
        print(f"\nResults saved to {output_path}")

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


def main():
    parser = argparse.ArgumentParser(description='Profile PPO.learn()')
    
    parser.add_argument('--use-gpu-profiler', action='store_true',
                        help='Use torch.profiler for GPU profiling')
    
    parser.add_argument('--dataset', type=str, default='family')
    parser.add_argument('--total-timesteps', type=int, default=1,
                        help='Total timesteps to profile (excluding warmup)')
    parser.add_argument('--batch-size-env', type=int, default=128,
                        help='Environment batch size')
    parser.add_argument('--n-steps', type=int, default=128,
                        help='Steps per rollout')
    parser.add_argument('--n-epochs', type=int, default=5,
                        help='PPO epochs per update')
    parser.add_argument('--batch-size', type=int, default=1024,
                        help='PPO minibatch size')
    
    parser.add_argument('--compile', default=True, type=lambda x: x.lower() != 'false',
                        help='Enable torch.compile for PPO and unification')
    parser.add_argument('--amp', default=True, type=lambda x: x.lower() != 'false')
    
    args = parser.parse_args()
    
    rollout_buffer_size = args.n_steps * args.batch_size_env
    if rollout_buffer_size % args.batch_size != 0:
        original = args.batch_size
        while rollout_buffer_size % args.batch_size != 0 and args.batch_size > 1:
            args.batch_size -= 1
        print(f"[WARNING] Adjusted batch_size from {original} to {args.batch_size}")
    
    config = SimpleNamespace(
        dataset=args.dataset,
        data_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'),
        total_timesteps=args.total_timesteps,
        batch_size_env=args.batch_size_env,
        n_envs=args.batch_size_env,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=5e-5,
        learning_rate=5e-5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.2,
        vf_coef=0.5,
        max_grad_norm=0.5,
        padding_atoms=6,
        padding_states={
            "countries_s3": 20, "countries_s2": 20, "countries_s1": 20,
            "family": 130, "wn18rr": 262, "fb15k237": 358,
        }.get(args.dataset, 150),
        max_depth=20,
        max_steps=20,
        end_proof_action=True,
        max_total_vars=100,
        atom_embedding_size=250,
        seed=42,
        use_amp=args.amp,
        verbose=True,
        parity=False,
        use_callbacks=True,  # Enable callbacks to test their impact
        log_per_depth=True,
        log_per_predicate=True,
        compile=args.compile,
    )
    
    print(f"Using padding_states={config.padding_states} for dataset={config.dataset}")
    
    if args.use_gpu_profiler:
        profile_gpu(config)
    else:
        profile_cprofile(config)


if __name__ == '__main__':
    main()
