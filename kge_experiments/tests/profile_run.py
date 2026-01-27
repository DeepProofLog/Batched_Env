"""
Profile a short full training run to identify overall bottlenecks.

This script runs a complete training pipeline (init, rollout, train, eval)
and provides detailed timing breakdown for each phase.

Usage:
    conda activate rl
    python kge_experiments/tests/profile_run.py
    python kge_experiments/tests/profile_run.py --use-gpu-profiler
    python kge_experiments/tests/profile_run.py --dataset countries_s3 --timesteps 5000
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import cProfile
import pstats
import io
from datetime import datetime
from time import time
from contextlib import contextmanager
from typing import Dict, Any, List, Tuple

import torch

try:
    from torch.profiler import profile, ProfilerActivity
except (ImportError, ModuleNotFoundError):
    profile = None
    ProfilerActivity = None


class TimingTracker:
    """Track timing for different phases of execution."""

    def __init__(self):
        self.phases: Dict[str, List[float]] = {}
        self.current_phase: str = None
        self.phase_start: float = None

    @contextmanager
    def phase(self, name: str):
        """Context manager to time a phase."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time()
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time() - start
            if name not in self.phases:
                self.phases[name] = []
            self.phases[name].append(elapsed)

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get timing summary for all phases."""
        summary = {}
        for name, times in self.phases.items():
            summary[name] = {
                'total': sum(times),
                'count': len(times),
                'avg': sum(times) / len(times) if times else 0,
                'min': min(times) if times else 0,
                'max': max(times) if times else 0,
            }
        return summary

    def print_summary(self, title: str = "TIMING BREAKDOWN"):
        """Print formatted timing summary."""
        summary = self.get_summary()
        total_time = sum(s['total'] for s in summary.values())

        print(f"\n{'='*80}")
        print(title)
        print(f"{'='*80}")
        print(f"{'Phase':<30} {'Total':>10} {'Count':>8} {'Avg':>10} {'%':>8}")
        print(f"{'-'*80}")

        # Sort by total time descending
        sorted_phases = sorted(summary.items(), key=lambda x: x[1]['total'], reverse=True)

        for name, stats in sorted_phases:
            pct = (stats['total'] / total_time * 100) if total_time > 0 else 0
            print(f"{name:<30} {stats['total']:>10.4f}s {stats['count']:>8} {stats['avg']:>10.4f}s {pct:>7.1f}%")

        print(f"{'-'*80}")
        print(f"{'TOTAL':<30} {total_time:>10.4f}s")


class Tee:
    """Write to multiple outputs."""
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


def create_config(args) -> 'TrainConfig':
    """Create TrainConfig from command line arguments."""
    from config import TrainConfig

    config = TrainConfig(
        dataset=args.dataset,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=3e-4,
        seed=42,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_callbacks=False,  # Disable callbacks for cleaner profiling
        verbose=False,
        # Eval settings
        eval_freq=0,  # No periodic eval
        test_neg_samples=args.eval_corruptions,
        n_test_queries=args.n_eval_queries,
    )

    return config


def profile_full_run(args):
    """Profile a full training run with phase-by-phase breakdown."""
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'profile_run.txt')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    log_file = open(output_path, 'w')
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = sys.stdout

    tracker = TimingTracker()

    try:
        print(f"Profile Full Run Results")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        print(f"\nConfiguration:")
        print(f"  Dataset: {args.dataset}")
        print(f"  Total timesteps: {args.timesteps}")
        print(f"  N envs: {args.n_envs}")
        print(f"  N steps: {args.n_steps}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Eval queries: {args.n_eval_queries}")
        print(f"  Eval corruptions: {args.eval_corruptions}")

        # Import after path setup
        from train import create_components
        from ppo import PPO
        from config import TrainConfig

        config = create_config(args)
        device = torch.device(config.device)

        # Phase 1: Initialization
        print("\n" + "="*80)
        print("PHASE 1: INITIALIZATION")
        print("="*80)

        with tracker.phase("1.1 create_components"):
            comp = create_components(config)

        policy = comp['policy']
        env = comp['env']
        im = comp['im']
        dh = comp['dh']
        sampler = comp['sampler']

        with tracker.phase("1.2 create_ppo"):
            ppo = PPO(policy, env, config)

        print(f"Components created. Policy params: {sum(p.numel() for p in policy.parameters()):,}")

        # Phase 2: Warmup (compilation)
        print("\n" + "="*80)
        print("PHASE 2: WARMUP (COMPILATION)")
        print("="*80)

        with tracker.phase("2.1 warmup_rollout"):
            # Run one rollout to compile
            ppo.learn(total_timesteps=config.n_steps * config.n_envs)

        print("Warmup complete - CUDA graphs compiled")

        # Phase 3: Training with detailed breakdown
        print("\n" + "="*80)
        print("PHASE 3: TRAINING (DETAILED BREAKDOWN)")
        print("="*80)

        steps_per_iter = config.n_steps * config.n_envs
        n_iterations = max(1, args.timesteps // steps_per_iter)
        total_train_steps = n_iterations * steps_per_iter

        print(f"Running {n_iterations} training iterations ({steps_per_iter} steps each)")
        print(f"Total training steps: {total_train_steps}")

        # Manual iteration to separate rollout vs update timing
        env.train()
        reset_res = env.reset()
        if isinstance(reset_res, tuple):
            obs, state = reset_res
        else:
            obs, state = reset_res, reset_res

        ep_starts = torch.ones(config.n_envs, device=device)
        curr_ep_rew = torch.zeros(config.n_envs, device=device)
        curr_ep_len = torch.zeros(config.n_envs, dtype=torch.long, device=device)
        ep_rews, ep_lens = [], []

        profiler = cProfile.Profile()
        profiler.enable()

        with tracker.phase("3.0 training_total"):
            for iteration in range(n_iterations):
                # Rollout collection
                with tracker.phase("3.1 rollout_collect"):
                    result = ppo.collect_rollouts(
                        state, obs, ep_starts, curr_ep_rew, curr_ep_len,
                        ep_rews, ep_lens, iteration, return_traces=False, on_step_callback=None
                    )
                    state, obs, ep_starts, curr_ep_rew, curr_ep_len, n_steps_collected, _ = result
                    state = state.clone()
                    obs = {k: v.clone() for k, v in obs.items()}

                # PPO update
                with tracker.phase("3.2 ppo_update"):
                    ppo.train(return_traces=False)

                ppo.num_timesteps += steps_per_iter

        profiler.disable()

        train_time = tracker.phases.get("3.0 training_total", [0])[0]
        rollout_time = sum(tracker.phases.get("3.1 rollout_collect", [0]))
        update_time = sum(tracker.phases.get("3.2 ppo_update", [0]))
        train_sps = total_train_steps / train_time if train_time > 0 else 0

        print(f"\nTraining breakdown:")
        print(f"  Rollout collection: {rollout_time:.2f}s ({rollout_time/train_time*100:.1f}%)")
        print(f"  PPO update:         {update_time:.2f}s ({update_time/train_time*100:.1f}%)")
        print(f"  Total:              {train_time:.2f}s ({train_sps:.0f} steps/sec)")

        # Phase 4: Evaluation
        print("\n" + "="*80)
        print("PHASE 4: EVALUATION")
        print("="*80)

        test_queries = dh.test_queries[:args.n_eval_queries]
        queries_tensor = im.queries_to_tensor(test_queries, device)

        policy.eval()

        with tracker.phase("4.1 eval_warmup"):
            # Eval warmup
            ppo.evaluate(
                queries=queries_tensor[:5],
                sampler=sampler,
                n_corruptions=5,
                corruption_modes=('head',),
                verbose=False,
            )

        with tracker.phase("4.2 eval_full"):
            eval_results = ppo.evaluate(
                queries=queries_tensor,
                sampler=sampler,
                n_corruptions=args.eval_corruptions,
                corruption_modes=('head', 'tail'),
                verbose=False,
            )

        mrr = eval_results.get('MRR', 0.0)
        hits1 = eval_results.get('Hits@1', 0.0)

        print(f"\nEval results: MRR={mrr:.4f}, Hits@1={hits1:.4f}")

        # Print timing summary
        tracker.print_summary()

        # Print cProfile results
        print(f"\n{'='*80}")
        print("CPROFILE - Top Functions by Cumulative Time")
        print(f"{'='*80}")
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.strip_dirs()
        ps.sort_stats('cumulative')
        ps.print_stats(30)
        print(s.getvalue())

        print(f"\n{'='*80}")
        print("CPROFILE - Top Functions by Total Time")
        print(f"{'='*80}")
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.strip_dirs()
        ps.sort_stats('tottime')
        ps.print_stats(30)
        print(s.getvalue())

        # GPU memory stats
        if torch.cuda.is_available():
            print(f"\n{'='*80}")
            print("GPU MEMORY")
            print(f"{'='*80}")
            print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"Reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")
            print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

        print(f"\nResults saved to {output_path}")

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


def profile_gpu_full_run(args):
    """Profile with torch.profiler for detailed GPU analysis."""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    if profile is None:
        print("torch.profiler not available")
        return

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'profile_run_gpu.txt')
    trace_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'profile_run_trace.json')

    log_file = open(output_path, 'w')
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = sys.stdout

    try:
        print(f"Profile Full Run GPU Results")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

        from train import create_components
        from ppo import PPO

        config = create_config(args)
        device = torch.device(config.device)

        print("\nCreating components...")
        comp = create_components(config)
        policy = comp['policy']
        env = comp['env']
        im = comp['im']
        dh = comp['dh']
        sampler = comp['sampler']

        ppo = PPO(policy, env, config)

        print("Warmup...")
        ppo.learn(total_timesteps=config.n_steps * config.n_envs)
        torch.cuda.synchronize()

        print(f"\nGPU Profiling {args.timesteps} training steps...")

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            torch.cuda.synchronize()
            start = time()

            ppo.learn(
                total_timesteps=args.timesteps,
                reset_num_timesteps=False,
            )

            torch.cuda.synchronize()
            runtime = time() - start

        steps_per_sec = args.timesteps / runtime if runtime > 0 else 0
        print(f"\nRuntime: {runtime:.2f}s ({steps_per_sec:.0f} steps/sec)")

        print(f"\n{'='*80}")
        print("GPU PROFILING - Top by CUDA Time")
        print(f"{'='*80}")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=40))

        print(f"\n{'='*80}")
        print("GPU PROFILING - Top by CPU Time")
        print(f"{'='*80}")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=40))

        print(f"\n{'='*80}")
        print("GPU PROFILING - Top by Memory")
        print(f"{'='*80}")
        print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))

        # Export trace for chrome://tracing
        prof.export_chrome_trace(trace_path)
        print(f"\nChrome trace saved to {trace_path}")
        print("Open chrome://tracing and load the trace file for visual analysis")

        print(f"\nResults saved to {output_path}")

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


def main():
    parser = argparse.ArgumentParser(description='Profile full training run')

    parser.add_argument('--use-gpu-profiler', action='store_true',
                        help='Use torch.profiler for detailed GPU analysis')

    # Dataset and training
    parser.add_argument('--dataset', type=str, default='family',
                        choices=['family', 'countries_s1', 'countries_s2', 'countries_s3', 'wn18rr', 'fb15k237'])
    parser.add_argument('--timesteps', type=int, default=10000,
                        help='Total training timesteps')
    parser.add_argument('--n-envs', type=int, default=128,
                        help='Number of parallel environments')
    parser.add_argument('--n-steps', type=int, default=128,
                        help='Steps per rollout')
    parser.add_argument('--n-epochs', type=int, default=5,
                        help='PPO epochs per update')
    parser.add_argument('--batch-size', type=int, default=1024,
                        help='PPO minibatch size')

    # Evaluation
    parser.add_argument('--n-eval-queries', type=int, default=50,
                        help='Number of queries to evaluate')
    parser.add_argument('--eval-corruptions', type=int, default=20,
                        help='Number of corruptions per query')

    args = parser.parse_args()

    # Adjust batch size if needed
    rollout_size = args.n_steps * args.n_envs
    if rollout_size % args.batch_size != 0:
        original = args.batch_size
        while rollout_size % args.batch_size != 0 and args.batch_size > 1:
            args.batch_size -= 1
        print(f"[INFO] Adjusted batch_size from {original} to {args.batch_size}")

    if args.use_gpu_profiler:
        profile_gpu_full_run(args)
    else:
        profile_full_run(args)


if __name__ == '__main__':
    main()
