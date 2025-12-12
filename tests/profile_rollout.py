#!/usr/bin/env python3
"""
Granular Rollout Collection Profiling Script.

This script uses torch.profiler with CUDA tracing to identify:
1. Kernel launch overhead
2. Memory transfer bottlenecks (CPU <-> GPU)
3. GPU idle time / synchronization points
4. Individual kernel execution times

Usage:
    python tests/profile_rollout.py [--n_steps N] [--n_envs N] [--warmup W]

Outputs:
    - tests/profile_rollout_results.txt: Human-readable summary
    - tests/profile_rollout_trace.json: Chrome trace viewer compatible (chrome://tracing)
"""

import sys
import os
import argparse
import torch
import time
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from torch.profiler import profile, record_function, ProfilerActivity, schedule
from tensordict import TensorDict

def setup_environment(n_envs: int, device: torch.device, dataset: str = "family"):
    """Setup environment and policy for profiling using the same flow as train.py."""
    from data_handler import DataHandler
    from index_manager import IndexManager
    from env import BatchedEnv
    from model import ActorCriticPolicy
    from unification import UnificationEngine
    from embeddings import EmbedderLearnable as TensorEmbedder
    from sampler import Sampler
    
    print(f"Setting up environment with {n_envs} parallel envs on {device}...")
    
    # Load data (matching train.py)
    dh = DataHandler(
        dataset_name=dataset,
        base_path="data",
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="train.txt",
    )
    
    # Create index manager (matching train.py)
    im = IndexManager(
        constants=dh.constants,
        predicates=dh.predicates,
        max_total_runtime_vars=100,
        max_arity=dh.max_arity,
        padding_atoms=6,
        device=device,
        rules=dh.rules,
    )
    
    # Materialize indices
    dh.materialize_indices(im=im, device=device)
    
    # Create sampler
    domain2idx, entity2domain = dh.get_sampler_domain_info()
    sampler = Sampler.from_data(
        all_known_triples_idx=dh.all_known_triples_idx,
        num_entities=im.constant_no,
        num_relations=im.predicate_no,
        device=device,
        default_mode="both",
        seed=42,
        domain2idx=domain2idx,
        entity2domain=entity2domain,
    )
    
    # Create stringifier params (no verbose or padding_idx - env passes those to DebugHelper explicitly)
    stringifier_params = {
        'idx2predicate': im.idx2predicate,
        'idx2constant': im.idx2constant,
        'idx2template_var': im.idx2template_var,
        'n_constants': im.constant_no
    }
    
    # Create unification engine
    engine = UnificationEngine.from_index_manager(
        im, take_ownership=True,
        stringifier_params=stringifier_params,
        end_pred_idx=im.end_pred_idx,
        end_proof_action=True,
        max_derived_per_state=130,  # Matching family dataset padding_states
    )
    engine.index_manager = im
    
    # Get train queries
    train_split = dh.get_materialized_split('train')
    train_queries_tensor = train_split.queries
    
    # Create environment
    env = BatchedEnv(
        batch_size=n_envs,
        queries=train_queries_tensor,
        labels=torch.ones(len(dh.train_queries), dtype=torch.long, device=device),
        query_depths=torch.as_tensor(dh.train_depths, dtype=torch.long, device=device),
        unification_engine=engine,
        mode='train',
        max_depth=20,
        memory_pruning=True,
        use_exact_memory=False,
        skip_unary_actions=True,
        end_proof_action=True,
        reward_type=4,
        padding_atoms=6,
        padding_states=130,
        true_pred_idx=im.predicate_str2idx.get('True'),
        false_pred_idx=im.predicate_str2idx.get('False'),
        end_pred_idx=im.predicate_str2idx.get('Endf'),
        verbose=0,
        prover_verbose=False,
        device=device,
        runtime_var_start_index=im.constant_no + 1,
        total_vocab_size=im.constant_no + 100,
        sample_deterministic_per_env=False,
        sampler=sampler,
        train_neg_ratio=1,
        corruption_mode=True,
        stringifier_params=stringifier_params,
    )
    
    # Create embedder
    embedder = TensorEmbedder(
        n_constants=im.constant_no,
        n_predicates=im.predicate_no,
        n_vars=im.variable_no,
        max_arity=dh.max_arity,
        padding_atoms=6,
        atom_embedder='transe',
        state_embedder='mean',
        constant_embedding_size=250,
        predicate_embedding_size=250,
        atom_embedding_size=250,
        device=str(device),
    )
    
    # Create policy
    policy = ActorCriticPolicy(
        embedder=embedder,
        embed_dim=250,
        action_dim=130,
        hidden_dim=256,
        num_layers=8,
        dropout_prob=0.1,
        device=device,
        temperature=0.1,
        use_l2_norm=True,
        sqrt_scale=False,
    ).to(device)
    
    return env, policy, im


def profile_rollout_collection(
    n_steps: int = 32,
    n_envs: int = 64,
    warmup_steps: int = 5,
    device: torch.device = None,
    output_dir: str = "tests"
):
    """
    Profile rollout collection with granular CUDA tracing.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 70)
    print("ROLLOUT COLLECTION PROFILING")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Steps: {n_steps}, Envs: {n_envs}, Warmup: {warmup_steps}")
    print()
    
    # Setup
    env, policy, im = setup_environment(n_envs, device)
    policy.eval()
    
    # Compile policy for realistic profiling
    print("Compiling policy...")
    policy = torch.compile(policy, mode='reduce-overhead', fullgraph=True)
    
    # Warmup to trigger compilation
    print(f"Running {warmup_steps} warmup steps...")
    current_obs = env.reset()
    for _ in range(warmup_steps):
        with torch.no_grad():
            actions, _, _ = policy(current_obs, deterministic=False)
        action_td = TensorDict({"action": actions}, batch_size=current_obs.batch_size, device=device)
        _, current_obs = env.step_and_maybe_reset(action_td)
    
    torch.cuda.synchronize()
    print("Warmup complete.\n")
    
    # Reset for actual profiling
    current_obs = env.reset()
    episode_starts = torch.ones(n_envs, dtype=torch.float32, device=device)
    dones = torch.zeros(n_envs, dtype=torch.bool, device=device)
    
    # Profiler configuration
    # wait=1: Skip first step (may have startup overhead)
    # warmup=1: Include one warmup iteration in profiling context
    # active=n_steps-2: Profile remaining steps
    # repeat=1: Only profile once
    prof_schedule = schedule(
        wait=1,
        warmup=1,
        active=n_steps - 2,
        repeat=1
    )
    
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    
    print(f"Starting profiling for {n_steps} steps...")
    start_time = time.perf_counter()
    
    with profile(
        activities=activities,
        schedule=prof_schedule,
        # DISABLED: tensorboard handler is memory-heavy
        # on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
        record_shapes=False,  # Reduced for memory
        profile_memory=False,  # Reduced for memory
        with_stack=False,     # Reduced for memory
        with_flops=False,     # Reduced for memory
    ) as prof:
        
        with torch.no_grad():
            for step in range(n_steps):
                # =============================================
                # PHASE 1: Observation Preparation
                # =============================================
                with record_function("01_obs_clone"):
                    obs_snapshot = TensorDict(
                        {k: current_obs[k].clone() for k in ("sub_index", "derived_sub_indices", "action_mask") if k in current_obs.keys()},
                        batch_size=current_obs.batch_size,
                        device=current_obs.device
                    )
                
                # =============================================
                # PHASE 2: Policy Forward Pass
                # =============================================
                with record_function("02_policy_forward"):
                    actions, values, log_probs = policy(obs_snapshot, deterministic=False)
                
                # =============================================
                # PHASE 3: Action TensorDict Creation
                # =============================================
                with record_function("03_action_td_create"):
                    action_td = TensorDict(
                        {"action": actions},
                        batch_size=current_obs.batch_size,
                        device=device
                    )
                
                # =============================================
                # PHASE 4: Environment Step (includes unification!)
                # =============================================
                with record_function("04_env_step"):
                    step_result, next_obs = env.step_and_maybe_reset(action_td)
                
                # =============================================
                # PHASE 5: Extract Results
                # =============================================
                with record_function("05_extract_results"):
                    step_info = step_result.get("next", step_result)
                    rewards = step_info.get("reward", torch.zeros(n_envs, device=device))
                    dones = step_info.get("done", torch.zeros(n_envs, dtype=torch.bool, device=device))
                    if rewards.dim() > 1:
                        rewards = rewards.squeeze(-1)
                    if dones.dim() > 1:
                        dones = dones.squeeze(-1)
                
                # =============================================
                # PHASE 6: Episode Tracking
                # =============================================
                with record_function("06_episode_tracking"):
                    if dones.any():
                        episode_starts = dones.float()
                    else:
                        episode_starts = torch.zeros(n_envs, dtype=torch.float32, device=device)
                
                current_obs = next_obs
                
                # Step the profiler
                prof.step()
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    fps = (n_steps * n_envs) / elapsed
    
    print(f"\nProfiling complete!")
    print(f"Total time: {elapsed:.2f}s")
    print(f"FPS: {fps:.1f}")
    print()
    
    # =========================================================================
    # Generate Human-Readable Summary
    # =========================================================================
    output_file = os.path.join(output_dir, "profile_rollout_results.txt")
    
    with open(output_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("ROLLOUT COLLECTION PROFILING RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Device: {device}\n")
        f.write(f"  Steps: {n_steps}\n")
        f.write(f"  Envs: {n_envs}\n")
        f.write(f"  Total samples: {n_steps * n_envs}\n")
        f.write(f"  Total time: {elapsed:.2f}s\n")
        f.write(f"  FPS: {fps:.1f}\n\n")
        
        # =====================================================================
        # BY CUSTOM PHASES (record_function)
        # =====================================================================
        f.write("=" * 80 + "\n")
        f.write("BREAKDOWN BY PHASE (Custom record_function regions)\n")
        f.write("=" * 80 + "\n\n")
        
        # Filter for our custom phases
        phase_averages = prof.key_averages()
        custom_phases = [e for e in phase_averages if e.key.startswith("0")]
        
        for event in sorted(custom_phases, key=lambda x: x.key):
            cpu_time_ms = event.cpu_time_total / 1000  # Convert to ms
            cuda_time_ms = event.device_time_total / 1000 if hasattr(event, 'device_time_total') else 0
            count = event.count
            
            # Per-call averages
            cpu_avg = cpu_time_ms / count if count > 0 else 0
            cuda_avg = cuda_time_ms / count if count > 0 else 0
            
            f.write(f"{event.key}:\n")
            f.write(f"  Count: {count}\n")
            f.write(f"  Total CPU time: {cpu_time_ms:.2f} ms\n")
            f.write(f"  Total CUDA time: {cuda_time_ms:.2f} ms\n")
            f.write(f"  Avg CPU per call: {cpu_avg:.3f} ms\n")
            f.write(f"  Avg CUDA per call: {cuda_avg:.3f} ms\n\n")
        
        # =====================================================================
        # TOP KERNELS BY CUDA TIME
        # =====================================================================
        f.write("=" * 80 + "\n")
        f.write("TOP 30 CUDA KERNELS BY TOTAL DEVICE TIME\n")
        f.write("=" * 80 + "\n\n")
        
        cuda_events = [e for e in phase_averages if hasattr(e, 'device_time') and e.device_time > 0]
        cuda_events_sorted = sorted(cuda_events, key=lambda x: x.device_time_total, reverse=True)[:30]
        
        for event in cuda_events_sorted:
            cuda_time_ms = event.device_time_total / 1000
            count = event.count
            avg_ms = cuda_time_ms / count if count > 0 else 0
            f.write(f"{event.key[:70]:<70} {cuda_time_ms:>8.2f} ms (x{count}, avg {avg_ms:.3f} ms)\n")
        
        f.write("\n")
        
        # =====================================================================
        # TOP OPERATORS BY CPU TIME
        # =====================================================================
        f.write("=" * 80 + "\n")
        f.write("TOP 30 OPERATORS BY CPU TIME\n")
        f.write("=" * 80 + "\n\n")
        
        cpu_events_sorted = sorted(phase_averages, key=lambda x: x.cpu_time_total, reverse=True)[:30]
        
        for event in cpu_events_sorted:
            cpu_time_ms = event.cpu_time_total / 1000
            count = event.count
            avg_ms = cpu_time_ms / count if count > 0 else 0
            f.write(f"{event.key[:70]:<70} {cpu_time_ms:>8.2f} ms (x{count}, avg {avg_ms:.3f} ms)\n")
        
        f.write("\n")
        
        # =====================================================================
        # MEMORY SUMMARY
        # =====================================================================
        f.write("=" * 80 + "\n")
        f.write("MEMORY OPERATIONS (TOP 20 BY ALLOCATION)\n")
        f.write("=" * 80 + "\n\n")
        
        mem_events = [e for e in phase_averages if hasattr(e, 'cpu_memory_usage') and e.cpu_memory_usage > 0]
        mem_sorted = sorted(mem_events, key=lambda x: x.cpu_memory_usage, reverse=True)[:20]
        
        for event in mem_sorted:
            mem_kb = event.cpu_memory_usage / 1024  # Convert to KB
            count = event.count
            f.write(f"{event.key[:50]:<50} {mem_kb:>10.1f} KB (x{count})\n")
        
        f.write("\n")
        
        # =====================================================================
        # SYNCHRONIZATION ANALYSIS
        # =====================================================================
        f.write("=" * 80 + "\n")
        f.write("POTENTIAL SYNCHRONIZATION POINTS\n")
        f.write("=" * 80 + "\n\n")
        
        sync_keywords = ["cudaStreamSynchronize", "cudaDeviceSynchronize", "item", "nonzero", "synchronize"]
        sync_events = [e for e in phase_averages if any(kw in e.key.lower() for kw in sync_keywords)]
        
        if sync_events:
            for event in sorted(sync_events, key=lambda x: x.cpu_time_total, reverse=True):
                cpu_time_ms = event.cpu_time_total / 1000
                count = event.count
                f.write(f"{event.key[:60]:<60} {cpu_time_ms:>8.2f} ms (x{count})\n")
        else:
            f.write("No explicit sync operations detected (good!)\n")
        
        f.write("\n")
        
        # =====================================================================
        # KERNEL LAUNCH OVERHEAD ESTIMATE
        # =====================================================================
        f.write("=" * 80 + "\n")
        f.write("KERNEL LAUNCH ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        # Count total CUDA kernel launches
        total_cuda_launches = sum(e.count for e in cuda_events)
        total_cuda_time = sum(e.device_time_total for e in cuda_events) / 1000  # ms
        
        if total_cuda_launches > 0:
            avg_kernel_time = total_cuda_time / total_cuda_launches
            f.write(f"Total CUDA kernel launches: {total_cuda_launches}\n")
            f.write(f"Total CUDA kernel time: {total_cuda_time:.2f} ms\n")
            f.write(f"Average kernel duration: {avg_kernel_time:.4f} ms\n\n")
            
            # Kernels under 0.1ms are likely launch-overhead-bound
            short_kernels = [e for e in cuda_events if (e.device_time_total / e.count) < 100]  # <0.1ms avg
            short_launches = sum(e.count for e in short_kernels)
            short_time = sum(e.device_time_total for e in short_kernels) / 1000
            
            f.write(f"Short kernels (<0.1ms avg):\n")
            f.write(f"  Launches: {short_launches} ({100*short_launches/total_cuda_launches:.1f}%)\n")
            f.write(f"  Time: {short_time:.2f} ms ({100*short_time/total_cuda_time:.1f}%)\n")
            f.write(f"  -> These may benefit from kernel fusion or CUDA graphs\n")
    
    print(f"Results written to: {output_file}")
    print(f"TensorBoard trace written to: {output_dir}")
    print(f"\nTo view detailed trace:")
    print(f"  tensorboard --logdir={output_dir}")
    print(f"  Or load .json file in chrome://tracing")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Profile rollout collection with CUDA tracing")
    parser.add_argument("--n_steps", type=int, default=32, help="Number of rollout steps")
    parser.add_argument("--n_envs", type=int, default=64, help="Number of parallel environments")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup steps before profiling")
    parser.add_argument("--output_dir", type=str, default="tests", help="Output directory")
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Profiling will be limited to CPU.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    profile_rollout_collection(
        n_steps=args.n_steps,
        n_envs=args.n_envs,
        warmup_steps=args.warmup,
        device=device,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
