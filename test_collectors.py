"""
Benchmark comparison: SyncDataCollector vs MultiSyncDataCollector

This script compares the performance of two TorchRL collectors:
1. SyncDataCollector - Single collector managing all environments
2. MultiSyncDataCollector - Multiple collectors, each managing a subset of environments
"""

import torch
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import DataHandler
from index_manager import IndexManager
from embeddings import EmbedderLearnable
from ppo.ppo_model import create_torchrl_modules
from neg_sampling import get_sampler
from ppo.ppo_rollout import MaskedPolicyWrapper, _reshape_time_env, _extract_episode_infos
from ppo.ppo_rollout_multisync import MultiSyncRolloutCollector
from torchrl.collectors import SyncDataCollector
from torchrl.envs import ExplorationType, set_exploration_type
from tensordict import TensorDict
from typing import List, Dict, Any, Tuple


def create_test_setup():
    """Create minimal test configuration."""
    class Args:
        def __init__(self):
            self.dataset_name = 'family'
            self.n_train_queries = None
            self.n_eval_queries = 500
            self.n_test_queries = None
            self.train_depth = None
            self.valid_depth = None
            self.test_depth = None
            self.prob_facts = False
            self.topk_facts = None
            self.topk_facts_threshold = 0.33
            self.model_emb_size = 100
            self.gamma = 0.99
            self.seed_run_i = 0
            self.n_envs = 4  # Use 4 envs for better MultiSync comparison
            self.n_steps = 256
            self.n_eval_envs = 4
            self.reward_type = 4
            self.train_neg_ratio = 1
            self.engine = 'python'
            self.engine_strategy = 'cmp'
            self.endf_action = True
            self.endt_action = False
            self.skip_unary_actions = True
            self.max_depth = 20
            self.memory_pruning = True
            self.corruption_mode = 'dynamic'
            self.corruption_scheme = ['head', 'tail']
            self.padding_atoms = 10
            self.padding_states = 20
            self.use_parallel_envs = True
            self.parallel_env_start_method = 'fork'  # Use fork to avoid CUDA tensor pickling issues
            self.max_total_vars = 10000
            self.kge_action = False
            self.pbrs = False
            self.pbrs_beta = 0.0
            self.pbrs_gamma = None
            self.verbose_env = 0
            self.verbose_prover = 0
    
    return Args()


def setup_environment():
    """Setup data, models, and environments."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = create_test_setup()
    
    # Load dataset
    base_path = '/home/castellanoontiv/Neural-guided-Grounding/data'
    data_handler = DataHandler(
        dataset_name=args.dataset_name,
        base_path=base_path,
        janus_file=f'{args.dataset_name}.pl',
        train_file='train.txt',
        valid_file='valid.txt',
        test_file='test.txt',
        rules_file='rules.txt',
        facts_file=f'{args.dataset_name}.pl',
        train_depth=args.train_depth,
        valid_depth=args.valid_depth,
        test_depth=args.test_depth,
        n_train_queries=args.n_train_queries,
        n_eval_queries=args.n_eval_queries,
        n_test_queries=args.n_test_queries,
        corruption_mode=args.corruption_mode,
        prob_facts=args.prob_facts,
        topk_facts=args.topk_facts,
        topk_facts_threshold=args.topk_facts_threshold,
    )
    
    # Create index manager
    index_manager = IndexManager(
        predicates=data_handler.predicates,
        constants=data_handler.constants,
        max_total_vars=args.max_total_vars,
        rules=data_handler.rules,
        max_arity=data_handler.max_arity,
        padding_atoms=args.padding_atoms,
        device=torch.device('cpu'),
    )
    index_manager.build_fact_index(data_handler.facts)
    
    # Create sampler on CPU (environments will run on CPU in worker processes)
    data_handler.sampler = get_sampler(
        data_handler=data_handler,
        index_manager=index_manager,
        corruption_scheme=args.corruption_scheme,
        device=torch.device('cpu'),  # Sampler must be on CPU for fork compatibility
    )
    
    # Create models
    embedder = EmbedderLearnable(
        n_predicates=index_manager.predicate_no,
        n_constants=index_manager.constant_no,
        n_vars=index_manager.variable_no,
        max_arity=data_handler.max_arity,
        padding_atoms=args.padding_atoms,
        atom_embedder='transe',
        state_embedder='sum',
        constant_embedding_size=args.model_emb_size,
        predicate_embedding_size=args.model_emb_size,
        atom_embedding_size=args.model_emb_size,
        device=device,
    )
    
    actor, critic = create_torchrl_modules(
        embedder=embedder,
        num_actions=args.padding_states,
        embed_dim=args.model_emb_size,
        hidden_dim=128,
        num_layers=8,
        dropout_prob=0.2,
        device=device,
        enable_kge_action=False,
        kge_inference_engine=None,
        index_manager=index_manager,
    )
    
    # Create environments
    from env_factory import create_environments
    train_env, eval_env, callback_env = create_environments(
        args, data_handler, index_manager,
        kge_engine=None, detailed_eval_env=False, device=device
    )
    
    return args, train_env, actor, critic, device


class SyncCollectorWrapper:
    """Wrapper for SyncDataCollector to match RolloutCollector interface."""
    
    def __init__(self, env, actor, n_envs, n_steps, device):
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.device = device
        
        frames_per_batch = int(n_envs) * int(n_steps)
        masked_policy = MaskedPolicyWrapper(actor, debug=False)
        
        self.collector = SyncDataCollector(
            create_env_fn=env,
            policy=masked_policy,
            frames_per_batch=frames_per_batch,
            total_frames=-1,
            device=device,
            storing_device=device,
            split_trajs=False,
            exploration_type=ExplorationType.RANDOM,
            init_random_frames=-1,
            use_buffers=False,
        )
        self._iterator = iter(self.collector)
    
    def collect(self, critic):
        with set_exploration_type(ExplorationType.RANDOM):
            batch_td = next(self._iterator)
        
        critic(batch_td)
        
        batch_td_time = _reshape_time_env(batch_td, n_steps=self.n_steps, n_envs=self.n_envs)
        experiences: List[TensorDict] = []
        for t in range(self.n_steps):
            step_td = TensorDict({
                "sub_index": batch_td_time[t]["sub_index"],
                "derived_sub_indices": batch_td_time[t]["derived_sub_indices"],
                "action_mask": batch_td_time[t]["action_mask"],
                "action": batch_td_time[t]["action"],
                "sample_log_prob": batch_td_time[t].get("sample_log_prob"),
                "state_value": batch_td_time[t].get("state_value"),
                "next": TensorDict({
                    "sub_index": batch_td_time[t]["next"]["sub_index"],
                    "derived_sub_indices": batch_td_time[t]["next"]["derived_sub_indices"],
                    "action_mask": batch_td_time[t]["next"]["action_mask"],
                    "reward": batch_td_time[t]["next"]["reward"],
                    "done": batch_td_time[t]["next"]["done"],
                }, batch_size=[self.n_envs]),
            }, batch_size=[self.n_envs])
            experiences.append(step_td)
        
        episode_info = _extract_episode_infos(
            batch_td, n_steps=self.n_steps, n_envs=self.n_envs,
            device=self.device, verbose=False
        )
        stats = {"episode_info": episode_info}
        
        return experiences, stats
    
    def shutdown(self):
        if hasattr(self, 'collector'):
            self.collector.shutdown()


class MultiSyncCollectorWrapper:
    """Wrapper for MultiSyncRolloutCollector to match RolloutCollector interface."""
    
    def __init__(self, env, actor, n_envs, n_steps, device, num_collectors=2):
        import time
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.device = device
        self.num_collectors = num_collectors
        
        # CRITICAL: Move actor to CPU before passing to MultiSync collector
        # Even with fork, CUDA tensors in multiprocessing cause issues
        # The actor will run on CPU in worker processes, environments are already on CPU
        actor_cpu = actor.to('cpu')
        
        # IMPORTANT: MultiSyncDataCollector will create its own worker processes using env_fns.
        # We MUST shut down the original ParallelEnv's workers to avoid conflicts.
        # Extract env_fns and shutdown the ParallelEnv completely
        if hasattr(env, 'env_fns'):
            env_fns = env.env_fns
            # Shutdown the original ParallelEnv workers BEFORE creating MultiSync
            try:
                env.shutdown()
                # Give it time to fully shutdown
                time.sleep(0.5)
            except Exception as e:
                print(f"Warning during env shutdown: {e}")
            # Use the env_fns directly
            env_to_pass = env_fns
        else:
            env_to_pass = env
        
        # MultiSyncRolloutCollector will handle env_fns or env appropriately
        self.collector = MultiSyncRolloutCollector(
            env=env_to_pass,
            actor=actor_cpu,  # Use CPU version of actor
            n_envs=n_envs,
            n_steps=n_steps,
            device=torch.device('cpu'),  # Collector runs on CPU
            num_sub_collectors=num_collectors,
            debug=False,  # Disable debug for cleaner output
        )
    
    def collect(self, critic):
        return self.collector.collect(critic=critic, rollout_callback=None)
    
    def shutdown(self):
        # Shutdown the MultiSync collector
        if hasattr(self, 'collector'):
            try:
                self.collector.shutdown()
            except Exception as e:
                print(f"Warning: Error shutting down MultiSync collector: {e}")


def benchmark_collector(collector_class, name, n_rollouts=5, **collector_kwargs):
    """Benchmark a specific collector implementation."""
    args, train_env, actor, critic, device = setup_environment()
    
    print(f"\n{'='*80}")
    print(f"BENCHMARK: {name}")
    print(f"{'='*80}")
    
    # Create collector
    print(f"Creating collector with {args.n_envs} environments...")
    collector = collector_class(
        env=train_env,
        actor=actor,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        device=device,
        **collector_kwargs
    )
    print("Collector created!\n")

    # Warmup
    print("Warming up...")
    collector.collect(critic=critic)
    
    # Benchmark
    print(f"Running {n_rollouts} rollouts...")
    times = []
    for i in range(n_rollouts):
        start = time.time()
        experiences, stats = collector.collect(critic=critic)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Rollout {i+1}: {elapsed:.3f}s")
    
    if not times:
        print("No successful rollouts!")
        collector.shutdown()
        # Only shutdown train_env for SyncDataCollector, not MultiSync
        if not isinstance(collector, MultiSyncCollectorWrapper):
            try:
                train_env.shutdown()
            except:
                pass
        return None
    
    avg_time = sum(times) / len(times)
    print(f"\nAverage time: {avg_time:.3f}s")
    print(f"Throughput: {args.n_envs * args.n_steps / avg_time:.1f} steps/sec")
    
    # Cleanup
    collector.shutdown()
    # Only shutdown train_env for SyncDataCollector, not MultiSync
    if not isinstance(collector, MultiSyncCollectorWrapper):
        try:
            train_env.shutdown()
        except:
            pass
    
    return avg_time


if __name__ == "__main__":
    print("\nCollector Performance Comparison: SyncDataCollector vs MultiSyncDataCollector")
    print("="*80)
    
    results = {}

    
    # Benchmark MultiSyncDataCollector with 4 sub-collectors
    multi4_time = benchmark_collector(
        MultiSyncCollectorWrapper,
        "MultiSyncDataCollector (4 sub-collectors)",
        n_rollouts=5,
        num_collectors=4
    )
    if multi4_time:
        results['MultiSync-4'] = multi4_time
    
    # Benchmark SyncDataCollector
    sync_time = benchmark_collector(
        SyncCollectorWrapper,
        "SyncDataCollector (Single collector for all environments)",
        n_rollouts=5
    )
    if sync_time:
        results['SyncDataCollector'] = sync_time


    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    if results:
        for name, avg_time in results.items():
            print(f"{name:40s}: {avg_time:.3f}s per rollout")
        
        if 'SyncDataCollector' in results:
            baseline = results['SyncDataCollector']
            print(f"\n{'='*80}")
            print("SPEEDUP vs SyncDataCollector:")
            print(f"{'='*80}")
            for name, avg_time in results.items():
                if name != 'SyncDataCollector':
                    speedup = baseline / avg_time
                    if speedup > 1:
                        print(f"{name:40s}: {speedup:.2f}x FASTER")
                    else:
                        print(f"{name:40s}: {1/speedup:.2f}x SLOWER")
    else:
        print("No successful benchmarks!")
    
    print(f"{'='*80}\n")
