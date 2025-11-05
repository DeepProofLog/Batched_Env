"""
Deep profiling of rollout collection to identify actual bottlenecks.

This script can profile different collector implementations:
- SyncDataCollector (TorchRL's single-collector implementation)
- MultiSyncDataCollector (TorchRL's multi-collector implementation)
- CustomRolloutCollector (Pure Python implementation without TorchRL collectors)

Usage:
    python test_rollout_profile.py [collector_type]
    
    collector_type: 'sync' (default), 'multisync', or 'custom'
"""

import torch
import time
import sys
import os
import cProfile
import pstats
from io import StringIO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import DataHandler
from index_manager import IndexManager
from embeddings import EmbedderLearnable
from ppo.ppo_model import create_torchrl_modules
from neg_sampling import get_sampler


def create_test_args():
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
            self.model_emb_size = 256
            self.gamma = 0.99
            self.seed_run_i = 0
            self.n_steps = 2048
            self.n_envs = 8  # Back to 8 for fair comparison
            self.n_eval_envs = 8
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
            self.parallel_env_start_method = 'fork'  # Use fork for CUDA compatibility
            self.max_total_vars = 10000
            self.kge_action = False
            self.pbrs = False
            self.pbrs_beta = 0.0
            self.pbrs_gamma = None
            self.verbose_env = 0
            self.verbose_prover = 0
    
    return Args()


def setup_test_environment():
    """Setup data, models, and environments for testing."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = create_test_args()
    
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
    
    # Create sampler
    data_handler.sampler = get_sampler(
        data_handler=data_handler,
        index_manager=index_manager,
        corruption_scheme=args.corruption_scheme,
        device=torch.device('cpu'),
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
        kge_engine=None, detailed_eval_env=False, device='cpu'
    )
    
    return args, train_env, eval_env, callback_env, actor, critic, device


def profile_rollout(collector_type: str = 'sync'):
    """Profile a single rollout collection.
    
    Args:
        collector_type: Type of collector to profile ('sync', 'multisync', or 'custom')
    
    Returns:
        float: Time taken for the rollout collection
    """
    if collector_type == 'sync':
        from ppo.ppo_rollout_sync import RolloutCollector
        print(f"\n{'='*80}")
        print(f"PROFILING: TorchRL SyncDataCollector")
        print(f"{'='*80}\n")
    elif collector_type == 'multisync':
        from ppo.ppo_rollout_multisync import MultiSyncRolloutCollector as RolloutCollector
        print(f"\n{'='*80}")
        print(f"PROFILING: TorchRL MultiSyncDataCollector")
        print(f"{'='*80}\n")
    elif collector_type == 'custom':
        from ppo.ppo_rollout_custom import CustomRolloutCollector as RolloutCollector
        print(f"\n{'='*80}")
        print(f"PROFILING: Custom Python Collector (No TorchRL)")
        print(f"{'='*80}\n")
    else:
        raise ValueError(f"Unknown collector type: {collector_type}. Use 'sync', 'multisync', or 'custom'")
    
    args, train_env, eval_env, callback_env, actor, critic, device = setup_test_environment()
    
    # For custom collector, we can pass the ParallelEnv directly
    # For MultiSync, we need to extract env_fns and move actor to CPU
    if collector_type == 'multisync':
        actor_cpu = actor.to('cpu')
        if hasattr(train_env, 'env_fns'):
            env_fns = train_env.env_fns
            try:
                train_env.shutdown()
                time.sleep(0.5)
            except Exception as e:
                print(f"Warning during env shutdown: {e}")
            env_to_use = env_fns
        else:
            env_to_use = train_env
        actor_to_use = actor_cpu
        device_to_use = torch.device('cpu')
    else:
        env_to_use = train_env
        actor_to_use = actor
        device_to_use = torch.device('cpu')
    
    # Create persistent collector ONCE
    print("Creating persistent collector...")
    print(f"Device: {device}, Collector device: {device_to_use}")
    collector = RolloutCollector(
        env=env_to_use,
        actor=actor_to_use,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        device=device_to_use,
        debug=False,
    )
    print("Collector created!\n")
    
    # # Warm up
    # print("Warming up...")
    # for _ in range(2):
    #     collector.collect(
    #         critic=critic,
    #         rollout_callback=None,
    #     )
    
    print("Running profiled rollout collection...")
    
    # Profile the rollout collection (reusing the same collector)
    profiler = cProfile.Profile()
    profiler.enable()
    
    start_time = time.time()
    experiences, stats = collector.collect(
        critic=critic,
        rollout_callback=None,
    )
    end_time = time.time()
    rollout_time = end_time - start_time
    
    profiler.disable()
    
    # Print profiling results
    print("\n" + "="*80)
    print("PROFILING RESULTS - Top 30 Time-Consuming Functions")
    print("="*80)
    
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats('cumulative')
    ps.print_stats(30)
    print(s.getvalue())
    
    print("\n" + "="*80)
    print("PROFILING RESULTS - Top 30 by Total Time")
    print("="*80)
    
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats('tottime')
    ps.print_stats(30)
    print(s.getvalue())
    
    print(f"\nRollout collection time: {rollout_time:.3f} seconds")
    print(f"Throughput: {args.n_envs * args.n_steps / rollout_time:.1f} steps/sec")
    
    # Cleanup
    collector.shutdown()
    # Don't shutdown train_env - already shutdown before creating collector
    
    return rollout_time


if __name__ == "__main__":
    import sys
    
    # Check if specific collector type is requested
    if len(sys.argv) > 1:
        collector_type = sys.argv[1].lower()
        if collector_type not in ['sync', 'multisync', 'custom']:
            print(f"Error: Unknown collector type '{sys.argv[1]}'")
            print("Usage: python test_rollout_profile.py [sync|multisync|custom]")
            print("Or run without arguments to profile the sync collector")
            sys.exit(1)
        
        # Profile single collector
        profile_rollout(collector_type)
    else:
        # Default to profiling sync collector
        # profile_rollout('sync')
        profile_rollout('custom')
