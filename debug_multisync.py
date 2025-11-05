"""Minimal script to debug MultiSyncDataCollector worker failures."""
import sys
import torch
import traceback
from torchrl.collectors import MultiSyncDataCollector
from torchrl.envs import ParallelEnv, ExplorationType

sys.path.insert(0, '/home/castellanoontiv/Neural-guided-Grounding')

from dataset import DataHandler
from index_manager import IndexManager
from embeddings import EmbedderLearnable
from ppo.ppo_model import create_torchrl_modules
from neg_sampling import get_sampler
from ppo.ppo_rollout_multisync import MaskedPolicyWrapper


def create_minimal_setup():
    """Create minimal test configuration."""
    class Args:
        dataset_name = 'family'
        n_train_queries = None
        n_eval_queries = 500
        n_test_queries = None
        train_depth = None
        valid_depth = None
        test_depth = None
        prob_facts = False
        topk_facts = None
        topk_facts_threshold = 0.33
        model_emb_size = 100
        gamma = 0.99
        seed_run_i = 0
        n_envs = 4
        n_steps = 256
        n_eval_envs = 2
        reward_type = 4
        train_neg_ratio = 1
        engine = 'python'
        engine_strategy = 'cmp'
        endf_action = True
        endt_action = False
        skip_unary_actions = True
        max_depth = 20
        memory_pruning = True
        corruption_mode = 'dynamic'
        corruption_scheme = ['head', 'tail']
        padding_atoms = 10
        padding_states = 20
        use_parallel_envs = True
        parallel_env_start_method = 'spawn'
        max_total_vars = 10000
        kge_action = False
        pbrs = False
        pbrs_beta = 0.0
        pbrs_gamma = None
        verbose_env = 0
        verbose_prover = 0
        
    return Args()


def main():
    device = torch.device('cpu')  # Use CPU to avoid CUDA pickling issues
    args = create_minimal_setup()
    
    print("Loading dataset...")
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
    
    print("Creating index manager...")
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
    
    print("Creating sampler...")
    data_handler.sampler = get_sampler(
        data_handler=data_handler,
        index_manager=index_manager,
        corruption_scheme=args.corruption_scheme,
        device=device,
    )
    
    print("Creating models...")
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
    
    print("Creating environments...")
    from env_factory import create_environments
    train_env, eval_env, callback_env = create_environments(
        args, data_handler, index_manager,
        kge_engine=None, detailed_eval_env=False, device=device
    )
    
    print("\n=== Attempting to create MultiSyncDataCollector ===")
    print(f"Environment type: {type(train_env)}")
    print(f"Has env_fns: {hasattr(train_env, 'env_fns')}")
    
    if isinstance(train_env, ParallelEnv) and hasattr(train_env, 'env_fns'):
        create_env_fn = train_env.env_fns
        print(f"Number of env_fns: {len(create_env_fn)}")
        print(f"Type of env_fn[0]: {type(create_env_fn[0])}")
    else:
        create_env_fn = train_env
        print(f"Using env directly")
    
    frames_per_batch = args.n_envs * args.n_steps
    masked_policy = MaskedPolicyWrapper(actor, debug=True)
    
    print(f"\nCreating MultiSyncDataCollector with:")
    print(f"  frames_per_batch: {frames_per_batch}")
    print(f"  num_workers: 2")
    print(f"  device: {device}")
    
    try:
        collector = MultiSyncDataCollector(
            create_env_fn=create_env_fn,
            policy=masked_policy,
            frames_per_batch=frames_per_batch,
            total_frames=-1,
            device=device,
            storing_device='cpu',
            split_trajs=False,
            exploration_type=ExplorationType.RANDOM,
            init_random_frames=-1,
            num_workers=2,
        )
        print("\n✓ Collector created successfully!")
        
        print("\nAttempting to collect data...")
        iterator = iter(collector)
        batch_td = next(iterator)
        print(f"✓ Successfully collected batch with shape: {batch_td.batch_size}")
        
        collector.shutdown()
        print("✓ Test passed!")
        
    except Exception as e:
        print(f"\n✗ ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
