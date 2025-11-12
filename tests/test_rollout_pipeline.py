"""
Full pipeline test that mimics train.py with a vectorized batched environment.

This test exercises the complete training pipeline with a single BatchedVecEnv
(batch_size > 1) to test vectorized operations.
"""
import sys
import os
import torch
from types import SimpleNamespace
from time import time
# Ensure repository root is on sys.path so local imports resolve when running from tests/ directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_handler import DataHandler
from index_manager import IndexManager
from sampler import Sampler, SamplerConfig
from embeddings import get_embedder
from ppo.ppo_model_torchrl import create_torchrl_modules
from ppo.ppo_model import create_torch_modules
from env import BatchedEnv
from ppo.ppo_agent_torchrl import PPOAgentTorchRL
from ppo.ppo_agent import PPOAgent
from ppo.ppo_rollout import RolloutCollector
from unification_engine import UnificationEngine

USE_TORCHRL = False

def test_vectorized_batched_pipeline(n_tests=1, device='None', use_torchrl=True):
    """
    Test the full pipeline with a vectorized batched environment.
    
    This test mimics train.py but uses a single BatchedVecEnv with batch_size > 1
    to test vectorized operations (no parallel envs, just batched).
    """
    print("\n" + "="*60)
    print("Testing Vectorized Batched Pipeline")
    print("="*60)
    
    # Configuration (small scale for testing)
    args = SimpleNamespace(
        dataset_name="countries_s3",
        max_depth=20,  # Reduce max depth so episodes complete faster
        batch_size_model=8192,  # PPO batch size
        batch_size_env=512,    # Number of queries in the env
        n_steps=64,    # Steps per rollout
        n_epochs=5,    # PPO epochs
        data_path="data",
        janus_file=None,  # Don't use janus file
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="train.txt",  # Use train.txt as facts
        n_train_queries=None,
        n_eval_queries=None,
        n_test_queries=None,
        corruption_mode=True,  # Enable negative sampling
        corruption_scheme=['head', 'tail'],
        train_neg_ratio=3,  # Encourage negatives so rewards cover ±1
        max_total_vars=1000000,
        padding_atoms=6,
        padding_states=20,
        atom_embedder='transe',
        state_embedder='sum',
        constant_embedding_size=256,
        predicate_embedding_size=256,
        atom_embedding_size=256,
        learn_embeddings=True,
        variable_no=100,
        seed_run_i=42,
        # Training params
        lr=3e-4,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.01,
        # Engine
        engine='python_tensor',
        endt_action=False,
        end_proof_action=True,  # include explicit END action to guarantee at least two choices
        skip_unary_actions=False,  # keep unary expansions for broader branching
        memory_pruning=False,  # DISABLED to test if this is the issue
        reward_type=4,  # Symmetric rewards so negatives matter
        verbose_env=0,  # Enable verbose environment logging
        verbose_prover=0,  # Enable prover logging
    )
    args.min_multiaction_ratio = 0.05
    args.corruption_scheme = args.corruption_scheme if args.dataset_name not in ['countries_s3'] else ['tail']
    torch.manual_seed(args.seed_run_i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed_run_i)

    # use cuda if available
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    args.use_amp = isinstance(device, torch.device) and device.type == 'cuda'

    # ============================================================
    # 1. Build DataHandler
    # ============================================================
    start_time = time()
    print("\n[1/10] Loading dataset...")
    dh = DataHandler(
        dataset_name=args.dataset_name,
        base_path=args.data_path,
        janus_file=args.janus_file,
        train_file=args.train_file,
        valid_file=args.valid_file,
        test_file=args.test_file,
        rules_file=args.rules_file,
        facts_file=args.facts_file,
        n_train_queries=args.n_train_queries,
        n_eval_queries=args.n_eval_queries,
        n_test_queries=args.n_test_queries,
    )
    end_time = time()
    print(f"  Step completed in {end_time - start_time:.2f} seconds")
    if n_tests == 1:
        return  # Early exit for quick test of step 1
    # ============================================================
    # 2. Build IndexManager
    # ============================================================
    start_time = time()
    print("\n[2/10] Building index manager...")
    im = IndexManager(
        constants=dh.constants,
        predicates=dh.predicates,
        max_total_runtime_vars=args.max_total_vars,
        padding_atoms=args.padding_atoms,
        max_arity=dh.max_arity,
        device=device,
    )
    dh.materialize_indices(im=im, device=device)
    
    end_time = time()
    print(f"  Step completed in {end_time - start_time:.2f} seconds")
    if n_tests == 2:
        return  # Early exit for quick test of steps 1-2
    # ============================================================
    # 3. Create sampler
    # ============================================================
    start_time = time()
    print("\n[3/10] Creating negative sampler")
    
    # Prepare all known triples for filtering
    all_triples_tensor = dh.all_known_triples_idx.to(device)
    
    sampler = Sampler.from_data(
        all_known_triples_idx=all_triples_tensor,
        num_entities=im.constant_no,
        num_relations=im.predicate_no,
        device=device,
        default_mode=args.corruption_scheme,
        seed=args.seed_run_i,
    )

    # # sample 3 negatives from each positive query (2 positives in batch)
    # train_split_for_sampler = dh.get_materialized_split('train')
    # if train_split_for_sampler.queries.shape[0] >= 2:
    #     positive_queries_tensor = train_split_for_sampler.queries[:2, 0]
    #     negatives = sampler.corrupt(positive_queries_tensor, num_negatives=3)  # [2, 3, 3]

    end_time = time()
    print(f"  Step completed in {end_time - start_time:.2f} seconds")
    if n_tests == 3:
        return  # Early exit for quick test of steps 1-3
    # ============================================================
    # 4. Create embedder
    # ============================================================
    start_time = time()
    print("\n[4/10] Creating embedder...")
    embedder_getter = get_embedder(
        args=args,
        data_handler=dh,
        constant_no=im.constant_no,
        predicate_no=im.predicate_no,
        runtime_var_end_index=im.runtime_var_end_index,
        constant_str2idx=im.constant_str2idx,
        predicate_str2idx=im.predicate_str2idx,
        constant_images_no=getattr(im, 'constant_images_no', 0),
        device=device
    )
    embedder = embedder_getter.embedder
    embed_dim = getattr(embedder, 'embed_dim', getattr(embedder, 'embedding_dim', args.atom_embedding_size))
    end_time = time()
    print(f"  Step completed in {end_time - start_time:.2f} seconds")
    if n_tests == 4:
        return  # Early exit for quick test of steps 1-4
    
    # ============================================================
    # 5. Create vectorized batched environment
    # ============================================================
    print(f"\n[5/10] Creating vectorized batched environment (batch_size={args.batch_size_env})...")
    start_time = time()
    
    # Use pre-materialized dataset splits for the environment
    train_split = dh.get_materialized_split('train')
    valid_split = dh.get_materialized_split('valid')

    unification_engine = UnificationEngine.from_index_manager(im, stringifier_params=None)
    print(f"  UnificationEngine ready")
    
    # Create a single BatchedVecEnv with batch_size for vectorized processing
    train_env = BatchedEnv(
        batch_size=args.batch_size_env,
        unification_engine=unification_engine,
        queries=train_split.queries,
        labels=train_split.labels,
        query_depths=train_split.depths,
        mode='train',
        max_depth=args.max_depth,
        memory_pruning=args.memory_pruning,
        padding_atoms=args.padding_atoms,
        padding_states=args.padding_states,
        reward_type=args.reward_type,
        verbose=0,  # Enable verbose to see derived state counts
        prover_verbose=0,
        device=device,
        corruption_mode=args.corruption_mode,
        sampler=sampler,
        train_neg_ratio=args.train_neg_ratio,
        end_proof_action=args.end_proof_action,
        skip_unary_actions=args.skip_unary_actions,
        end_pred_idx=im.predicate_str2idx.get('End', None) if args.end_proof_action else None,
        true_pred_idx=im.true_pred_idx,
        false_pred_idx=im.false_pred_idx,
        max_arity=im.max_arity,
        padding_idx=im.padding_idx,
    )

    
    end_time = time()
    print(f"  Step completed in {end_time - start_time:.2f} seconds")

    if n_tests == 5:
        return  # Early exit for quick test of steps 1-5    
    # ============================================================
    # 6. Create TorchRL actor/critic modules
    # ============================================================
    print("\n[6/10] Creating actor-critic modules...")
    start_time = time()
    hidden_dim = 256
    num_layers = 4
    if not use_torchrl:
        actor, critic = create_torch_modules(
            embedder=embedder,
            num_actions=args.padding_states,
            embed_dim=args.atom_embedding_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_prob=0.0,
            device=device,
            enable_kge_action=False,
            kge_inference_engine=None,
            index_manager=im,
        )
    else:
        actor, critic = create_torchrl_modules(
            embedder=embedder,
            num_actions=args.padding_states,
            embed_dim=args.atom_embedding_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_prob=0.0,
            device=device,
            enable_kge_action=False,
            kge_inference_engine=None,
            index_manager=im,
        )

    # Create optimizer with fused Adam for better performance on CUDA
    params_dict = {id(p): p for p in list(actor.parameters()) + list(critic.parameters())}
    use_fused = device.type == 'cuda' and torch.cuda.is_available()
    optimizer = torch.optim.AdamW(params_dict.values(), lr=args.lr, fused=use_fused)
    if use_fused:
        print(f"  Using fused AdamW optimizer for faster training")
    end_time = time()
    print(f"  Step completed in {end_time - start_time:.2f} seconds")
    if n_tests == 6:
        return  # Early exit for quick test of steps 1-6
    
 
    # ============================================================
    # 7. Test rollout collection
    # ============================================================
    print("\n" + "="*60)
    print("Testing Rollout Collection")
    print("="*60)
    start_time = time()
    
    rollout_collector = RolloutCollector(
        env=train_env,
        actor=actor,
        n_envs=args.batch_size_env,
        n_steps=args.n_steps,
        device=device,
        debug=False,
        debug_action_space=False,
    )    
    experiences, stats = rollout_collector.collect(critic=critic)

    # Verify experience structure
    if len(experiences) > 0:
        exp = experiences[0]
        # Check rewards
        if 'next' in exp.keys() and 'reward' in exp['next'].keys():
            rewards = torch.stack([experiences[i]['next']['reward'] for i in range(len(experiences))])
            unique_rewards = torch.unique(rewards).tolist()
            assert any(r < 0 for r in unique_rewards), "Rollout should include negative rewards from corrupted queries"
            assert any(r > 0 for r in unique_rewards), "Rollout should include positive rewards from authentic queries"
    end_time = time()
    print(f"  Step completed in {end_time - start_time:.2f} seconds")
    if n_tests == 7:
        return  # Early exit for quick test of steps 1-8
    
    # ============================================================
    # 8. Test PPO learning step
    # ============================================================
    print("\n" + "="*60)
    print("Testing PPO Learning")
    print("="*60)
    start_time = time()

    if len(experiences) > 0:
        if not use_torchrl:
            ppo_agent = PPOAgent(
                actor=actor,
                critic=critic,
                optimizer=optimizer,
                train_env=train_env,
                eval_env=None,
                sampler=sampler,
                data_handler=dh,
                args=args,
                n_envs=args.batch_size_env,
                n_steps=args.n_steps,
                n_epochs=args.n_epochs,
                batch_size=args.batch_size_model,
                gamma=args.gamma,
                gae_lambda=0.95,
                clip_range=args.clip_range,
                ent_coef=args.ent_coef,
                value_coef=0.5,
                max_grad_norm=0.5,
                device=device,
                debug_mode=True,
                min_multiaction_ratio=args.min_multiaction_ratio,
                use_amp=args.use_amp,
            )
        else:
            ppo_agent = PPOAgentTorchRL(
                actor=actor,
                critic=critic,
                optimizer=optimizer,
                train_env=train_env,
                eval_env=None,
                sampler=sampler,
                data_handler=dh,
                args=args,
                n_envs=args.batch_size_env,
                n_steps=args.n_steps,
                n_epochs=args.n_epochs,
                batch_size=args.batch_size_model,
                gamma=args.gamma,
                gae_lambda=0.95,
                clip_range=args.clip_range,
                ent_coef=args.ent_coef,
                value_coef=0.5,
                max_grad_norm=0.5,
                device=device,
                debug_mode=True,
                min_multiaction_ratio=args.min_multiaction_ratio,
                use_amp=args.use_amp,
            )
        
        # Run learning step
        ppo_agent.learn(
            experiences=experiences,
            n_steps=args.n_steps,
            n_envs=args.batch_size_env,
        )

    end_time = time()
    print(f"  Step completed in {end_time - start_time:.2f} seconds")
    if n_tests == 8:
        return  # Early exit for quick test of steps 1-9


if __name__ == '__main__':
    # use cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    test_vectorized_batched_pipeline(n_tests=8, device=device, use_torchrl=USE_TORCHRL)
