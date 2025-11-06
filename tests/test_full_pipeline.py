"""
Full pipeline test that mimics train.py with a vectorized batched environment.

This test exercises the complete training pipeline with a single BatchedVecEnv
(batch_size > 1) to test vectorized operations.
"""
import sys
import os
import torch
from types import SimpleNamespace

# Ensure repository root is on sys.path so local imports resolve when running from tests/ directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset import DataHandler
from index_manager import IndexManager
from neg_sampling import get_sampler, share_sampler_storage
from embeddings import get_embedder
from ppo.ppo_model import create_torchrl_modules
from batched_env import BatchedVecEnv
from ppo.ppo_agent import PPOAgent
from ppo.ppo_rollout_custom import CustomRolloutCollector


def test_vectorized_batched_pipeline():
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
        dataset_name="wn18rr",
        data_path="data",
        janus_file="wn18rr.pl",
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="wn18rr.pl",
        n_train_queries=20,
        n_eval_queries=5,
        n_test_queries=5,
        corruption_mode=None,
        corruption_scheme=['tail'],
        max_total_vars=50,
        padding_atoms=6,
        padding_states=12,
        max_depth=5,
        memory_pruning=False,
        atom_embedder='transe',
        state_embedder='sum',
        constant_embedding_size=32,
        predicate_embedding_size=32,
        atom_embedding_size=32,
        learn_embeddings=True,
        variable_no=20,
        seed_run_i=42,
        # Training params
        batch_size=4,  # Vectorized batch size (process 4 queries in parallel)
        n_steps=16,    # Steps per rollout
        n_epochs=2,    # PPO epochs
        lr=3e-4,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.01,
        # Engine
        engine='python_tensor',
        endt_action=False,
        endf_action=True,
        skip_unary_actions=True,
        train_neg_ratio=1.0,
        reward_type=4,
    )

    # device = torch.device("cpu")
    # use cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ============================================================
    # 1. Build DataHandler and IndexManager
    # ============================================================
    print("\n[1/7] Loading dataset...")
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
    print(f"  Loaded {len(dh.train_queries)} train queries, {len(dh.valid_queries)} valid queries")

    print("\n[2/7] Building index manager...")
    im = IndexManager(
        constants=dh.constants,
        predicates=dh.predicates,
        max_total_vars=args.max_total_vars,
        rules=dh.rules,
        padding_atoms=args.padding_atoms,
        max_arity=dh.max_arity,
        device=device,
    )
    im.build_fact_index(dh.facts)
    print(f"  Index manager ready: {im.constant_no} constants, {im.predicate_no} predicates")
    print(f"  Debug: runtime_var_start={im.runtime_var_start_index}, runtime_var_end={im.runtime_var_end_index}")
    print(f"  Debug: variable_no={im.variable_no}, template_variable_no={im.template_variable_no}, runtime_variable_no={im.runtime_variable_no}")

    # ============================================================
    # 2. Create sampler and embedder
    # ============================================================
    print("\n[3/7] Creating negative sampler...")
    sampler = get_sampler(
        data_handler=dh,
        index_manager=im,
        corruption_scheme=args.corruption_scheme,
        device=device,
    )
    share_sampler_storage(sampler)
    print("  Sampler ready")

    print("\n[4/7] Creating embedder...")
    embedder_getter = get_embedder(args, dh, im, device)
    embedder = embedder_getter.embedder
    embed_dim = getattr(embedder, 'embed_dim', getattr(embedder, 'embedding_dim', args.atom_embedding_size))
    print(f"  Embedder ready: embed_dim={embed_dim}")

    # ============================================================
    # 3. Create vectorized batched environment
    # ============================================================
    print(f"\n[5/7] Creating vectorized batched environment (batch_size={args.batch_size})...")
    
    # Create a single BatchedVecEnv with batch_size for vectorized processing
    train_env = BatchedVecEnv(
        batch_size=args.batch_size,
        index_manager=im,
        data_handler=dh,
        queries=dh.train_queries,
        labels=[1] * len(dh.train_queries),
        query_depths=[0] * len(dh.train_queries),
        facts=set(dh.facts),
        mode='train',
        seed=args.seed_run_i,
        max_depth=args.max_depth,
        memory_pruning=args.memory_pruning,
        padding_atoms=args.padding_atoms,
        padding_states=args.padding_states,
        verbose=0,
        prover_verbose=0,
        device=device,
        engine=args.engine,
    )
    print(f"  Train env ready: batch_size={train_env.batch_size_int}")
    
    # Create eval env (smaller batch for testing)
    eval_env = BatchedVecEnv(
        batch_size=2,
        index_manager=im,
        data_handler=dh,
        queries=dh.valid_queries,
        labels=[1] * len(dh.valid_queries),
        query_depths=[0] * len(dh.valid_queries),
        facts=set(dh.facts),
        mode='eval',
        seed=args.seed_run_i + 1000,
        max_depth=args.max_depth,
        memory_pruning=False,
        padding_atoms=args.padding_atoms,
        padding_states=args.padding_states,
        verbose=0,
        prover_verbose=0,
        device=device,
        engine=args.engine,
    )
    print(f"  Eval env ready: batch_size={eval_env.batch_size_int}")

    # ============================================================
    # 4. Create TorchRL actor/critic modules
    # ============================================================
    print("\n[6/7] Creating actor-critic modules...")
    actor, critic = create_torchrl_modules(
        embedder=embedder,
        num_actions=args.padding_states,
        embed_dim=args.atom_embedding_size,
        hidden_dim=32,
        num_layers=2,
        dropout_prob=0.0,
        device=device,
        enable_kge_action=False,
        kge_inference_engine=None,
        index_manager=im,
    )
    print(f"  Actor params: {sum(p.numel() for p in actor.parameters()):,}")
    print(f"  Critic params: {sum(p.numel() for p in critic.parameters()):,}")

    # Create optimizer
    params_dict = {id(p): p for p in list(actor.parameters()) + list(critic.parameters())}
    optimizer = torch.optim.Adam(params_dict.values(), lr=args.lr)

    # ============================================================
    # 5. Test environment reset and step
    # ============================================================
    print("\n[7/7] Testing environment operations...")
    
    # Test reset
    obs = train_env.reset()
    print(f"  Reset observation keys: {list(obs.keys())}")
    print(f"  Observation shapes:")
    print(f"    sub_index: {obs['sub_index'].shape}")
    print(f"    derived_sub_indices: {obs['derived_sub_indices'].shape}")
    print(f"    action_mask: {obs['action_mask'].shape}")
    
    # Test actor forward
    print(f"\n  Testing actor forward...")
    print(f"    Debugging derived_sub_indices...")
    derived = obs['derived_sub_indices']
    print(f"    Shape: {derived.shape}")
    print(f"    Min value: {derived.min().item()}, Max value: {derived.max().item()}")
    print(f"    Predicate indices ([:,0]): min={derived[:,:,:,0].min().item()}, max={derived[:,:,:,0].max().item()}")
    print(f"    Constant indices ([:,1:]): min={derived[:,:,:,1:].min().item()}, max={derived[:,:,:,1:].max().item()}")
    print(f"    Constant embedder size: {sum(p.numel() for p in actor.module[0].module.actor_critic_model.feature_extractor.embedder.constant_embedder.parameters())}")
    print(f"    Predicate embedder size: {sum(p.numel() for p in actor.module[0].module.actor_critic_model.feature_extractor.embedder.predicate_embedder.parameters())}")
    
    td_out = actor(obs)
    print(f"\n  Actor output keys: {list(td_out.keys())}")
    print(f"  Action shape: {td_out['action'].shape}")
    print(f"  Log prob shape: {td_out['sample_log_prob'].shape}")
    
    # Test critic forward
    value_out = critic(obs)
    print(f"\n  Critic output keys: {list(value_out.keys())}")
    print(f"  State value shape: {value_out['state_value'].shape}")
    
    # Test environment step
    action = td_out['action']
    next_obs = train_env.step(td_out)
    print(f"\n  Step output keys: {list(next_obs.keys())}")
    if 'next' in next_obs.keys():
        print(f"  Next state keys: {list(next_obs['next'].keys())}")
        print(f"  Reward shape: {next_obs['next']['reward'].shape}")
        print(f"  Done shape: {next_obs['next']['done'].shape}")
    else:
        print(f"  Reward shape: {next_obs['reward'].shape}")
        print(f"  Done shape: {next_obs['done'].shape}")
    
    # ============================================================
    # 6. Test rollout collection
    # ============================================================
    print("\n" + "="*60)
    print("Testing Rollout Collection")
    print("="*60)
    
    rollout_collector = CustomRolloutCollector(
        env=train_env,
        actor=actor,
        n_envs=args.batch_size,
        n_steps=args.n_steps,
        device=device,
        debug=False,
    )
    
    print(f"Collecting {args.n_steps} steps from {args.batch_size} parallel queries...")
    experiences, stats = rollout_collector.collect(critic=critic)
    
    print(f"\nRollout collection complete:")
    print(f"  Collected {len(experiences)} experience steps")
    print(f"  Episodes completed: {len(stats.get('episode_info', []))}")
    
    # Verify experience structure
    if len(experiences) > 0:
        exp = experiences[0]
        print(f"\n  Experience tensordict keys: {list(exp.keys())}")
        print(f"  Experience batch size: {exp.batch_size}")
    
    # ============================================================
    # 7. Test PPO learning step
    # ============================================================
    print("\n" + "="*60)
    print("Testing PPO Learning")
    print("="*60)
    
    if len(experiences) > 0:
        print(f"Running PPO update on {len(experiences)} experiences...")
        
        # Create minimal PPO agent for testing
        ppo_agent = PPOAgent(
            actor=actor,
            critic=critic,
            optimizer=optimizer,
            train_env=train_env,
            eval_env=eval_env,
            sampler=sampler,
            data_handler=dh,
            args=args,
            n_envs=args.batch_size,
            n_steps=args.n_steps,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            gamma=args.gamma,
            gae_lambda=0.95,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            value_coef=0.5,
            max_grad_norm=0.5,
            device=device,
        )
        
        # Run learning step
        metrics = ppo_agent.learn(
            experiences=experiences,
            n_steps=args.n_steps,
            n_envs=args.batch_size,
        )
        
        print("\nPPO update complete:")
        print(f"  Policy loss: {metrics.get('policy_loss', 0.0):.4f}")
        print(f"  Value loss: {metrics.get('value_loss', 0.0):.4f}")
        print(f"  Entropy: {metrics.get('entropy', 0.0):.4f}")
        print(f"  Approx KL: {metrics.get('approx_kl', 0.0):.4f}")
    
    # ============================================================
    # 8. Final assertions
    # ============================================================
    print("\n" + "="*60)
    print("Running Assertions")
    print("="*60)
    
    assert 'action' in td_out.keys(), "Actor should output action"
    assert 'sample_log_prob' in td_out.keys(), "Actor should output log prob"
    assert 'state_value' in value_out.keys(), "Critic should output state value"
    # Reward and done are in next_obs['next'] after step
    assert 'next' in next_obs.keys(), "Step should return next state"
    assert 'reward' in next_obs['next'].keys(), "Next state should contain reward"
    assert 'done' in next_obs['next'].keys(), "Next state should contain done"
    assert len(experiences) == args.n_steps, f"Should collect {args.n_steps} experiences"
    
    print("\n✓ All assertions passed!")
    print("\n" + "="*60)
    print("Vectorized Batched Pipeline Test Complete")
    print("="*60)


if __name__ == '__main__':
    test_vectorized_batched_pipeline()
