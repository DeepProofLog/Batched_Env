"""
Test script for model_eval_torchrl.py - Pure TorchRL evaluation without SB3 dependencies.

This test verifies that the new TorchRL-native evaluation functions work correctly
with ParallelEnv and single EnvBase instances.
"""

import sys
import torch
import argparse
import numpy as np
from pathlib import Path

# Import dataset and model components
from dataset import DataHandler
from index_manager import IndexManager
from env_factory import create_environments
from neg_sampling import get_sampler
from embeddings import get_embedder

# Import the new pure TorchRL evaluation module
from model_eval import evaluate_policy_torchrl, eval_corruptions_torchrl


def create_data_handler(args):
    """Helper to create DataHandler with proper arguments."""
    return DataHandler(
        dataset_name=args.dataset_name,
        base_path=args.data_path,
        janus_file=None,
        train_file=args.train_file,
        valid_file=args.valid_file,
        test_file=args.test_file,
        rules_file=args.rules_file,
        facts_file=args.facts_file,
        n_train_queries=args.n_train_queries,
        n_eval_queries=args.n_eval_queries,
        n_test_queries=args.n_test_queries,
        corruption_mode=args.corruption_mode,
        train_depth=args.train_depth,
        valid_depth=args.valid_depth,
        test_depth=args.test_depth,
        prob_facts=args.prob_facts,
        topk_facts=args.topk_facts,
        topk_facts_threshold=args.topk_facts_threshold,
    )


def create_index_manager(data_handler, args):
    """Helper to create IndexManager with proper arguments."""
    im = IndexManager(
        constants=data_handler.constants,
        predicates=data_handler.predicates,
        max_total_vars=args.max_total_vars,
        rules=data_handler.rules,
        max_arity=data_handler.max_arity,
        padding_atoms=args.padding_atoms,
        device='cpu',
    )
    # Build fact index
    im.build_fact_index(data_handler.facts)
    
    # Create sampler and attach to data_handler (required by environments)
    data_handler.sampler = get_sampler(
        data_handler=data_handler,
        index_manager=im,
        corruption_scheme=args.corruption_scheme,
        device=torch.device('cpu'),
    )
    
    return im


class DummyActor(torch.nn.Module):
    """Minimal dummy actor for testing evaluation functions."""
    
    def __init__(self, n_actions=20, device='cpu'):
        super().__init__()
        self.n_actions = n_actions
        self._device = device
        # Simple linear layer to produce logits
        self.fc = torch.nn.Linear(1, n_actions)
    
    def forward(self, tensordict):
        """Forward pass - returns random actions with log probs."""
        from tensordict import TensorDict
        
        # Get batch size
        batch_size = tensordict.batch_size
        if len(batch_size) == 0:
            bs = 1
        else:
            bs = batch_size[0]
        
        # Get action mask
        action_mask = tensordict.get("action_mask")
        if action_mask is None:
            # No mask, uniform over all actions
            logits = torch.zeros(bs, self.n_actions, device=self._device)
        else:
            # Apply mask
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            
            # Create logits: valid actions get 0, invalid get -inf
            logits = torch.where(
                action_mask.to(torch.bool),
                torch.zeros_like(action_mask, dtype=torch.float),
                torch.full_like(action_mask, float('-inf'), dtype=torch.float)
            )
        
        # Sample action
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Return TensorDict with action and log prob
        result = tensordict.clone()
        result.set("action", action)
        result.set("sample_log_prob", log_prob)
        
        return result


def test_evaluate_policy():
    """Test evaluate_policy_torchrl with ParallelEnv."""
    print("\n" + "="*70)
    print("TEST 1: evaluate_policy_torchrl with ParallelEnv")
    print("="*70 + "\n")
    
    # Create minimal test configuration
    args = argparse.Namespace(
        dataset_name='countries_s3',
        data_path='./data/',
        rules_file='rules.txt',
        facts_file='train.txt',
        train_file='train_depths.txt',
        valid_file='valid_depths.txt',
        test_file='test_depths.txt',
        n_train_queries=5,
        n_eval_queries=10,
        n_test_queries=5,
        prob_facts=False,
        topk_facts=None,
        topk_facts_threshold=None,
        train_depth=None,
        valid_depth=None,
        test_depth=None,
        seed=[0],
        seed_run_i=0,
        
        # Environment params
        n_envs=2,
        n_eval_envs=2,
        use_parallel_envs=True,
        parallel_env_start_method='fork',
        corruption_mode='dynamic',
        corruption_scheme=['head', 'tail'],
        train_neg_ratio=1,
        max_depth=20,
        memory_pruning=True,
        endt_action=False,
        endf_action=True,
        skip_unary_actions=True,
        reward_type=4,
        padding_atoms=6,
        padding_states=20,
        engine='python',
        kge_action=False,
        verbose_env=0,
        verbose_prover=0,
        pbrs_beta=0.0,
        pbrs_gamma=0.99,
        gamma=0.99,
        
        # Embedding params
        atom_embedder='transe',
        state_embedder='mean',
        atom_embedding_size=64,
        constant_embedding_size=64,
        predicate_embedding_size=64,
        state_embedding_size=64,
        learn_embeddings=True,
        max_total_vars=100,
        
        device='cpu',
    )
    
    try:
        # Load dataset
        print("Loading dataset...")
        data_handler = create_data_handler(args)
        index_manager = create_index_manager(data_handler, args)
        
        # Create environments
        print("Creating environments...")
        device = torch.device(args.device)
        train_env, eval_env, callback_env = create_environments(
            args=args,
            data_handler=data_handler,
            index_manager=index_manager,
            kge_engine=None,
            device=device,
        )
        
        # Create actor
        print("Creating actor...")
        actor = DummyActor(n_actions=args.padding_states, device=args.device)
        
        # Test evaluation
        print(f"\nRunning evaluation on {args.n_eval_queries} queries...")
        print(f"Eval env type: {type(eval_env).__name__}")
        if hasattr(eval_env, 'num_workers'):
            print(f"Number of parallel workers: {eval_env.num_workers}")
        
        rewards, lengths, logps, mask, proof_successful = evaluate_policy_torchrl(
            actor=actor,
            env=eval_env,
            n_eval_episodes=10,
            deterministic=True,
            verbose=1,
            track_logprobs=False,
        )
        
        print("\nResults:")
        print(f"  Rewards shape: {rewards.shape}")
        print(f"  Lengths shape: {lengths.shape}")
        print(f"  Logps shape: {logps.shape}")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Proof successful shape: {proof_successful.shape}")
        print(f"  Mean reward: {rewards[mask].mean():.4f}")
        print(f"  Mean length: {lengths[mask].mean():.2f}")
        print(f"  Success rate: {proof_successful[mask].mean():.2%}")
        
        # Cleanup
        try:
            if hasattr(eval_env, 'close'):
                eval_env.close()
        except RuntimeError:
            pass  # Already closed
        try:
            if hasattr(train_env, 'close'):
                train_env.close()
        except RuntimeError:
            pass  # Already closed
        try:
            if hasattr(callback_env, 'close'):
                callback_env.close()
        except RuntimeError:
            pass  # Already closed
        
        print("\n✓ Test PASSED: evaluate_policy_torchrl works!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_eval_corruptions():
    """Test eval_corruptions_torchrl with corruption-based ranking."""
    print("\n" + "="*70)
    print("TEST 2: eval_corruptions_torchrl with RL-only mode")
    print("="*70 + "\n")
    
    # Create minimal test configuration
    args = argparse.Namespace(
        dataset_name='countries_s3',
        data_path='./data/',
        rules_file='rules.txt',
        facts_file='train.txt',
        train_file='train_depths.txt',
        valid_file='valid_depths.txt',
        test_file='test_depths.txt',
        n_train_queries=5,
        n_eval_queries=5,  # Small number for quick test
        n_test_queries=5,
        prob_facts=False,
        topk_facts=None,
        topk_facts_threshold=None,
        train_depth=None,
        valid_depth=None,
        test_depth=None,
        seed=[0],
        seed_run_i=0,
        
        # Environment params
        n_envs=2,
        n_eval_envs=2,
        use_parallel_envs=True,
        parallel_env_start_method='fork',
        corruption_mode='dynamic',
        corruption_scheme=['head', 'tail'],
        train_neg_ratio=1,
        max_depth=20,
        memory_pruning=True,
        endt_action=False,
        endf_action=True,
        skip_unary_actions=True,
        reward_type=4,
        padding_atoms=6,
        padding_states=20,
        engine='python',
        kge_action=False,
        verbose_env=0,
        verbose_prover=0,
        pbrs_beta=0.0,
        pbrs_gamma=0.99,
        gamma=0.99,
        
        # Embedding params
        atom_embedder='transe',
        state_embedder='mean',
        atom_embedding_size=64,
        constant_embedding_size=64,
        predicate_embedding_size=64,
        state_embedding_size=64,
        learn_embeddings=True,
        max_total_vars=100,
        
        device='cpu',
    )
    
    try:
        # Load dataset
        print("Loading dataset...")
        data_handler = create_data_handler(args)
        index_manager = create_index_manager(data_handler, args)
        
        # Create environments
        print("Creating environments...")
        device = torch.device(args.device)
        train_env, eval_env, callback_env = create_environments(
            args=args,
            data_handler=data_handler,
            index_manager=index_manager,
            kge_engine=None,
            device=device,
        )
        
        # Create actor
        print("Creating actor...")
        actor = DummyActor(n_actions=args.padding_states, device=args.device)
        
        # Create negative sampler
        print("Creating negative sampler...")
        sampler = get_sampler(
            data_handler=data_handler,
            index_manager=index_manager,
            corruption_scheme=args.corruption_scheme,
            device=device,
        )
        
        # Test corruption evaluation
        print(f"\nRunning corruption evaluation on {args.n_eval_queries} queries...")
        print(f"Eval env type: {type(eval_env).__name__}")
        
        results = eval_corruptions_torchrl(
            actor=actor,
            env=eval_env,
            data=data_handler.valid_queries[:args.n_eval_queries],
            sampler=sampler,
            n_corruptions=3,  # Small number for quick test
            deterministic=True,
            verbose=1,
            plot=False,
            kge_inference_engine=None,
            evaluation_mode='rl_only',
            corruption_scheme=['head', 'tail'],
            data_depths=data_handler.valid_queries_depths[:args.n_eval_queries] if data_handler.valid_queries_depths else None,
        )
        
        print("\nResults:")
        print(f"  MRR: {results.get('mrr_mean', 0.0):.4f}")
        print(f"  Hits@1: {results.get('h1_mean', 0.0):.4f}")
        print(f"  Hits@3: {results.get('h3_mean', 0.0):.4f}")
        print(f"  Hits@10: {results.get('h10_mean', 0.0):.4f}")
        print(f"  Accuracy: {results.get('accuracy', 0.0):.4f}")
        print(f"  Precision: {results.get('precision', 0.0):.4f}")
        print(f"  Recall: {results.get('recall', 0.0):.4f}")
        print(f"  F1: {results.get('f1', 0.0):.4f}")
        
        # Cleanup
        try:
            if hasattr(eval_env, 'close'):
                eval_env.close()
        except RuntimeError:
            pass  # Already closed
        try:
            if hasattr(train_env, 'close'):
                train_env.close()
        except RuntimeError:
            pass  # Already closed
        try:
            if hasattr(callback_env, 'close'):
                callback_env.close()
        except RuntimeError:
            pass  # Already closed
        
        print("\n✓ Test PASSED: eval_corruptions_torchrl works!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_env():
    """Test with single (non-parallel) environment."""
    print("\n" + "="*70)
    print("TEST 3: evaluate_policy_torchrl with single EnvBase")
    print("="*70 + "\n")
    
    # Create minimal test configuration
    args = argparse.Namespace(
        dataset_name='countries_s3',
        data_path='./data/',
        rules_file='rules.txt',
        facts_file='train.txt',
        train_file='train_depths.txt',
        valid_file='valid_depths.txt',
        test_file='test_depths.txt',
        n_train_queries=5,
        n_eval_queries=5,
        n_test_queries=5,
        prob_facts=False,
        topk_facts=None,
        topk_facts_threshold=None,
        train_depth=None,
        valid_depth=None,
        test_depth=None,
        seed=[0],
        seed_run_i=0,
        
        # Environment params - single env
        n_envs=1,
        n_eval_envs=1,
        use_parallel_envs=False,  # Disable parallel envs
        corruption_mode='dynamic',
        corruption_scheme=['head', 'tail'],
        train_neg_ratio=1,
        max_depth=20,
        memory_pruning=True,
        endt_action=False,
        endf_action=True,
        skip_unary_actions=True,
        reward_type=4,
        padding_atoms=6,
        padding_states=20,
        engine='python',
        kge_action=False,
        verbose_env=0,
        verbose_prover=0,
        pbrs_beta=0.0,
        pbrs_gamma=0.99,
        gamma=0.99,
        
        # Embedding params
        atom_embedder='transe',
        state_embedder='mean',
        atom_embedding_size=64,
        constant_embedding_size=64,
        predicate_embedding_size=64,
        state_embedding_size=64,
        learn_embeddings=True,
        max_total_vars=100,
        
        device='cpu',
    )
    
    try:
        # Load dataset
        print("Loading dataset...")
        data_handler = create_data_handler(args)
        index_manager = create_index_manager(data_handler, args)
        
        # Create environments
        print("Creating single environment...")
        device = torch.device(args.device)
        train_env, eval_env, callback_env = create_environments(
            args=args,
            data_handler=data_handler,
            index_manager=index_manager,
            kge_engine=None,
            device=device,
        )
        
        # Create actor
        print("Creating actor...")
        actor = DummyActor(n_actions=args.padding_states, device=args.device)
        
        # Test evaluation
        print(f"\nRunning evaluation on single environment...")
        print(f"Eval env type: {type(eval_env).__name__}")
        
        rewards, lengths, logps, mask, proof_successful = evaluate_policy_torchrl(
            actor=actor,
            env=eval_env,
            n_eval_episodes=5,
            deterministic=True,
            verbose=1,
            track_logprobs=False,
        )
        
        print("\nResults:")
        print(f"  Rewards shape: {rewards.shape}")
        print(f"  Lengths shape: {lengths.shape}")
        print(f"  Mean reward: {rewards[mask].mean():.4f}")
        print(f"  Mean length: {lengths[mask].mean():.2f}")
        print(f"  Success rate: {proof_successful[mask].mean():.2%}")
        
        # Cleanup
        try:
            if hasattr(eval_env, 'close'):
                eval_env.close()
        except RuntimeError:
            pass  # Already closed
        try:
            if hasattr(train_env, 'close'):
                train_env.close()
        except RuntimeError:
            pass  # Already closed
        try:
            if hasattr(callback_env, 'close'):
                callback_env.close()
        except RuntimeError:
            pass  # Already closed
        
        print("\n✓ Test PASSED: Single env evaluation works!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("Testing model_eval_torchrl.py - Pure TorchRL Evaluation")
    print("="*70)
    
    # Check if data exists
    if not Path("./data/countries_s3").exists():
        print("\nERROR: Dataset 'countries_s3' not found in ./data/")
        print("Please ensure the dataset is available before running this test.")
        sys.exit(1)
    
    results = []
    
    # Run tests
    results.append(("Parallel Env - evaluate_policy", test_evaluate_policy()))
    results.append(("Parallel Env - eval_corruptions", test_eval_corruptions()))
    results.append(("Single Env - evaluate_policy", test_single_env()))
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
    
    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n✓ All tests PASSED!")
        sys.exit(0)
    else:
        print("\n✗ Some tests FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
