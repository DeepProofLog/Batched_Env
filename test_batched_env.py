"""
Test Batched Environment

This test runs the batched environment with wn18rr dataset.

Usage:
    python test_batched_env.py
"""

import torch
import time
import argparse
from typing import Dict, List

from batched_env import BatchedVecEnv
from index_manager import IndexManager
from dataset import DataHandler
from utils import Term


def test_batched_env(
    dataset_name: str = "wn18rr",
    batch_size: int = 32,
    num_rollouts: int = 10,
    max_steps_per_rollout: int = 20,
    verbose: int = 1
):
    """
    Test batched environment.
    
    Args:
        dataset_name: Dataset to use (default: wn18rr)
        batch_size: Number of parallel environments
        num_rollouts: Number of rollouts to perform
        max_steps_per_rollout: Maximum steps per rollout
        verbose: Verbosity level
    """
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"Testing Batched Environment")
    print(f"{'='*80}")
    print(f"Dataset: {dataset_name}")
    print(f"Device: {device}")
    print(f"Batch Size: {batch_size}")
    print(f"Num Rollouts: {num_rollouts}")
    print(f"Max Steps per Rollout: {max_steps_per_rollout}")
    print(f"{'='*80}\n")
    
    # Load dataset
    print("Loading dataset...")
    data_handler = DataHandler(
        dataset_name=dataset_name,
        base_path='data',
        train_file='train.txt',
        valid_file='valid.txt',
        test_file='test.txt',
        rules_file='rules.txt',
        facts_file='train.txt'
    )
    
    # Initialize index manager
    index_manager = IndexManager(
        constants=data_handler.constants,
        predicates=data_handler.predicates,
        max_total_vars=1000,
        rules=data_handler.rules,
        device=device  # Set device for GPU tensors
    )
    
    # Get queries and labels from validation set
    queries = data_handler.valid_queries[:batch_size * num_rollouts]
    labels = [1] * len(queries)  # Assume all queries are positive for testing
    
    # Get query depths from dataset or use default
    query_depths = data_handler.valid_queries_depths[:batch_size * num_rollouts]
    # If depths are None, use max_depth
    query_depths = [d if d is not None else 10 for d in query_depths]
    
    # Get facts
    facts = set(data_handler.facts)  # Convert to set as expected by environment
    
    print(f"Loaded {len(queries)} queries")
    print(f"Number of facts: {len(facts)}")
    print(f"Number of rules: {len(data_handler.rules)}")
    
    # Create batched environment 
    print("\nCreating batched environment")
    env = BatchedVecEnv(
        batch_size=batch_size,
        index_manager=index_manager,
        data_handler=data_handler,
        queries=queries,
        labels=labels,
        query_depths=query_depths,
        facts=facts,
        mode='train',
        max_depth=10,
        memory_pruning=True,
        padding_atoms=10,
        padding_states=20,
        verbose=0,
        prover_verbose=0,
        device=device,
    )
    
    # Statistics
    total_steps = 0
    total_rewards = 0.0
    total_time = 0.0
    completed_rollouts = 0
    all_step_rewards = []  # Track all rewards to check for non-zero values
    
    print("\nStarting rollouts...")
    print("-" * 80)
    
    # Run rollouts
    for rollout_idx in range(num_rollouts):
        rollout_start = time.time()
        
        # Reset environment
        tensordict = env.reset()
        
        rollout_steps = 0
        rollout_reward = 0.0
        rollout_done = False
        
        if verbose > 0:
            print(f"\nRollout {rollout_idx + 1}/{num_rollouts}")
        
        # Run steps until done or max steps
        for step in range(max_steps_per_rollout):
            # Random actions for testing (choose from derived states)
            # Action space is determined by padding_states parameter
            actions = torch.randint(
                0, 
                20,  # padding_states parameter from env creation
                (batch_size,), 
                device=device
            )
            tensordict['action'] = actions
            
            # Step environment
            tensordict = env.step(tensordict)
            
            # Collect statistics
            # TorchRL stores outputs in 'next' tensordict
            rewards = tensordict.get(('next', 'reward'), tensordict.get('reward', torch.zeros(batch_size, 1, device=device)))
            dones = tensordict.get('done', tensordict.get(('next', 'done'), torch.zeros(batch_size, 1, dtype=torch.bool, device=device)))
            
            rollout_steps += 1
            rollout_reward += rewards.sum().item()
            all_step_rewards.extend(rewards.cpu().flatten().tolist())  # Track all rewards
            
            if verbose > 1:
                print(f"  Step {step + 1}: "
                      f"Avg Reward: {rewards.mean().item():.4f}, "
                      f"Done: {dones.sum().item()}/{batch_size}")
            
            # Check if all done
            if dones.all():
                rollout_done = True
                if verbose > 0:
                    print(f"  All environments done at step {step + 1}")
                break
        
        rollout_time = time.time() - rollout_start
        
        # Update statistics
        total_steps += rollout_steps
        total_rewards += rollout_reward
        total_time += rollout_time
        if rollout_done:
            completed_rollouts += 1
        
        if verbose > 0:
            avg_reward = rollout_reward / batch_size
            print(f"  Steps: {rollout_steps}, "
                  f"Avg Reward: {avg_reward:.4f}, "
                  f"Time: {rollout_time:.2f}s, "
                  f"Steps/sec: {rollout_steps/rollout_time:.1f}")
    
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total Rollouts: {num_rollouts}")
    print(f"Completed Rollouts: {completed_rollouts} ({100*completed_rollouts/num_rollouts:.1f}%)")
    print(f"Total Steps: {total_steps}")
    print(f"Average Steps per Rollout: {total_steps/num_rollouts:.1f}")
    print(f"Total Rewards: {total_rewards:.2f}")
    print(f"Average Reward per Rollout: {total_rewards/num_rollouts:.4f}")
    print(f"Average Reward per Step: {total_rewards/total_steps:.4f}" if total_steps > 0 else "N/A")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Average Time per Rollout: {total_time/num_rollouts:.2f}s")
    print(f"Average Steps per Second: {total_steps/total_time:.1f}" if total_time > 0 else "N/A")
    
    # Check for non-zero rewards
    non_zero_rewards = [r for r in all_step_rewards if r != 0]
    print(f"\nReward Analysis:")
    print(f"  Total reward samples: {len(all_step_rewards)}")
    print(f"  Non-zero rewards: {len(non_zero_rewards)}")
    if non_zero_rewards:
        print(f"  Non-zero reward values: {set(non_zero_rewards)}")
        print(f"  Max reward: {max(non_zero_rewards):.4f}")
        print(f"  Min reward: {min(non_zero_rewards):.4f}")
    print("=" * 80)
    
    # GPU memory statistics if available
    if device.type == 'cuda':
        print("\nGPU Memory Statistics:")
        print(f"  Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        print(f"  Reserved: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")
        print(f"  Max Allocated: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")
    
    return {
        'num_rollouts': num_rollouts,
        'completed_rollouts': completed_rollouts,
        'total_steps': total_steps,
        'total_rewards': total_rewards,
        'total_time': total_time,
        'avg_steps_per_rollout': total_steps / num_rollouts,
        'avg_reward_per_rollout': total_rewards / num_rollouts,
        'avg_steps_per_second': total_steps / total_time if total_time > 0 else 0
    }


def main():
    parser = argparse.ArgumentParser(description='Test Batched Environment')
    parser.add_argument('--dataset', type=str, default='wn18rr',
                        help='Dataset to use (default: wn18rr)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--num_rollouts', type=int, default=10,
                        help='Number of rollouts (default: 10)')
    parser.add_argument('--max_steps', type=int, default=20,
                        help='Maximum steps per rollout (default: 20)')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbosity level (default: 1)')
    
    args = parser.parse_args()

    test_batched_env(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_rollouts=args.num_rollouts,
        max_steps_per_rollout=args.max_steps,
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()
