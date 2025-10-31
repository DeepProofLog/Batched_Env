#!/usr/bin/env python3
"""
Test script to verify the TorchRL batched environment works correctly with countries_s3 dataset.
This tests the custom_dummy_env module for vectorized environment support.
"""

import torch
from tensordict import TensorDict
import sys

print("="*60)
print("TorchRL Batched Environment Test - countries_s3 Dataset")
print("="*60)

# Test 1: Import modules
print("\n" + "="*60)
print("TEST 1: Importing Modules")
print("="*60)
try:
    from env import LogicEnv_gym
    from dataset import DataHandler
    from index_manager import IndexManager
    from custom_dummy_env import create_environments, CustomBatchedEnv
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)


# Test 2: Load countries_s3 dataset
print("\n" + "="*60)
print("TEST 2: Loading countries_s3 Dataset")
print("="*60)
try:
    data_handler = DataHandler(
        dataset_name='countries_s3',
        base_path='./data',
        janus_file='countries_s3.pl',
        train_file='train.txt',
        valid_file='valid.txt',
        test_file='test.txt',
        rules_file='rules.txt',
        facts_file='countries_s3.pl'
    )
    
    index_manager = IndexManager(
        data_handler.constants,
        data_handler.predicates,
        max_total_vars=100,
        rules=data_handler.rules,
        max_arity=data_handler.max_arity,
        device="cpu",
        padding_atoms=10,
    )
    
    # Build fact index for efficient unification (done once, shared across all envs)
    index_manager.build_fact_index(data_handler.facts)
    
    print("✓ DataHandler and IndexManager created successfully")
    print(f"  - Dataset: {data_handler.dataset_name}")
    print(f"  - Train queries: {len(data_handler.train_queries)}")
    print(f"  - Valid queries: {len(data_handler.valid_queries)}")
    
except Exception as e:
    print(f"✗ DataHandler/IndexManager creation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)


# Test 3: Create mock args for environment creation
print("\n" + "="*60)
print("TEST 3: Creating Mock Arguments")
print("="*60)

class MockArgs:
    """Mock arguments object for testing."""
    def __init__(self):
        self.dataset_name = 'countries_s3'
        self.data_path = './data'
        self.n_envs = 4  # 4 parallel training environments
        self.n_eval_envs = 2  # 2 parallel eval environments
        self.seed_run_i = 42
        self.corruption_mode = None
        self.corruption_scheme = []
        self.train_neg_ratio = 1
        self.max_depth = 20
        self.memory_pruning = True
        self.endt_action = False
        self.endf_action = True
        self.skip_unary_actions = True
        self.padding_atoms = 10
        self.padding_states = 20
        self.engine = 'python'
        self.kge_action = False
        self.reward_type = 4
        self.pbrs_beta = 0.0
        self.pbrs_gamma = None
        self.gamma = 0.99

args = MockArgs()
print(f"✓ Mock args created")
print(f"  - Training envs: {args.n_envs}")
print(f"  - Eval envs: {args.n_eval_envs}")


# Test 4: Create batched environments
print("\n" + "="*60)
print("TEST 4: Creating Batched Environments")
print("="*60)
try:
    train_env, eval_env, callback_env = create_environments(
        args=args,
        data_handler=data_handler,
        index_manager=index_manager,
        kge_engine=None,
        detailed_eval_env=False
    )
    
    print("✓ Batched environments created successfully")
    print(f"  - Train env type: {type(train_env).__name__}")
    print(f"  - Train env batch size: {train_env.batch_size}")
    print(f"  - Eval env type: {type(eval_env).__name__}")
    print(f"  - Eval env batch size: {eval_env.batch_size}")
    print(f"  - Callback env type: {type(callback_env).__name__}")
    
except Exception as e:
    print(f"✗ Batched environment creation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)


# Test 5: Reset batched training environment
print("\n" + "="*60)
print("TEST 5: Resetting Batched Training Environment")
print("="*60)
try:
    td = train_env.reset()
    print("✓ Reset successful")
    
    # Check TensorDict properties
    assert isinstance(td, TensorDict), f"Expected TensorDict, got {type(td)}"
    print("✓ Returns TensorDict")
    
    # Check batch dimension
    print(f"\n  Batched observation details:")
    print(f"    Batch size: {td.batch_size}")
    print(f"    sub_index shape: {td['sub_index'].shape}")
    print(f"    derived_sub_indices shape: {td['derived_sub_indices'].shape}")
    print(f"    action_mask shape: {td['action_mask'].shape}")
    print(f"    done shape: {td['done'].shape}")
    
    # Verify batch size matches
    assert td.batch_size == torch.Size([args.n_envs]), f"Batch size mismatch: {td.batch_size} vs {args.n_envs}"
    print("✓ Correct batch size")
    
    # Check if we have valid actions in each environment
    for i in range(args.n_envs):
        valid_actions = torch.where(td['action_mask'][i])[0]
        print(f"  Env {i}: {len(valid_actions)} valid actions")
    
except Exception as e:
    print(f"✗ Reset failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)


# Test 6: Take batched steps
print("\n" + "="*60)
print("TEST 6: Taking Batched Steps")
print("="*60)
try:
    # Select actions for each environment
    actions = []
    for i in range(args.n_envs):
        valid_actions = torch.where(td['action_mask'][i])[0]
        if len(valid_actions) > 0:
            actions.append(valid_actions[0].item())
        else:
            actions.append(0)  # Fallback
    
    actions_tensor = torch.tensor(actions)
    print(f"  Actions selected: {actions_tensor.tolist()}")
    
    # Create action TensorDict
    action_td = TensorDict({"action": actions_tensor}, batch_size=[args.n_envs])
    
    # Step
    result_td = train_env.step(action_td)
    print("✓ Batched step successful")
    
    # Check results - TorchRL environments nest results under "next" key
    if "next" in result_td.keys():
        next_td = result_td["next"]
    else:
        next_td = result_td
    
    print(f"\n  Batched step results:")
    print(f"    Rewards shape: {next_td['reward'].shape}")
    print(f"    Rewards: {next_td['reward'].squeeze().tolist()}")
    print(f"    Done: {next_td['done'].squeeze().tolist()}")
    
except Exception as e:
    print(f"✗ Batched step failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)


# Test 7: Run batched episode
print("\n" + "="*60)
print("TEST 7: Running Batched Episode")
print("="*60)
try:
    td = train_env.reset()
    steps = 0
    max_steps = 50
    total_rewards = torch.zeros(args.n_envs)
    
    print(f"  Running batched episode (max {max_steps} steps)...")
    
    while steps < max_steps and not td['done'].all():
        # Select actions for each environment
        actions = []
        for i in range(args.n_envs):
            if not td['done'][i]:
                valid_actions = torch.where(td['action_mask'][i])[0]
                if len(valid_actions) > 0:
                    action_idx = torch.randint(len(valid_actions), (1,)).item()
                    actions.append(valid_actions[action_idx].item())
                else:
                    actions.append(0)
            else:
                actions.append(0)  # Dummy action for done environments
        
        # Create action TensorDict
        action_td = TensorDict({"action": torch.tensor(actions)}, batch_size=[args.n_envs])
        
        # Step
        result_td = train_env.step(action_td)
        
        # Extract next tensordict
        if "next" in result_td.keys():
            td = result_td["next"]
        else:
            td = result_td
        
        total_rewards += td['reward'].squeeze()
        steps += 1
        
        if steps <= 3:  # Print first few steps
            print(f"    Step {steps}: rewards={td['reward'].squeeze().tolist()}")
    
    print(f"\n✓ Batched episode completed in {steps} steps")
    print(f"  Total rewards per env: {total_rewards.tolist()}")
    print(f"  Average reward: {total_rewards.mean().item():.4f}")
    print(f"  Done states: {td['done'].squeeze().tolist()}")
    
except Exception as e:
    print(f"✗ Batched episode failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)


# Test 8: Test eval environment
print("\n" + "="*60)
print("TEST 8: Testing Eval Environment")
print("="*60)
try:
    td = eval_env.reset()
    print(f"✓ Eval environment reset successful")
    print(f"  Batch size: {td.batch_size}")
    print(f"  Done: {td['done'].squeeze().tolist()}")
    
    # Take one step
    actions = []
    for i in range(args.n_eval_envs):
        valid_actions = torch.where(td['action_mask'][i])[0]
        if len(valid_actions) > 0:
            actions.append(valid_actions[0].item())
        else:
            actions.append(0)
    
    action_td = TensorDict({"action": torch.tensor(actions)}, batch_size=[args.n_eval_envs])
    result_td = eval_env.step(action_td)
    
    # Extract next tensordict
    if "next" in result_td.keys():
        next_td = result_td["next"]
    else:
        next_td = result_td
    
    print(f"✓ Eval environment step successful")
    print(f"  Rewards: {next_td['reward'].squeeze().tolist()}")
    
except Exception as e:
    print(f"✗ Eval environment failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)


# Summary
print("\n" + "="*60)
print("TEST SUMMARY - Batched Environments (countries_s3)")
print("="*60)
print("✓ ALL TESTS PASSED")
print("\nThe TorchRL batched environments are working correctly!")
print("Key achievements:")
print("  • Successfully created batched training environments")
print("  • Successfully created batched eval environments")
print("  • Verified correct batch dimensions")
print("  • Successfully reset batched environments")
print("  • Successfully took batched steps")
print("  • Completed batched episodes")
print("  • Verified eval environment functionality")
print("\n✓ TorchRL migration verified for batched environments with countries_s3!")

exit(0)
