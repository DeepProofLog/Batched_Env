#!/usr/bin/env python3
"""
Test script to verify the TorchRL environment works correctly with countries_s3 dataset.
This test ensures proper loading and interaction with the countries_s3 dataset.
"""

import torch
from tensordict import TensorDict
import sys

print("="*60)
print("TorchRL Environment Test - countries_s3 Dataset")
print("="*60)

# Test 1: Import modules
print("\n" + "="*60)
print("TEST 1: Importing Modules")
print("="*60)
try:
    from env import LogicEnv_gym
    from dataset import DataHandler
    from index_manager import IndexManager
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
    
    print("✓ DataHandler created successfully")
    print(f"  - Dataset: {data_handler.dataset_name}")
    print(f"  - Constants: {len(data_handler.constants)}")
    print(f"  - Predicates: {len(data_handler.predicates)}")
    print(f"  - Facts: {len(data_handler.facts)}")
    print(f"  - Train queries: {len(data_handler.train_queries)}")
    print(f"  - Valid queries: {len(data_handler.valid_queries)}")
    print(f"  - Test queries: {len(data_handler.test_queries)}")
    print(f"  - Rules: {len(data_handler.rules)}")
    print(f"  - Max arity: {data_handler.max_arity}")
    
except Exception as e:
    print(f"✗ DataHandler creation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)


# Test 3: Create IndexManager
print("\n" + "="*60)
print("TEST 3: Creating IndexManager")
print("="*60)
try:
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
    
    print("✓ IndexManager created successfully")
    print(f"  - Total constants: {len(data_handler.constants)}")
    print(f"  - Total predicates: {len(data_handler.predicates)}")
    print(f"  - Rules indexed: {len(data_handler.rules)}")
    
except Exception as e:
    print(f"✗ IndexManager creation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)


# Test 4: Create TorchRL environment
print("\n" + "="*60)
print("TEST 4: Creating TorchRL Environment")
print("="*60)
try:
    env = LogicEnv_gym(
        index_manager=index_manager,
        data_handler=data_handler,
        queries=data_handler.train_queries[:10],  # Use first 10 queries for testing
        labels=[1] * 10,
        query_depths=data_handler.train_queries_depths[:10] if data_handler.train_queries_depths else None,
        facts=set(data_handler.facts),
        mode='train',
        device=torch.device('cpu'),
        seed=42,
        max_depth=20,
        padding_atoms=10,
        padding_states=20,
        verbose=0,
        prover_verbose=0,
    )
    
    print("✓ Environment created successfully")
    
    # Check specs
    assert hasattr(env, 'observation_spec'), "Missing observation_spec"
    assert hasattr(env, 'action_spec'), "Missing action_spec"
    assert hasattr(env, 'reward_spec'), "Missing reward_spec"
    assert hasattr(env, 'done_spec'), "Missing done_spec"
    print("✓ All TorchRL specs defined")
    
    # Print spec information
    print(f"\n  Observation spec keys: {list(env.observation_spec.keys())}")
    print(f"  Action spec: {env.action_spec}")
    print(f"  Reward spec: {env.reward_spec}")
    print(f"  Done spec: {env.done_spec}")
    
except Exception as e:
    print(f"✗ Environment creation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)


# Test 5: Reset environment
print("\n" + "="*60)
print("TEST 5: Environment Reset")
print("="*60)
try:
    td = env.reset()
    print("✓ Reset successful")
    
    # Check TensorDict properties
    assert isinstance(td, TensorDict), f"Expected TensorDict, got {type(td)}"
    print("✓ Returns TensorDict")
    
    # Check required keys
    required_keys = ['sub_index', 'derived_sub_indices', 'action_mask', 'done']
    for key in required_keys:
        assert key in td.keys(), f"Missing key: {key}"
    print(f"✓ Has required keys: {required_keys}")
    
    # Print shapes
    print(f"\n  Observation details:")
    print(f"    sub_index shape: {td['sub_index'].shape}")
    print(f"    derived_sub_indices shape: {td['derived_sub_indices'].shape}")
    print(f"    action_mask shape: {td['action_mask'].shape}")
    print(f"    done: {td['done'].item()}")
    
    # Check dtypes
    assert td['sub_index'].dtype == torch.int32, "Wrong dtype for sub_index"
    assert td['action_mask'].dtype == torch.bool, "Wrong dtype for action_mask"
    assert td['done'].dtype == torch.bool, "Wrong dtype for done"
    print("✓ Correct data types")
    
    # Check if we have valid actions
    valid_actions = torch.where(td['action_mask'])[0]
    print(f"\n  Available actions: {len(valid_actions)}")
    if len(valid_actions) > 0:
        print(f"  First few action indices: {valid_actions[:5].tolist()}")
    
except Exception as e:
    print(f"✗ Reset failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)


# Test 6: Take a step
print("\n" + "="*60)
print("TEST 6: Environment Step")
print("="*60)
try:
    # Find valid action from action mask
    valid_actions = torch.where(td['action_mask'])[0]
    if len(valid_actions) == 0:
        print("⚠ No valid actions available - skipping step test")
    else:
        action = valid_actions[0].item()
        print(f"  Taking action: {action}")
        
        # Create action TensorDict
        action_td = TensorDict({"action": torch.tensor(action)}, batch_size=[])
        
        # Step
        next_td = env._step(action_td)
        print("✓ Step successful")
        
        # Check return type
        assert isinstance(next_td, TensorDict), f"Expected TensorDict, got {type(next_td)}"
        print("✓ Returns TensorDict")
        
        # Check required keys
        required_keys = ['sub_index', 'derived_sub_indices', 'action_mask', 'reward', 'done']
        for key in required_keys:
            assert key in next_td.keys(), f"Missing key: {key}"
        print(f"✓ Has required keys: {required_keys}")
        
        # Check reward and done
        print(f"\n  Step results:")
        print(f"    Reward: {next_td['reward'].item()}")
        print(f"    Done: {next_td['done'].item()}")
        print(f"    New action mask sum: {next_td['action_mask'].sum().item()}")
        
        assert next_td['reward'].shape == torch.Size([1]), "Reward should have shape [1]"
        print("✓ Correct reward shape")
        
except Exception as e:
    print(f"✗ Step failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)


# Test 7: Run a full episode
print("\n" + "="*60)
print("TEST 7: Full Episode")
print("="*60)
try:
    td = env.reset()
    steps = 0
    max_steps = 100
    total_reward = 0.0
    action_history = []
    
    print(f"  Running episode (max {max_steps} steps)...")
    
    while not td['done'].item() and steps < max_steps:
        # Get valid action
        valid_actions = torch.where(td['action_mask'])[0]
        if len(valid_actions) == 0:
            print(f"  No valid actions at step {steps}")
            break
        
        # Random action selection
        action_idx = torch.randint(len(valid_actions), (1,)).item()
        action = valid_actions[action_idx].item()
        action_history.append(action)
        
        # Step
        action_td = TensorDict({"action": torch.tensor(action)}, batch_size=[])
        td = env._step(action_td)
        
        reward = td['reward'].item()
        total_reward += reward
        steps += 1
        
        if steps <= 5 or reward != 0:  # Print first few steps and reward steps
            print(f"    Step {steps}: action={action}, reward={reward:.4f}, done={td['done'].item()}")
    
    print(f"\n✓ Episode completed")
    print(f"  Total steps: {steps}")
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  Final done: {td['done'].item()}")
    print(f"  Episode ended: {'Naturally' if td['done'].item() else 'Max steps reached'}")
    
    if 'is_success' in td:
        print(f"  Success: {td['is_success'].item()}")
    
except Exception as e:
    print(f"✗ Episode failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)


# Test 8: Multiple episodes
print("\n" + "="*60)
print("TEST 8: Multiple Episodes")
print("="*60)
try:
    num_episodes = 5
    episode_stats = []
    
    print(f"  Running {num_episodes} episodes...")
    
    for ep in range(num_episodes):
        td = env.reset()
        steps = 0
        max_steps = 50
        total_reward = 0.0
        
        while not td['done'].item() and steps < max_steps:
            valid_actions = torch.where(td['action_mask'])[0]
            if len(valid_actions) == 0:
                break
            
            action_idx = torch.randint(len(valid_actions), (1,)).item()
            action = valid_actions[action_idx].item()
            
            action_td = TensorDict({"action": torch.tensor(action)}, batch_size=[])
            td = env._step(action_td)
            
            total_reward += td['reward'].item()
            steps += 1
        
        episode_stats.append({
            'episode': ep + 1,
            'steps': steps,
            'reward': total_reward,
            'done': td['done'].item()
        })
        
        print(f"    Episode {ep+1}: {steps} steps, reward={total_reward:.4f}, done={td['done'].item()}")
    
    print(f"\n✓ Completed {num_episodes} episodes")
    
    avg_steps = sum(s['steps'] for s in episode_stats) / num_episodes
    avg_reward = sum(s['reward'] for s in episode_stats) / num_episodes
    print(f"  Average steps: {avg_steps:.2f}")
    print(f"  Average reward: {avg_reward:.4f}")
    
except Exception as e:
    print(f"✗ Multiple episodes failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)


# Summary
print("\n" + "="*60)
print("TEST SUMMARY - countries_s3 Dataset")
print("="*60)
print("✓ ALL TESTS PASSED")
print("\nThe TorchRL environment is working correctly with countries_s3!")
print("Key achievements:")
print("  • Successfully loaded countries_s3 dataset")
print("  • Created TorchRL-compatible environment")
print("  • Verified all TorchRL specs (observation, action, reward, done)")
print("  • Successfully reset environment")
print("  • Successfully took steps with valid actions")
print("  • Completed full episodes")
print("  • Ran multiple episodes successfully")
print("\n✓ Migration to TorchRL verified for countries_s3 dataset!")

exit(0)
