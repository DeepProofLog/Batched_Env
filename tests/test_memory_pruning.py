"""
Test memory pruning functionality to ensure states are properly added to Bloom filter.
"""
import torch
from env import BatchedEnv

def test_memory_pruning_basic():
    """Test that memory pruning actually prunes visited states."""
    
    # Create a simple environment with memory pruning enabled
    env = BatchedEnv(
        data_path='data/family',
        batch_size=2,
        mode='train',
        max_depth=5,
        padding_atoms=10,
        padding_states=20,
        memory_pruning=True,
        memory_bits_pow=16,
        memory_hashes=3,
        max_skip_unary_iters=10,  # Use vectorized path
        verbose=2
    )
    
    print("\n=== Test 1: Basic Memory Pruning ===")
    print(f"Memory pruning enabled: {env.memory_pruning}")
    
    # Reset and get initial state
    obs = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Current queries shape: {env.current_queries.shape}")
    
    # Check that Bloom filter was initialized
    print(f"Bloom filter shape: {env.bloom_filter._mem_bloom.shape}")
    initial_bloom_sum = env.bloom_filter._mem_bloom.sum().item()
    print(f"Initial Bloom filter sum (should be non-zero after reset): {initial_bloom_sum}")
    
    # Take a step
    action = torch.zeros((2, 2), dtype=torch.long)  # dummy action
    obs, reward, done, info = env.step(action)
    
    print(f"\nAfter step 1:")
    print(f"Derived states count: {env.derived_states_counts}")
    bloom_sum_after_step = env.bloom_filter._mem_bloom.sum().item()
    print(f"Bloom filter sum after step 1: {bloom_sum_after_step}")
    print(f"Bloom filter changed: {bloom_sum_after_step != initial_bloom_sum}")
    
    # The Bloom filter should have more bits set after processing states
    if bloom_sum_after_step == initial_bloom_sum:
        print("\n⚠️  WARNING: Bloom filter did not change after step!")
        print("This suggests states are not being added to the Bloom filter.")
        return False
    else:
        print("\n✓ Bloom filter updated correctly")
    
    # Take another step to check if pruning actually happens
    action = torch.zeros((2, 2), dtype=torch.long)
    obs, reward, done, info = env.step(action)
    
    print(f"\nAfter step 2:")
    print(f"Derived states count: {env.derived_states_counts}")
    
    return True


def test_memory_pruning_with_logging():
    """Test memory pruning with verbose logging to see if states are pruned."""
    
    print("\n\n=== Test 2: Memory Pruning with Verbose Logging ===")
    
    env = BatchedEnv(
        data_path='data/family',
        batch_size=1,
        mode='train',
        max_depth=5,
        padding_atoms=10,
        padding_states=20,
        memory_pruning=True,
        memory_bits_pow=16,
        memory_hashes=3,
        max_skip_unary_iters=10,
        verbose=2  # Enable verbose logging to see pruning messages
    )
    
    print(f"Memory pruning enabled: {env.memory_pruning}")
    
    # Reset
    obs = env.reset()
    initial_bloom = env.bloom_filter._mem_bloom.clone()
    
    # Take multiple steps to generate states that might be revisited
    for step_num in range(5):
        print(f"\n--- Step {step_num + 1} ---")
        action = torch.zeros((1, 2), dtype=torch.long)
        obs, reward, done, info = env.step(action)
        
        current_bloom = env.bloom_filter._mem_bloom.clone()
        bits_set = current_bloom.sum().item()
        print(f"Bloom filter bits set: {bits_set}")
        
        if step_num > 0:
            changed = (current_bloom != initial_bloom).any().item()
            print(f"Bloom filter changed from initial: {changed}")


def test_bloom_filter_insertion_directly():
    """Test Bloom filter insertion methods directly."""
    
    print("\n\n=== Test 3: Direct Bloom Filter Testing ===")
    
    env = BatchedEnv(
        data_path='data/family',
        batch_size=2,
        mode='train',
        max_depth=5,
        padding_atoms=10,
        padding_states=20,
        memory_pruning=True,
        memory_bits_pow=16,
        memory_hashes=3,
        verbose=1
    )
    
    # Reset to initialize
    env.reset()
    
    # Get a sample state from current queries
    sample_state = env.current_queries[0:1].clone()  # [1, M, D]
    print(f"Sample state shape: {sample_state.shape}")
    
    # Check initial membership (should be True since reset adds initial state)
    sample_state_4d = sample_state.unsqueeze(1)  # [1, 1, M, D]
    owners = torch.tensor([0], dtype=torch.long, device=env._device)
    initial_membership = env.bloom_filter.membership(sample_state_4d, owners)
    print(f"Initial state in Bloom filter: {initial_membership[0, 0].item()}")
    
    # Create a new artificial state
    new_state = sample_state.clone()
    new_state[0, 0, 0] += 1  # Modify it slightly
    
    # Check membership (should be False)
    new_state_4d = new_state.unsqueeze(1)  # [1, 1, M, D]
    before_insert = env.bloom_filter.membership(new_state_4d, owners)
    print(f"New state in Bloom filter before insert: {before_insert[0, 0].item()}")
    
    # Manually add to Bloom filter
    env.current_queries[0] = new_state[0]
    bloom_before = env.bloom_filter._mem_bloom.sum().item()
    env.bloom_filter.add_current(torch.tensor([0], dtype=torch.long, device=env._device), env.current_queries)
    bloom_after = env.bloom_filter._mem_bloom.sum().item()
    
    print(f"Bloom filter bits before insert: {bloom_before}")
    print(f"Bloom filter bits after insert: {bloom_after}")
    print(f"Bloom filter changed: {bloom_after != bloom_before}")
    
    # Check membership again (should be True now)
    after_insert = env.bloom_filter.membership(new_state_4d, owners)
    print(f"New state in Bloom filter after insert: {after_insert[0, 0].item()}")
    
    if after_insert[0, 0].item():
        print("\n✓ Bloom filter insertion works correctly")
        return True
    else:
        print("\n⚠️  Bloom filter insertion failed!")
        return False


if __name__ == '__main__':
    print("Testing Memory Pruning Functionality")
    print("=" * 60)
    
    test3_pass = test_bloom_filter_insertion_directly()
    test1_pass = test_memory_pruning_basic()
    test_memory_pruning_with_logging()
    
    print("\n\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Direct Bloom filter test: {'✓ PASS' if test3_pass else '✗ FAIL'}")
    print(f"Basic memory pruning test: {'✓ PASS' if test1_pass else '✗ FAIL'}")
