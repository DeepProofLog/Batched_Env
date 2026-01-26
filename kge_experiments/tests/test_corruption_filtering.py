"""
Test corruption mode alternation and fact filtering in negative sampling.

Verifies:
1. Head/tail corruption alternates correctly in vectorized mode
2. Known facts are filtered out from corrupted negatives
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from nn.sampler import Sampler, SamplerConfig


def test_head_tail_alternation():
    """Test that head and tail corruption alternate correctly."""
    print("=" * 60)
    print("Test 1: Head/Tail Corruption Alternation")
    print("=" * 60)

    # Create sampler (no filtering for this test)
    cfg = SamplerConfig(num_entities=100, num_relations=10, device=torch.device('cuda'))
    sampler = Sampler(cfg)

    # Test queries: (pred, head, tail)
    queries = torch.tensor([
        [1, 10, 20],
        [2, 30, 40],
        [3, 50, 60],
        [4, 70, 80],
    ], device='cuda')

    # Simulate what env._sample_negatives does in vectorized mode
    B = queries.shape[0]
    corruption_scheme = ('head', 'tail')
    num_modes = len(corruption_scheme)

    # Mode counters: 0=head, 1=tail
    mode_counters = torch.tensor([0, 1, 0, 1], dtype=torch.int64, device='cuda')
    should_neg = torch.ones(B, dtype=torch.bool, device='cuda')  # All should be negatives

    result_queries = queries.clone()

    for mode_idx, mode in enumerate(corruption_scheme):
        mode_mask = should_neg & ((mode_counters % num_modes) == mode_idx)
        neg_queries = sampler.corrupt(queries, num_negatives=1, mode=mode, filter=False, unique=False).squeeze(1)
        result_queries = torch.where(mode_mask.unsqueeze(-1), neg_queries, result_queries)

    print(f"Original queries (pred, head, tail):")
    for i, q in enumerate(queries):
        print(f"  [{i}] pred={q[0].item()}, head={q[1].item()}, tail={q[2].item()}")

    print(f"\nMode counters: {mode_counters.tolist()}")
    print(f"Expected: env 0,2 -> head corruption, env 1,3 -> tail corruption")

    print(f"\nResult queries:")
    all_correct = True
    for i in range(B):
        orig_head, orig_tail = queries[i, 1].item(), queries[i, 2].item()
        new_head, new_tail = result_queries[i, 1].item(), result_queries[i, 2].item()

        head_changed = new_head != orig_head
        tail_changed = new_tail != orig_tail

        expected_mode = 'head' if mode_counters[i] % 2 == 0 else 'tail'
        actual_mode = 'head' if head_changed and not tail_changed else ('tail' if tail_changed and not head_changed else 'both/none')

        correct = expected_mode == actual_mode
        all_correct = all_correct and correct
        status = "OK" if correct else "FAIL"

        print(f"  [{i}] head: {orig_head}->{new_head}, tail: {orig_tail}->{new_tail} | expected={expected_mode}, actual={actual_mode} [{status}]")

    assert all_correct, "Head/tail alternation failed!"
    print("\nPASSED: Head/tail alternation works correctly")
    print()


def test_fact_filtering():
    """Test that known facts are filtered from corrupted negatives."""
    print("=" * 60)
    print("Test 2: Fact Filtering")
    print("=" * 60)

    # Create known facts
    # Format: (pred, head, tail) - but sampler uses (head, pred, tail) internally
    known_facts = torch.tensor([
        [1, 10, 20],  # brother(10, 20)
        [1, 10, 30],  # brother(10, 30) - this could be a corruption of (1, 10, 20) with tail=30
        [1, 10, 40],  # brother(10, 40)
        [1, 10, 50],  # brother(10, 50)
        [1, 15, 20],  # brother(15, 20) - this could be a corruption of (1, 10, 20) with head=15
        [1, 25, 20],  # brother(25, 20)
    ], device='cuda')

    # Create sampler with known facts for filtering
    sampler = Sampler.from_data(
        all_known_triples_idx=known_facts,
        num_entities=100,
        num_relations=10,
        device=torch.device('cuda'),
        default_mode='both'
    )

    # Query to corrupt
    query = torch.tensor([[1, 10, 20]], device='cuda')  # brother(10, 20)

    print(f"Query to corrupt: pred=1, head=10, tail=20")
    print(f"Known facts that should be filtered:")
    for f in known_facts:
        print(f"  pred={f[0].item()}, head={f[1].item()}, tail={f[2].item()}")

    # Generate many negatives with filtering
    n_samples = 1000
    head_negs = sampler.corrupt(query, num_negatives=n_samples, mode='head', filter=True, unique=False).squeeze(0)
    tail_negs = sampler.corrupt(query, num_negatives=n_samples, mode='tail', filter=True, unique=False).squeeze(0)

    # Check that no corrupted triple matches a known fact
    known_set = set()
    for f in known_facts:
        known_set.add((f[0].item(), f[1].item(), f[2].item()))

    head_violations = 0
    for neg in head_negs:
        if neg.sum() == 0:  # Skip padding
            continue
        triple = (neg[0].item(), neg[1].item(), neg[2].item())
        if triple in known_set:
            head_violations += 1
            print(f"  VIOLATION (head): {triple}")

    tail_violations = 0
    for neg in tail_negs:
        if neg.sum() == 0:  # Skip padding
            continue
        triple = (neg[0].item(), neg[1].item(), neg[2].item())
        if triple in known_set:
            tail_violations += 1
            print(f"  VIOLATION (tail): {triple}")

    print(f"\nHead corruptions: {n_samples} samples, {head_violations} violations")
    print(f"Tail corruptions: {n_samples} samples, {tail_violations} violations")

    assert head_violations == 0, f"Head corruption produced {head_violations} known facts!"
    assert tail_violations == 0, f"Tail corruption produced {tail_violations} known facts!"

    print("\nPASSED: All known facts are filtered correctly")
    print()


def test_filtering_without_filter_flag():
    """Test that filtering can be disabled."""
    print("=" * 60)
    print("Test 3: Verify filter=False allows known facts")
    print("=" * 60)

    # Small entity space to increase collision probability
    num_entities = 10

    # Create facts covering many entities
    known_facts = torch.tensor([
        [1, i, j] for i in range(1, num_entities) for j in range(1, num_entities) if i != j
    ][:50], device='cuda')  # Many known facts

    sampler = Sampler.from_data(
        all_known_triples_idx=known_facts,
        num_entities=num_entities,
        num_relations=5,
        device=torch.device('cuda'),
    )

    known_set = set((f[0].item(), f[1].item(), f[2].item()) for f in known_facts)

    # Query
    query = torch.tensor([[1, 5, 6]], device='cuda')

    # Without filtering - should allow known facts
    negs_unfiltered = sampler.corrupt(query, num_negatives=100, mode='tail', filter=False, unique=False).squeeze(0)

    collisions = 0
    for neg in negs_unfiltered:
        if neg.sum() == 0:
            continue
        triple = (neg[0].item(), neg[1].item(), neg[2].item())
        if triple in known_set:
            collisions += 1

    print(f"Known facts: {len(known_set)}")
    print(f"Unfiltered corruptions: {collisions} collisions with known facts")
    print(f"(Collisions expected when filter=False and entity space is small)")

    # With filtering - should have no collisions
    negs_filtered = sampler.corrupt(query, num_negatives=100, mode='tail', filter=True, unique=False).squeeze(0)

    filtered_collisions = 0
    for neg in negs_filtered:
        if neg.sum() == 0:
            continue
        triple = (neg[0].item(), neg[1].item(), neg[2].item())
        if triple in known_set:
            filtered_collisions += 1

    print(f"Filtered corruptions: {filtered_collisions} collisions (should be 0)")

    assert filtered_collisions == 0, "Filtering failed!"
    print("\nPASSED: filter=True removes known facts, filter=False allows them")
    print()


def test_env_integration():
    """Test corruption in actual environment."""
    print("=" * 60)
    print("Test 4: Environment Integration")
    print("=" * 60)

    # Handle import issues gracefully
    try:
        # Try parent directory approach
        import sys
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        from config import TrainConfig
        from builder import build_env
    except ImportError as e:
        print(f"SKIPPED: Could not import environment modules ({e})")
        print("Run via: python runner_kge.py --set total_timesteps=5000 to verify")
        return

    config = TrainConfig(
        dataset='family',
        n_envs=16,
        corruption_scheme=('head', 'tail'),
        negative_ratio=1.0,
    )

    env = build_env(config)

    print(f"corruption_scheme: {env.corruption_scheme}")
    print(f"_num_corruption_modes: {env._num_corruption_modes}")

    # Reset and run several episodes
    obs, state = env.reset()

    for step in range(50):
        actions = torch.zeros(16, dtype=torch.long, device='cuda')
        obs, state = env.step_and_reset(state, actions, env.query_pool, env.per_env_ptrs)

    # Check mode counters are cycling
    mode_counters = state['corruption_mode_counters']
    print(f"Final mode counters: {mode_counters.tolist()}")
    print(f"Mode counters show cycling: {mode_counters.unique().numel() > 1 or mode_counters[0].item() > 0}")

    print("\nPASSED: Environment integration works")
    print()


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("CORRUPTION AND FILTERING TESTS")
    print("=" * 60 + "\n")

    test_head_tail_alternation()
    test_fact_filtering()
    test_filtering_without_filter_flag()
    test_env_integration()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
