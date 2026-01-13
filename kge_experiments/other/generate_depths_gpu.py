"""
GPU-accelerated depth generation using the compiled unification engine.
Runs batched exhaustive search to find minimum proof depth for each query.
Output format: "query depth" where depth=-1 means non-provable.
"""
from typing import List, Dict, Tuple, Optional
import time
import os
import sys

import torch
from torch import Tensor

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_handler import DataHandler
from index_manager import IndexManager
from unification import UnificationEngineVectorized, all_atoms_are_ground_facts


def check_provability_batched(
    queries: Tensor,                    # [N, 3] query atoms (pred, arg1, arg2)
    engine: UnificationEngineVectorized,
    max_depth: int = 7,
    max_atoms: int = 20,
    is_train: bool = False,
    batch_size: int = 256,
    verbose: bool = True,
) -> Tuple[Tensor, Tensor]:
    """
    Check provability for a batch of queries using GPU-accelerated unification.

    Args:
        queries: [N, 3] tensor of query atoms
        engine: Compiled unification engine
        max_depth: Maximum proof depth to search
        max_atoms: Maximum atoms per state (for pruning)
        is_train: If True, exclude query from facts (cycle prevention)
        batch_size: Batch size for processing
        verbose: Print progress

    Returns:
        proven_mask: [N] bool tensor - True if query is provable
        min_depth: [N] int tensor - minimum proof depth (-1 if not provable)
    """
    device = queries.device
    N = queries.shape[0]
    pad = engine.padding_idx

    # Results tracking
    proven_mask = torch.zeros(N, dtype=torch.bool, device=device)
    min_depth = torch.full((N,), -1, dtype=torch.long, device=device)

    # Process in batches
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_queries = queries[start:end]  # [B, 3]
        B = batch_queries.shape[0]

        # Initialize states: each query as initial state [B, A, 3]
        A = max_atoms
        states = torch.full((B, A, 3), pad, dtype=torch.long, device=device)
        states[:, 0, :] = batch_queries  # First atom is the query

        # Track which queries in batch are still active
        active = torch.ones(B, dtype=torch.bool, device=device)
        batch_min_depth = torch.full((B,), -1, dtype=torch.long, device=device)

        # Next variable indices
        next_vars = torch.full((B,), engine.constant_no + 1, dtype=torch.long, device=device)

        # Excluded queries for cycle prevention (train mode)
        excluded = None
        if is_train:
            excluded = batch_queries.unsqueeze(1)  # [B, 1, 3]

        # BFS-style exploration: track all frontier states per query
        # frontier[i] = list of states for query i
        # For simplicity, we'll process depth-by-depth and track multiple states

        # For each depth level
        for depth in range(1, max_depth + 1):
            if not active.any():
                break

            # Get derived states for current frontier
            derived, counts, new_vars = engine.get_derived_states_compiled(
                states, next_vars, excluded
            )
            # derived: [B, K_max, M_max, 3], counts: [B]

            K_max = derived.shape[1]
            M_max = derived.shape[2]

            # Check for proofs: True predicate in first atom
            first_preds = derived[:, :, 0, 0]  # [B, K_max]
            is_true = (first_preds == engine.true_pred_idx)  # [B, K_max]

            # A query is proven if any derived state is True
            proven_this_round = is_true.any(dim=1) & active  # [B]

            # Record depth for newly proven queries
            batch_min_depth = torch.where(
                proven_this_round & (batch_min_depth == -1),
                torch.full_like(batch_min_depth, depth),
                batch_min_depth
            )

            # Mark proven queries as inactive
            active = active & ~proven_this_round

            # For remaining active queries, expand frontier
            # Take first valid derived state as next state (BFS approximation)
            # In full BFS, we'd track all states - this is a simplification

            # Find first valid state per batch element
            valid_mask = (derived[:, :, 0, 0] != pad) & (derived[:, :, 0, 0] != engine.false_pred_idx)

            # Get first valid state index
            first_valid_idx = valid_mask.long().argmax(dim=1)  # [B]
            has_valid = valid_mask.any(dim=1)  # [B]

            # Extract next states
            batch_idx = torch.arange(B, device=device)
            next_states_raw = derived[batch_idx, first_valid_idx]  # [B, M_max, 3]

            # Pad/truncate to A atoms
            if M_max >= A:
                next_states = next_states_raw[:, :A, :]
            else:
                next_states = torch.full((B, A, 3), pad, dtype=torch.long, device=device)
                next_states[:, :M_max, :] = next_states_raw

            # Update states only for active queries with valid successors
            update_mask = active & has_valid
            states = torch.where(
                update_mask.view(B, 1, 1),
                next_states,
                states
            )

            # Mark queries with no valid successors as not provable
            active = active & has_valid

            next_vars = new_vars

        # Store results
        proven_mask[start:end] = (batch_min_depth != -1)
        min_depth[start:end] = batch_min_depth

        if verbose and (end % 500 == 0 or end == N):
            proven_so_far = proven_mask[:end].sum().item()
            print(f"\rProcessed {end}/{N} queries, proven: {proven_so_far}", end='', flush=True)

    if verbose:
        print()

    return proven_mask, min_depth


def generate_depths_for_dataset(
    dataset_name: str,
    splits: List[str],
    data_path: str,
    device: str = 'cuda',
    max_depth: int = 7,
    max_atoms: int = 20,
    batch_size: int = 256,
):
    """Generate depth files for a dataset using GPU-accelerated search."""

    root_dir = os.path.join(data_path, dataset_name)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Device: {device}, Max depth: {max_depth}, Batch size: {batch_size}")
    print(f"{'='*60}\n")

    # Load data (once per dataset)
    dh = DataHandler(
        dataset_name=dataset_name,
        base_path=data_path,
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="train.txt",
        load_depth_info=False,  # Don't try to load existing depth files
        filter_queries_by_rules=False,  # Load all queries
        corruption_mode="dynamic",
    )

    # Create index manager
    im = IndexManager(
        constants=dh.constants,
        predicates=dh.predicates,
        max_total_runtime_vars=1000,
        max_arity=dh.max_arity,
        padding_atoms=max_atoms,
        device=device,
        rules=dh.rules,
    )

    # Materialize indices
    dh.materialize_indices(im=im, device=device)

    # Create unification engine
    engine = UnificationEngineVectorized.from_index_manager(
        im,
        padding_atoms=max_atoms,
        parity_mode=False,
        max_derived_per_state=128,
        end_proof_action=False,
    )

    # Fix device issues - move segment tensors to GPU
    engine.fact_seg_starts = engine.fact_seg_starts.to(device)
    engine.fact_seg_lens = engine.fact_seg_lens.to(device)
    engine.rule_seg_starts = engine.rule_seg_starts.to(device)
    engine.rule_seg_lens = engine.rule_seg_lens.to(device)

    # Disable torch.compile for variable batch sizes - reinitialize standardization fn
    from unification import standardize_vars
    extra_new_vars = engine.max_rule_body_size + 2

    def _std_fn_uncompiled(states, counts, next_var_indices, input_states):
        return standardize_vars(
            states, counts, next_var_indices,
            engine.constant_no, engine.padding_idx,
            input_states=input_states,
            extra_new_vars=extra_new_vars,
            out_of_place=True,
        )

    engine._standardize_vars_fn = _std_fn_uncompiled
    engine._standardize_compiled = False
    engine.compile_standardize = False

    print(f"Loaded {len(dh.facts)} facts, {len(dh.rules)} rules")
    print(f"Constants: {im.constant_no}, Predicates: {im.predicate_no}")

    for split in splits:
        print(f"\n--- Processing {split} split ---")
        start_time = time.time()

        # Get queries for this split
        query_map = {
            'train': dh.train_queries,
            'valid': dh.valid_queries,
            'test': dh.test_queries
        }
        queries_list = query_map.get(split, [])

        if not queries_list:
            print(f"No queries found for {split}, skipping.")
            continue

        # Convert to tensor
        queries_tensor = im.queries_to_tensor(queries_list, device)  # [N, 1, 3]
        queries_tensor = queries_tensor.squeeze(1)  # [N, 3]
        N = queries_tensor.shape[0]

        print(f"Processing {N} queries...")

        is_train = (split == 'train')

        # Run batched provability check
        proven_mask, min_depth = check_provability_batched(
            queries_tensor, engine,
            max_depth=max_depth,
            max_atoms=max_atoms,
            is_train=is_train,
            batch_size=batch_size,
            verbose=True
        )

        # Save results
        output_file = os.path.join(root_dir, f'{split}_depths.txt')
        with open(output_file, 'w') as f:
            for i, query in enumerate(queries_list):
                clean_query = str(query).replace(' ', '')
                depth_val = min_depth[i].item()
                f.write(f"{clean_query} {depth_val}\n")

        elapsed = time.time() - start_time
        total_proven = proven_mask.sum().item()

        # Depth distribution
        depth_counts = {}
        for d in range(1, max_depth + 1):
            count = (min_depth == d).sum().item()
            if count > 0:
                depth_counts[d] = count

        print(f"\n{split} Summary:")
        print(f"  Total queries: {N}")
        print(f"  Provable: {total_proven} ({total_proven/N:.1%})")
        print(f"  Non-provable: {N - total_proven}")
        print(f"  Depth distribution: {depth_counts}")
        print(f"  Time: {elapsed:.1f}s ({N/elapsed:.1f} queries/sec)")
        print(f"  Saved to: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate depth files for datasets')
    parser.add_argument('--datasets', nargs='+', default=['nations', 'umls'],
                        help='Datasets to process')
    parser.add_argument('--splits', nargs='+', default=['train', 'valid', 'test'],
                        help='Splits to process')
    parser.add_argument('--max_depth', type=int, default=7,
                        help='Maximum proof depth')
    parser.add_argument('--max_atoms', type=int, default=20,
                        help='Maximum atoms per state')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for GPU processing')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    args = parser.parse_args()

    data_path = os.path.join(os.path.dirname(__file__), '..', 'data') + '/'

    for dataset in args.datasets:
        generate_depths_for_dataset(
            dataset_name=dataset,
            splits=args.splits,
            data_path=data_path,
            device=args.device,
            max_depth=args.max_depth,
            max_atoms=args.max_atoms,
            batch_size=args.batch_size,
        )

    print("\n" + "="*60)
    print("All datasets processed!")
    print("="*60)
