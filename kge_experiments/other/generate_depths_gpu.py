"""
GPU-accelerated depth generation using the existing unification engine.
Processes queries one at a time with BFS to find minimum proof depth.
"""
from typing import List, Tuple, Optional
import time
import os
import sys

import torch
from torch import Tensor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_handler import DataHandler
from index_manager import IndexManager
from unification import UnificationEngineVectorized, standardize_vars, all_atoms_are_ground_facts


@torch.no_grad()
def check_single_query(
    query: Tensor,           # [3]
    engine: UnificationEngineVectorized,
    max_depth: int,
    max_atoms: int,
    excluded: Optional[Tensor] = None,
    chunk_size: int = 32,    # Process frontier in chunks to avoid OOM
) -> int:
    """Check provability for a single query. Returns min depth or -1."""
    device = query.device
    pad = engine.padding_idx
    A = max_atoms
    true_idx = engine.true_pred_idx
    false_idx = engine.false_pred_idx

    # Initialize frontier
    frontier_states = torch.full((1, A, 3), pad, dtype=torch.long, device=device)
    frontier_states[0, 0, :] = query
    frontier_vars = torch.tensor([engine.constant_no + 1], dtype=torch.long, device=device)

    # Excluded for train
    if excluded is not None:
        excluded_single = excluded.unsqueeze(0).unsqueeze(0)  # [1, 1, 3]
    else:
        excluded_single = None

    # Depth 0: is query a fact?
    if excluded is None:
        is_fact = all_atoms_are_ground_facts(
            frontier_states, engine.fact_hashes, engine.pack_base,
            engine.constant_no, pad, excluded_queries=None
        )
        if is_fact.any():
            return 0

    for depth in range(1, max_depth + 1):
        N = frontier_states.shape[0]
        if N == 0:
            break

        # Process frontier in chunks to avoid OOM
        all_next_states = []
        all_next_vars = []

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk_states = frontier_states[start:end]
            chunk_vars = frontier_vars[start:end]
            chunk_n = chunk_states.shape[0]

            if excluded is not None:
                exc = excluded_single.expand(chunk_n, -1, -1)
            else:
                exc = None

            derived, counts, new_vars = engine.get_derived_states_compiled(
                chunk_states, chunk_vars, exc
            )

            K = derived.shape[1]
            M = derived.shape[2]

            # Flatten
            derived_flat = derived.view(chunk_n * K, M, 3)
            first_preds = derived_flat[:, 0, 0]

            # Check for proofs
            if (first_preds == true_idx).any():
                return depth

            # Filter valid states
            is_pad = (first_preds == pad)
            is_false = (first_preds == false_idx)
            is_true = (first_preds == true_idx)
            atom_counts = (derived_flat[:, :, 0] != pad).sum(dim=1)
            valid = ~is_pad & ~is_false & ~is_true & (atom_counts <= max_atoms)

            if valid.any():
                valid_idx = valid.nonzero(as_tuple=True)[0]
                valid_states = derived_flat[valid_idx]
                state_indices = valid_idx // K
                valid_vars = new_vars[state_indices]

                # Pad to A atoms
                num_valid = valid_states.shape[0]
                if M < A:
                    padded = torch.full((num_valid, A, 3), pad, dtype=torch.long, device=device)
                    padded[:, :M, :] = valid_states
                    all_next_states.append(padded)
                else:
                    all_next_states.append(valid_states[:, :A, :].clone())
                all_next_vars.append(valid_vars)

        if not all_next_states:
            break

        frontier_states = torch.cat(all_next_states, dim=0)
        frontier_vars = torch.cat(all_next_vars, dim=0)

    return -1


def generate_depths_for_dataset(
    dataset_name: str,
    splits: List[str],
    data_path: str,
    max_depth: int = 7,
    max_atoms: int = 20,
    device: str = 'cuda',
    max_derived: int = 256,
):
    """Generate depth files using GPU-accelerated BFS."""
    root_dir = os.path.join(data_path, dataset_name)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Device: {device}, Max depth: {max_depth}, Max atoms: {max_atoms}")
    print(f"{'='*60}\n")

    # Load data
    dh = DataHandler(
        dataset_name=dataset_name,
        base_path=data_path,
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="train.txt",
        load_depth_info=False,
        filter_queries_by_rules=False,
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
    dh.materialize_indices(im=im, device=device)

    # Create engine with moderate parameters
    max_fact_pairs = min(2000, len(dh.facts))
    max_rule_pairs = min(4000, len(dh.rules) * 100)

    engine = UnificationEngineVectorized.from_index_manager(
        im,
        padding_atoms=max_atoms,
        parity_mode=True,
        max_derived_per_state=max_derived,
        end_proof_action=False,
        max_fact_pairs=max_fact_pairs,
        max_rule_pairs=max_rule_pairs,
    )

    # Move to GPU
    engine.fact_seg_starts = engine.fact_seg_starts.to(device)
    engine.fact_seg_lens = engine.fact_seg_lens.to(device)
    engine.rule_seg_starts = engine.rule_seg_starts.to(device)
    engine.rule_seg_lens = engine.rule_seg_lens.to(device)

    # Use uncompiled standardization for variable batch sizes
    extra_new_vars = engine.max_rule_body_size + 2
    def _std_fn(states, counts, next_var_indices, input_states):
        return standardize_vars(
            states, counts, next_var_indices,
            engine.constant_no, engine.padding_idx,
            input_states=input_states,
            extra_new_vars=extra_new_vars,
            out_of_place=True,
        )
    engine._standardize_vars_fn = _std_fn
    engine._standardize_compiled = False
    engine.compile_standardize = False

    print(f"Loaded {len(dh.facts)} facts, {len(dh.rules)} rules")
    print(f"Constants: {im.constant_no}, Predicates: {im.predicate_no}")
    print(f"Max derived: {max_derived}, K_max: {engine.K_max}")

    for split in splits:
        print(f"\n--- Processing {split} split ---")
        start_time = time.time()

        query_map = {
            'train': dh.train_queries,
            'valid': dh.valid_queries,
            'test': dh.test_queries
        }
        queries_list = query_map.get(split, [])

        if not queries_list:
            print(f"No queries found for {split}, skipping.")
            continue

        queries_tensor = im.queries_to_tensor(queries_list, device).squeeze(1)
        N = queries_tensor.shape[0]
        is_train = (split == 'train')
        output_file = os.path.join(root_dir, f'{split}_depths_gpu.txt')

        print(f"Processing {N} queries...")

        depths = []
        proven_count = 0

        for i in range(N):
            query = queries_tensor[i]
            excluded = query if is_train else None

            depth = check_single_query(
                query, engine,
                max_depth=max_depth,
                max_atoms=max_atoms,
                excluded=excluded,
            )
            depths.append(depth)
            if depth >= 0:
                proven_count += 1

            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                qps = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"\r  {i+1}/{N} queries, proven: {proven_count}, {qps:.1f} q/s", end='', flush=True)

        print()

        # Save
        with open(output_file, 'w') as f:
            for i, query in enumerate(queries_list):
                clean_query = str(query).replace(' ', '')
                f.write(f"{clean_query} {depths[i]}\n")

        elapsed = time.time() - start_time

        depth_counts = {}
        for d in depths:
            depth_counts[d] = depth_counts.get(d, 0) + 1

        print(f"\n{split} Summary:")
        print(f"  Total queries: {N}")
        print(f"  Provable: {proven_count} ({proven_count/N:.1%})")
        print(f"  Depth distribution: {dict(sorted(depth_counts.items()))}")
        print(f"  Time: {elapsed:.1f}s ({N/elapsed:.1f} queries/sec)")
        print(f"  Saved to: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate depth files using GPU engine')
    parser.add_argument('--datasets', nargs='+', default=['family'])
    parser.add_argument('--splits', nargs='+', default=['train', 'valid', 'test'])
    parser.add_argument('--max_depth', type=int, default=4)
    parser.add_argument('--max_atoms', type=int, default=20)
    parser.add_argument('--max_derived', type=int, default=256)
    args = parser.parse_args()

    data_path = os.path.join(os.path.dirname(__file__), '..', 'data') + '/'

    for dataset in args.datasets:
        generate_depths_for_dataset(
            dataset_name=dataset,
            splits=args.splits,
            data_path=data_path,
            max_depth=args.max_depth,
            max_atoms=args.max_atoms,
            max_derived=args.max_derived,
        )

    print("\n" + "="*60)
    print("All datasets processed!")
    print("="*60)
