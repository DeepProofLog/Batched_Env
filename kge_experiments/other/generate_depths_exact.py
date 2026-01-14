"""
Exact Depth Engine: Gold standard for minimum proof depth calculation.

Uses exhaustive BFS with explicit rule/fact application to guarantee
finding the absolute shortest proof for each query.

SEMANTICS DIFFERENCE: GPU vs Prolog
===================================

This engine uses TRUE PROLOG SEMANTICS where each unification/substitution
counts as ONE step. The GPU-based depth generation uses a different semantics.

Example: locatedInCR(jordan, asia)

GPU Semantics (used in generate_depths_gpu.py):
  Depth 0: locatedInCR(jordan, asia)
  Depth 1: {neighborOf(jordan, V), locatedInCR(V, asia)}  [rule applied]
  Depth 2: V=palestine makes BOTH atoms facts -> PROOF
  Total depth: 2

Prolog Semantics (used in this engine):
  Depth 0: locatedInCR(jordan, asia)
  Depth 1: {neighborOf(jordan, V), locatedInCR(V, asia)}  [rule applied]
  Depth 2: {locatedInCR(palestine, asia)}  [resolved neighborOf(jordan,palestine)]
  Depth 3: {}  [resolved locatedInCR(palestine,asia)] -> PROOF
  Total depth: 3

EXPECTED DIFFERENCES:
- Prolog depths >= GPU depths (Prolog counts each substitution)
"""
from typing import List, Dict, Set, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import Term, Rule

# Use tuples for atoms: (predicate, arg1, arg2) - much faster than dataclass
Atom = Tuple[str, str, str]


def is_var(term: str) -> bool:
    """Check if term is a variable (uppercase first letter)."""
    return term[0].isupper()


def unify_atoms(a1: Atom, a2: Atom) -> Optional[Dict[str, str]]:
    """Unify two atoms, return bindings or None."""
    if a1[0] != a2[0]:  # predicate mismatch
        return None

    bindings = {}

    # Unify arg1
    t1, t2 = a1[1], a2[1]
    while t1 in bindings:
        t1 = bindings[t1]
    while t2 in bindings:
        t2 = bindings[t2]

    if t1 != t2:
        if is_var(t1):
            bindings[t1] = t2
        elif is_var(t2):
            bindings[t2] = t1
        else:
            return None

    # Unify arg2
    t1, t2 = a1[2], a2[2]
    while t1 in bindings:
        t1 = bindings[t1]
    while t2 in bindings:
        t2 = bindings[t2]

    if t1 != t2:
        if is_var(t1):
            bindings[t1] = t2
        elif is_var(t2):
            bindings[t2] = t1
        else:
            return None

    return bindings


def apply_bindings_atom(atom: Atom, bindings: Dict[str, str]) -> Atom:
    """Apply bindings to single atom."""
    a1 = bindings.get(atom[1], atom[1])
    a2 = bindings.get(atom[2], atom[2])
    return (atom[0], a1, a2)


def apply_bindings_state(state: Tuple[Atom, ...], bindings: Dict[str, str]) -> Tuple[Atom, ...]:
    """Apply bindings to state (tuple of atoms)."""
    if not bindings:
        return state
    return tuple(apply_bindings_atom(a, bindings) for a in state)


def normalize_state(state: Tuple[Atom, ...]) -> Tuple[Atom, ...]:
    """Normalize variable names for canonical comparison."""
    if not state:
        return state

    # Sort first for deterministic ordering
    sorted_state = sorted(state)

    var_map = {}
    counter = 0
    result = []

    for pred, a1, a2 in sorted_state:
        if is_var(a1):
            if a1 not in var_map:
                var_map[a1] = f"_{counter}"
                counter += 1
            a1 = var_map[a1]
        if is_var(a2):
            if a2 not in var_map:
                var_map[a2] = f"_{counter}"
                counter += 1
            a2 = var_map[a2]
        result.append((pred, a1, a2))

    return tuple(result)


class FastExactEngine:
    """Optimized BFS engine using tuples for speed."""

    def __init__(
        self,
        facts: Set[Atom],
        rules: List[Tuple[Atom, Tuple[Atom, ...]]],
        max_depth: int = 7,
        max_atoms: int = 10,
        max_frontier: int = 50000,
    ):
        self.facts = facts
        self.rules = rules
        self.max_depth = max_depth
        self.max_atoms = max_atoms
        self.max_frontier = max_frontier

        # Index facts by predicate
        self.facts_by_pred: Dict[str, List[Atom]] = {}
        for f in facts:
            self.facts_by_pred.setdefault(f[0], []).append(f)

        # Index rules by head predicate
        self.rules_by_pred: Dict[str, List[Tuple[Atom, Tuple[Atom, ...]]]] = {}
        for head, body in rules:
            self.rules_by_pred.setdefault(head[0], []).append((head, body))

    def find_depth(self, query: Atom, exclude_query: bool = False) -> int:
        """Find minimum proof depth using BFS with Prolog-style left-to-right resolution."""
        excluded = query if exclude_query else None

        # Depth 0: direct fact check
        if query in self.facts and not exclude_query:
            return 0

        # BFS - Prolog style: always resolve FIRST atom only
        frontier = [(query,)]  # List of states (tuples of atoms)
        visited: Set[Tuple[Atom, ...]] = {(query,)}
        var_counter = 1000

        for depth in range(1, self.max_depth + 1):
            if not frontier:
                break

            next_frontier = []

            for state in frontier:
                # Prolog-style: only resolve the FIRST atom
                atom = state[0]
                remaining = state[1:]
                pred = atom[0]
                is_ground = not (is_var(atom[1]) or is_var(atom[2]))

                # Try fact resolution
                if is_ground:
                    # Ground atom - check if it's a fact
                    if atom in self.facts and atom != excluded:
                        if not remaining:  # Empty = proof!
                            return depth
                        norm = normalize_state(remaining)
                        if norm not in visited:
                            visited.add(norm)
                            next_frontier.append(remaining)
                else:
                    # Non-ground - unify with facts
                    for fact in self.facts_by_pred.get(pred, []):
                        if fact == excluded:
                            continue
                        bindings = unify_atoms(atom, fact)
                        if bindings is not None:
                            new_state = apply_bindings_state(remaining, bindings)
                            if not new_state:  # Empty = proof!
                                return depth
                            if len(new_state) <= self.max_atoms:
                                norm = normalize_state(new_state)
                                if norm not in visited:
                                    visited.add(norm)
                                    next_frontier.append(new_state)

                # Try rule application
                for head, body in self.rules_by_pred.get(pred, []):
                    # Standardize variables
                    rule_vars = set()
                    for a in (head,) + body:
                        if is_var(a[1]):
                            rule_vars.add(a[1])
                        if is_var(a[2]):
                            rule_vars.add(a[2])

                    renaming = {v: f"V{var_counter + j}" for j, v in enumerate(sorted(rule_vars))}
                    var_counter += len(rule_vars)

                    std_head = apply_bindings_atom(head, renaming)
                    std_body = tuple(apply_bindings_atom(b, renaming) for b in body)

                    bindings = unify_atoms(atom, std_head)
                    if bindings is not None:
                        new_body = apply_bindings_state(std_body, bindings)
                        new_remaining = apply_bindings_state(remaining, bindings)
                        new_state = new_remaining + new_body

                        if len(new_state) <= self.max_atoms:
                            norm = normalize_state(new_state)
                            if norm not in visited:
                                visited.add(norm)
                                next_frontier.append(new_state)

                if len(next_frontier) >= self.max_frontier:
                    break

            frontier = next_frontier

        return -1


# Global engine for worker processes
_worker_engine = None


def _init_worker(facts, rules, max_depth, max_atoms, max_frontier):
    """Initialize worker process with engine."""
    global _worker_engine
    _worker_engine = FastExactEngine(
        facts=facts,
        rules=rules,
        max_depth=max_depth,
        max_atoms=max_atoms,
        max_frontier=max_frontier,
    )


def _process_query(args):
    """Worker function to process single query."""
    query_atom, exclude_query, idx = args
    depth = _worker_engine.find_depth(query_atom, exclude_query=exclude_query)
    return idx, depth


def load_dataset(dataset_name: str, data_path: str):
    """Load dataset and return facts, rules, queries."""
    from data_handler import DataHandler

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

    # Convert to tuples
    facts = {(f.predicate, f.args[0], f.args[1]) for f in dh.facts}

    rules = []
    for rule in dh.rules:
        head = (rule.head.predicate, rule.head.args[0], rule.head.args[1])
        body = tuple((b.predicate, b.args[0], b.args[1]) for b in rule.body)
        rules.append((head, body))

    return facts, rules, dh.train_queries, dh.valid_queries, dh.test_queries


def generate_depths_for_dataset(
    dataset_name: str,
    splits: List[str],
    data_path: str,
    max_depth: int = 7,
    max_atoms: int = 10,
    n_workers: int = None,
):
    """Generate depth files using exact Prolog semantics."""
    if n_workers is None:
        n_workers = mp.cpu_count()

    print(f"\n{'='*60}")
    print(f"Exact Depth Engine (Prolog Semantics) - Parallel")
    print(f"Dataset: {dataset_name}, Workers: {n_workers}")
    print(f"Max depth: {max_depth}, Max atoms: {max_atoms}")
    print(f"{'='*60}\n")

    facts, rules, train_q, valid_q, test_q = load_dataset(dataset_name, data_path)

    print(f"Loaded {len(facts)} facts, {len(rules)} rules")
    print(f"Queries: train={len(train_q)}, valid={len(valid_q)}, test={len(test_q)}")

    print("\nRules:")
    for i, (head, body) in enumerate(rules):
        body_str = " & ".join(f"{b[0]}({b[1]},{b[2]})" for b in body)
        print(f"  {i}: {head[0]}({head[1]},{head[2]}) :- {body_str}")

    query_sets = {'train': train_q, 'valid': valid_q, 'test': test_q}
    root_dir = os.path.join(data_path, dataset_name)

    for split in splits:
        queries = query_sets.get(split, [])
        if not queries:
            print(f"No queries for {split}, skipping.")
            continue

        print(f"\n--- Processing {split} ({len(queries)} queries) ---")
        start = time.time()

        is_train = (split == 'train')

        # Prepare work items
        work_items = [
            ((q.predicate, q.args[0], q.args[1]), is_train, i)
            for i, q in enumerate(queries)
        ]

        # Process in parallel
        depths = [None] * len(queries)
        completed = 0

        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(facts, rules, max_depth, max_atoms, 100000),
        ) as executor:
            futures = {executor.submit(_process_query, item): item for item in work_items}

            for future in as_completed(futures):
                idx, depth = future.result()
                depths[idx] = depth
                completed += 1
                proven = sum(1 for d in depths if d is not None and d >= 0)
                print(f"\r  {completed}/{len(queries)}, proven: {proven}", end='', flush=True)

        print()
        elapsed = time.time() - start

        # Collect results
        results = list(zip(queries, depths))
        depth_counts = {}
        for d in depths:
            depth_counts[d] = depth_counts.get(d, 0) + 1

        # Save
        out_file = os.path.join(root_dir, f'{split}_depths_exact.txt')
        with open(out_file, 'w') as f:
            for q, d in results:
                f.write(f"{q.predicate}({q.args[0]},{q.args[1]}) {d}\n")

        total_proven = sum(1 for d in depths if d >= 0)
        print(f"\n{split} Summary:")
        print(f"  Total: {len(queries)}, Provable: {total_proven} ({total_proven/len(queries):.1%})")
        print(f"  Depths: {dict(sorted(depth_counts.items()))}")
        print(f"  Time: {elapsed:.1f}s ({len(queries)/elapsed:.1f} q/s)")
        print(f"  Saved: {out_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate depth files (Prolog semantics)')
    parser.add_argument('--datasets', nargs='+', default=['countries_s3'])
    parser.add_argument('--splits', nargs='+', default=['train','valid', 'test'])
    parser.add_argument('--max_depth', type=int, default=15)
    parser.add_argument('--max_atoms', type=int, default=10)
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers')
    args = parser.parse_args()

    data_path = os.path.join(os.path.dirname(__file__), '..', 'data') + '/'

    for dataset in args.datasets:
        generate_depths_for_dataset(
            dataset_name=dataset,
            splits=args.splits,
            data_path=data_path,
            max_depth=args.max_depth,
            max_atoms=args.max_atoms,
            n_workers=args.workers,
        )

    print("\n" + "="*60)
    print("Done!")
    print("="*60)
