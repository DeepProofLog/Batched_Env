"""
Fast depth generation using optimized BFS with SB3 unification.

Key optimizations:
1. Reuse SB3's optimized unification functions
2. Optimized BFS loop with minimal overhead
3. Aggressive caching and early termination
4. Multiprocessing for parallel query processing

Usage:
    from fast_unifier import FastUnifier
    unifier = FastUnifier(data_handler)
    depth = unifier.check_provability(query, max_depth=7)

    # For batch processing with parallelization:
    depths = unifier.check_provability_batch(queries, max_depth=7, n_workers=4)
"""
from typing import Dict, List, Tuple, Optional, Set, FrozenSet
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sb3.sb3_dataset import DataHandler
from sb3.sb3_index_manager import IndexManager
from sb3.sb3_utils import Term, Rule
from sb3.sb3_unification import get_next_unification_python


# Global variables for worker processes (set during pool initialization)
_worker_facts_set: Optional[FrozenSet[Term]] = None
_worker_fact_index: Optional[Dict] = None
_worker_rules: Optional[Dict] = None
_worker_var_start: Optional[int] = None


def _init_worker(facts_set, fact_index, rules, var_start):
    """Initialize worker process with shared data."""
    global _worker_facts_set, _worker_fact_index, _worker_rules, _worker_var_start
    _worker_facts_set = facts_set
    _worker_fact_index = fact_index
    _worker_rules = rules
    _worker_var_start = var_start


def _worker_check_provability(query: Term, max_depth: int, max_atoms: int) -> int:
    """Worker function for parallel provability check."""
    excluded_fact = query
    facts_set = _worker_facts_set
    fact_index = _worker_fact_index
    rules = _worker_rules
    var_start = _worker_var_start
    _frozenset = frozenset

    initial_state = [query]
    frontier = deque([(initial_state, 0)])
    visited: Set[FrozenSet[Term]] = {_frozenset(initial_state)}
    visited_add = visited.add

    while frontier:
        current_state, current_depth = frontier.popleft()

        if current_depth >= max_depth:
            continue

        next_states, _ = get_next_unification_python(
            current_state,
            facts_set=facts_set,
            facts_indexed=fact_index,
            rules=rules,
            excluded_fact=excluded_fact,
            verbose=0,
            next_var_index=var_start,
        )

        next_depth = current_depth + 1

        for next_state in next_states:
            if not next_state:
                continue

            first_pred = next_state[0].predicate

            if first_pred == 'True':
                if all(t.predicate == 'True' for t in next_state):
                    return next_depth
                continue

            if first_pred == 'False':
                continue
            if any(t.predicate == 'False' for t in next_state):
                continue

            if len(next_state) > max_atoms:
                continue

            state_key = _frozenset(next_state)
            if state_key in visited:
                continue
            visited_add(state_key)

            frontier.append((next_state, next_depth))

    return -1


class FastUnifier:
    """Optimized BFS prover reusing SB3 unification."""

    def __init__(self, data_handler: DataHandler):
        """Initialize from DataHandler."""
        self.facts_set = frozenset(data_handler.facts)

        # Create index manager for fact indexing
        self.index_manager = IndexManager(
            data_handler.constants,
            data_handler.predicates,
            max_total_vars=100,
            constants_images=set(),
            constant_images_no=0,
            rules=data_handler.rules,
            max_arity=data_handler.max_arity,
            device='cpu',
            padding_atoms=4
        )
        self.index_manager.build_fact_index(data_handler.facts)
        self.rules = self.index_manager.rules_by_pred
        self.var_start = self.index_manager.variable_start_index

    def check_provability(
        self,
        query: Term,
        is_train_data: bool = False,
        max_depth: int = 7,
        max_atoms: int = 20,
    ) -> int:
        """Check provability using optimized BFS.

        Args:
            query: Query term to prove
            is_train_data: If True, exclude query from facts (SB3 always does this)
            max_depth: Maximum proof depth
            max_atoms: Maximum atoms per state

        Returns:
            Minimum proof depth, or -1 if not provable
        """
        # Always exclude query from facts (matching SB3 behavior)
        excluded_fact = query

        # Cache lookups
        facts_set = self.facts_set
        fact_index = self.index_manager.fact_index
        rules = self.rules
        var_start = self.var_start
        _frozenset = frozenset

        # BFS with deque
        initial_state = [query]
        frontier = deque([(initial_state, 0)])
        visited: Set[FrozenSet[Term]] = {_frozenset(initial_state)}
        visited_add = visited.add

        while frontier:
            current_state, current_depth = frontier.popleft()

            if current_depth >= max_depth:
                continue

            # Generate next states using SB3's optimized function
            next_states, _ = get_next_unification_python(
                current_state,
                facts_set=facts_set,
                facts_indexed=fact_index,
                rules=rules,
                excluded_fact=excluded_fact,
                verbose=0,
                next_var_index=var_start,
            )

            next_depth = current_depth + 1

            for next_state in next_states:
                # Quick empty check
                if not next_state:
                    continue

                # Check first predicate
                first_pred = next_state[0].predicate

                # Check for proof (all True)
                if first_pred == 'True':
                    if all(t.predicate == 'True' for t in next_state):
                        return next_depth
                    continue  # Skip states starting with True (will be processed)

                # Skip dead ends (False)
                if first_pred == 'False':
                    continue
                if any(t.predicate == 'False' for t in next_state):
                    continue

                # Skip if too many atoms
                if len(next_state) > max_atoms:
                    continue

                # Deduplicate
                state_key = _frozenset(next_state)
                if state_key in visited:
                    continue
                visited_add(state_key)

                frontier.append((next_state, next_depth))

        return -1

    def check_provability_batch(
        self,
        queries: List[Term],
        max_depth: int = 7,
        max_atoms: int = 20,
        n_workers: Optional[int] = None,
    ) -> List[int]:
        """Check provability for multiple queries in parallel.

        Args:
            queries: List of query terms to prove
            max_depth: Maximum proof depth
            max_atoms: Maximum atoms per state
            n_workers: Number of worker processes (default: CPU count)

        Returns:
            List of depths (-1 if not provable)
        """
        if n_workers is None:
            n_workers = mp.cpu_count()

        # For small batches, sequential is faster due to overhead
        if len(queries) < n_workers * 2:
            return [
                self.check_provability(q, is_train_data=True, max_depth=max_depth, max_atoms=max_atoms)
                for q in queries
            ]

        # Parallel processing
        facts_set = self.facts_set
        fact_index = self.index_manager.fact_index
        rules = self.rules
        var_start = self.var_start

        results = [0] * len(queries)

        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(facts_set, fact_index, rules, var_start)
        ) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(_worker_check_provability, q, max_depth, max_atoms): i
                for i, q in enumerate(queries)
            }

            # Collect results
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()

        return results
