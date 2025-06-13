from typing import List, Dict, Optional, TypedDict, Tuple
from utils import Term, Rule
from python_unification import get_next_unification_python
from python_unification import get_next_unification_python_old as get_next_unification_python
import time
from dataset import DataHandler
from index_manager import IndexManager


class PathCounts(TypedDict):
    proven: int
    failed: int
    truncated: int
    open_at_depth: int
    error: Optional[str]


class QueryResult(TypedDict):
    query: str
    depth_proven: Optional[int]
    counts: PathCounts

class PerDepthStats(TypedDict):
    """Holds the new, more detailed statistics for a single depth."""
    queries_without_proof: int
    queries_with_proof: int
    # For fully proven queries:
    fully_proven_count: int
    fully_proven_avg_paths: float
    # For distribution of all provable queries:
    successful_path_ratio_bins: List[int] # Holds counts for 5 bins: 0-20%, 20-40%, ..., 80-100%


class AnalysisResult(TypedDict):
    """The main result object, updated to use the new PerDepthStats."""
    max_depth: int
    max_atoms: int
    total_queries: int
    per_depth_stats: Dict[int, PerDepthStats]
    query_results: List[QueryResult]
    duration: float

def count_paths_at_depth(
    initial: List[Term],
    depth: int,
    max_atoms: int,
    rules: List[Rule],
    facts: List[Term],
    index_manager=None,
    is_train: bool = False,
) -> PathCounts:
    """
    BFS expand initial state up to exactly `depth` levels and count path outcomes.
    (This function remains unchanged)
    """
    if not initial:
        return PathCounts(proven=0, failed=0, truncated=0, open_at_depth=0, error='empty_query')
    if depth < 0 or max_atoms <= 0:
        return PathCounts(proven=0, failed=0, truncated=0, open_at_depth=0, error='invalid_params')

    proven = failed = truncated = 0
    current = [initial]
    open_at_depth = 0

    for d in range(depth):
        next_states = []
        entering = len(current)
        resolved = 0
        final_step = (d == depth - 1)

        for state in current:
            branches, _ = get_next_unification_python(
                state,
                facts_set=facts,
                facts_indexed=getattr(index_manager, 'fact_index', None),
                rules=getattr(index_manager, 'rules', rules),
                excluded_fact=initial[0] if is_train else None,
                next_var_index=getattr(index_manager, 'next_var_index', 0),
                verbose=0,
            )
            if not branches:
                failed += 1
                resolved += 1
                continue
            live = False

            for s in branches:
                if not s or any(t.predicate == 'False' for t in s):
                    failed += 1
                elif all(t.predicate == 'True' for t in s):
                    proven += 1
                    resolved += 1
                elif len(s) > max_atoms:
                    truncated += 1
                else:
                    live = True
                    next_states.append(s)

            if not live and not any(all(t.predicate == 'True' for t in s) for s in branches):
                resolved += 1

        if final_step:
            open_at_depth = entering - resolved
        current = next_states

    return {
        'proven': proven,
        'failed': failed,
        'truncated': truncated,
        'open_at_depth': open_at_depth,
        'error': None,
    }

def analyze_queries_up_to_depth(
    queries: List[Term],
    max_depth: int,
    max_atoms: int,
    rules: List[Rule],
    facts: set,
    index_manager=None,
    is_train: bool = False,
) -> AnalysisResult:
    """
    Analyzes queries at each depth, providing detailed stats on fully proven queries
    and a success ratio distribution for all provable queries. Prints results in real-time.
    """
    start = time.time()
    total = len(queries)
    depths = list(range(1, max_depth + 1))

    final_query_results: List[QueryResult] = [{} for _ in queries]
    depth_first_proven = {}

    per_depth_stats: Dict[int, PerDepthStats] = {
        d: {
            'queries_without_proof': 0,
            'queries_with_proof': 0,
            'fully_proven_count': 0,
            'fully_proven_avg_paths': 0.0,
            'successful_path_ratio_bins': [0] * 5,
        } for d in depths
    }

    # Print the new, wider header before starting the analysis loop
    print(f"\nAnalysis Summary (Total Queries: {total}, Max Atoms: {max_atoms})")
    print("─" * 125)
    header = (f"{'Depth':<7} | {'i) Without Proof':<18} | {'ii) With Proof':<16} | "
              f"{'iii) Fully Proven (Count/Avg Paths)':<38} | {'iv) Success Ratio Bins [0-20%,...,80-100%,100%]':<40}")
    print(header)
    print("─" * 125)

    for d in depths:
        # Reset counters for the current depth
        queries_with_proof_at_d = 0
        fully_proven_count_at_d = 0
        fully_proven_total_paths_at_d = 0
        ratio_bins = [0, 0, 0, 0, 0, 0] # Bins for [0-20, 20-40, 40-60, 60-80, 80-100, 100]

        for i, query in enumerate(queries):
            counts = count_paths_at_depth(
                [query], d, max_atoms, rules, facts, index_manager, is_train
            )
            if counts['error']: continue

            # Analyze queries with at least one proof path
            if counts['proven'] > 0:
                queries_with_proof_at_d += 1

                # 1. Calculate success ratio and place in the correct bin
                total_paths = counts['proven'] + counts['failed'] + counts['truncated'] + counts['open_at_depth']
                if total_paths > 0:
                    success_ratio = counts['proven'] / total_paths
                    if 0 < success_ratio <= 0.2: ratio_bins[0] += 1
                    elif success_ratio <= 0.4: ratio_bins[1] += 1
                    elif success_ratio <= 0.6: ratio_bins[2] += 1
                    elif success_ratio <= 0.8: ratio_bins[3] += 1
                    elif success_ratio < 1.0: ratio_bins[4] += 1
                    else: ratio_bins[5] += 1 # For ratios 1.0

                # 2. Check if it's a "fully proven" query and gather data for the average
                is_fully_proven = (counts['failed'] == 0 and counts['truncated'] == 0 and counts['open_at_depth'] == 0)
                if is_fully_proven:
                    fully_proven_count_at_d += 1
                    fully_proven_total_paths_at_d += counts['proven']
            
            # Update final results (only needs to be done once at the end)
            if d == max_depth:
                if i not in depth_first_proven and counts['proven'] > 0:
                    depth_first_proven[i] = d
                final_query_results[i] = {
                    'query': str(query),
                    'depth_proven': depth_first_proven.get(i),
                    'counts': counts,
                }

        # Finalize calculations for the current depth
        avg_paths_for_fully_proven = (fully_proven_total_paths_at_d / fully_proven_count_at_d 
                                      if fully_proven_count_at_d > 0 else 0)

        # Store all stats for the depth
        per_depth_stats[d]['queries_with_proof'] = queries_with_proof_at_d
        per_depth_stats[d]['queries_without_proof'] = total - queries_with_proof_at_d
        per_depth_stats[d]['fully_proven_count'] = fully_proven_count_at_d
        per_depth_stats[d]['fully_proven_avg_paths'] = avg_paths_for_fully_proven
        per_depth_stats[d]['successful_path_ratio_bins'] = ratio_bins

        # Print the results row for the completed depth
        stats = per_depth_stats[d]
        fully_proven_str = f"{stats['fully_proven_count']} (avg paths: {stats['fully_proven_avg_paths']:.2f})"
        ratio_bins_str = str(stats['successful_path_ratio_bins'])
        row = (f"{d:<7} | {stats['queries_without_proof']:<18} | {stats['queries_with_proof']:<16} | "
               f"{fully_proven_str:<38} | {ratio_bins_str:<40}")
        print(row)

    duration = time.time() - start

    return {
        'max_depth': max_depth,
        'max_atoms': max_atoms,
        'total_queries': total,
        'per_depth_stats': per_depth_stats,
        'query_results': final_query_results,
        'duration': duration,
    }

def load_queries(dataset_name: str, set_file: str, data_path: str, root_dir: str) -> Tuple[List[Term], List[Rule], List[Term], IndexManager]:
    """Load queries, rules, and facts from the specified dataset."""
    
    data_handler = DataHandler(
        dataset_name=dataset_name,
        base_path=data_path,
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="train.txt",
        corruption_mode='dynamic'
    )

    rules = data_handler.rules
    facts_set = set(data_handler.facts)

    index_manager = IndexManager(
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
    index_manager.build_fact_index(data_handler.facts)

    # Load queries based on set_file
    query_map = {
        'train': data_handler.train_queries,
        'valid': data_handler.valid_queries,
        'test': data_handler.test_queries
    }
    queries = query_map.get(set_file, [])
    
    print(f"Loaded {len(queries)} queries from {set_file}.txt")
    return queries, rules, facts_set, index_manager


# ========================
# ==== Main Execution ====
# ========================
if __name__ == "__main__":

    TARGET_DEPTH = 4
    MAX_ATOMS_PER_STATE = 5

    dataset_name = 'family'
    data_path = "./data/"
    root_dir = data_path + dataset_name + '/'

    for set_file in ['test']:

        queries, rules, facts_set, index_manager = load_queries(
            dataset_name, set_file, data_path, root_dir
        )

        is_train_data = 'train' in set_file

        # The function now handles all calculations and printing internally
        analysis = analyze_queries_up_to_depth(
            queries=queries,
            max_depth=TARGET_DEPTH,
            max_atoms=MAX_ATOMS_PER_STATE,
            rules=rules,
            facts=facts_set,
            index_manager=index_manager,
            is_train=is_train_data
        )

        # Print the footer to close the table
        print("─" * 125)
        print(f"Analysis complete. Total execution time: {analysis['duration']:.2f} seconds")