from typing import List, Dict, Tuple, TypedDict
from utils import Term, Rule, apply_substitution, is_variable  
from dataset import get_rules_from_file
import os
import time

from typing import List, Dict, Set, Tuple
from collections import deque
from utils import Term, Rule, apply_substitution
from dataset import get_rules_from_file, get_atom_from_string
from python_unification import get_next_unification_python


class PathCounts(TypedDict):
    """Structure to hold the counts for a single query."""
    proven: int
    failed: int
    truncated: int # Added count for truncated paths
    open_at_depth: int
    error: str | None


def count_proof_paths_at_depth(
    initial_state: List[Term],
    target_depth: int,
    max_atoms: int, # New parameter: max number of atoms allowed in a state
    rules: List[Rule],
    facts: List[Term],
    verbose: bool = False
) -> PathCounts:
    """
    Counts successful, failed, truncated, and open paths at a specific target depth.
    Truncates paths where the state (list of goals) exceeds max_atoms.
    "Open" means paths active entering the final depth step that didn't resolve during it.

    Args:
        initial_state: The initial query state as a list of Term objects.
        target_depth: The exact depth to explore to (integer >= 0).
        max_atoms: Max number of atoms/goals allowed in a state list before truncation.
        rules: List of Rule objects.
        facts: List of Term objects representing facts.
        verbose: If True, prints detailed step information.

    Returns:
        A PathCounts dictionary.
    """
    if verbose: print(f"\n--- Counting paths for Query {initial_state} at Depth {target_depth} (Max Atoms: {max_atoms}) ---")

    # --- Input validation ---
    if not initial_state:
        if verbose: print("Error: Initial state is empty.")
        return {'proven': 0, 'failed': 0, 'truncated': 0, 'open_at_depth': 0, 'error': 'error_empty_query'}
    if target_depth < 0:
         if verbose: print("Error: Target depth cannot be negative.")
         return {'proven': 0, 'failed': 0, 'truncated': 0, 'open_at_depth': 0, 'error': 'error_negative_depth'}
    if max_atoms <= 0:
         if verbose: print("Error: max_atoms must be positive.")
         return {'proven': 0, 'failed': 0, 'truncated': 0, 'open_at_depth': 0, 'error': 'error_invalid_max_atoms'}
    # ---

    # Initialize counts
    proven_paths_count = 0
    failed_paths_count = 0
    truncated_paths_count = 0 # Initialize truncation counter
    open_paths_at_target_depth = 0

    current_states = [initial_state]
    # Check initial state length
    if len(initial_state) > max_atoms:
        if verbose: print(f"Initial state exceeds atom limit ({max_atoms}). Truncated immediately.")
        truncated_paths_count = 1
        current_states = [] # No states to process

    # Breadth-first exploration
    for depth in range(target_depth):
        if verbose: print(f"\nProcessing Depth {depth} (Target: {target_depth}). Current active states: {len(current_states)}")
        if not current_states:
            if verbose: print("No more active states.")
            break

        next_generation_states = []
        resolved_in_this_step = 0
        num_entering_this_step = len(current_states)
        is_final_depth_step = (depth == target_depth - 1)

        if verbose and is_final_depth_step: print(f"  *Final depth step*: {num_entering_this_step} paths entering.")
        if verbose: print(f"States at start of Depth {depth}: {current_states}")

        for i, current_state in enumerate(current_states):
            if verbose: print(f"  Expanding state {i+1}/{num_entering_this_step}: {current_state}")

            branch_next_states = get_next_unification_python(current_state, facts, rules, verbose=verbose > 1)
            if verbose: print(f"    -> Next states for branch: {branch_next_states}")

            # --- Check if this entering state gets resolved/truncated in this step ---
            branch_resolved_here = False
            branch_failed_here = False
            branch_proven_here = False
            branch_truncated_here = False # Did *any* sub-path truncate?
            has_continuing_sub_paths = False

            if not branch_next_states: # Implicit failure
                if verbose: print(f"    -> Branch failed (no next states).")
                failed_paths_count += 1
                branch_resolved_here = True
                branch_failed_here = True
            else:
                for sub_state in branch_next_states:
                    if not sub_state or sub_state[0].predicate == 'False':
                         if verbose: print(f"    -> Sub-path failed ('False'/empty).")
                         failed_paths_count += 1
                         branch_failed_here = True
                    elif sub_state[0].predicate == 'True':
                         if verbose: print(f"    -> Sub-path proven ('True').")
                         proven_paths_count += 1
                         branch_proven_here = True
                         branch_resolved_here = True # Proof resolves the parent branch
                    elif len(sub_state) > max_atoms: # *** ATOM COUNT CHECK ***
                         if verbose: print(f"    -> Sub-path truncated (>{max_atoms} atoms): {sub_state}")
                         truncated_paths_count += 1 # Increment global truncated count
                         branch_truncated_here = True # Mark truncation occurred in this branch
                         # Do NOT add to next_generation_states
                    else:
                         # Sub-path continues and is within limits
                         if verbose: print(f"    -> Sub-path continues: {sub_state}")
                         next_generation_states.append(sub_state)
                         has_continuing_sub_paths = True

                # Determine if the *parent* branch is fully resolved *if no proof was found*
                if not branch_proven_here:
                    if not has_continuing_sub_paths:
                        # If ALL sub-paths ended (failed or truncated) and none proved true,
                        # then the parent branch is resolved.
                        branch_resolved_here = True
                        # Note: We already counted the individual failures/truncations above.
                        if verbose: print(f"    -> Branch resolved (all sub-paths failed or truncated).")
                    # Else (if it has continuing paths), it's not resolved yet.

            # Increment count of parent branches resolved in this step
            if branch_resolved_here:
                 resolved_in_this_step += 1
                 if verbose: print(f"    -> Branch resolved this step. Total resolved this step: {resolved_in_this_step}")
            elif verbose:
                 print(f"    -> Branch continues (not resolved this step).")


        current_states = next_generation_states

        # Calculate Open Paths after the final depth step
        if is_final_depth_step:
            # Open paths are those entering the step minus those resolved (proven/failed) during the step.
            # Truncated paths are counted separately and don't reduce the 'open' count by this definition.
            open_paths_at_target_depth = num_entering_this_step - resolved_in_this_step
            if verbose:
                print(f"  *Calculation at end of final step*:")
                print(f"    Paths entering final step : {num_entering_this_step}")
                print(f"    Paths resolved (proven/failed) in final step: {resolved_in_this_step}")
                print(f"    => Open paths at depth {target_depth}: {open_paths_at_target_depth}")

        if verbose: print(f"End of Depth {depth}. Cumulative Proven: {proven_paths_count}, Failed: {failed_paths_count}, Truncated: {truncated_paths_count}. Active states for next depth: {len(current_states)}")

    # After the loop
    if verbose:
        print(f"\n--- Final Counts for Query {initial_state} at Depth {target_depth} ---")
        print(f"Paths Proven by depth {target_depth}: {proven_paths_count}")
        print(f"Paths Failed by depth {target_depth}: {failed_paths_count}")
        print(f"Paths Truncated by depth {target_depth}: {truncated_paths_count}") # Report truncated
        print(f"Paths Open at depth {target_depth}: {open_paths_at_target_depth}")

    return {
        'proven': proven_paths_count,
        'failed': failed_paths_count,
        'truncated': truncated_paths_count, # Include in return dict
        'open_at_depth': open_paths_at_target_depth,
        'error': None
    }


def analyze_queries_at_depth(
    queries: list[Term],
    target_depth: int,
    max_atoms: int, # Added parameter
    rules: list[Rule],
    facts: list[Term],
    is_train_data: bool = False,
    verbose_query: bool = False,
    verbose_engine: bool = False
) -> Dict | None:
    """
    Analyzes queries at a specific depth, counting proven, failed, truncated, and open paths.

    Args:
        queries: A list of Term objects representing the queries.
        target_depth: The specific depth to analyze at.
        max_atoms: Max number of atoms/goals allowed in a state list.
        rules: List of Rule objects.
        facts: List of Term objects representing facts.
        is_train_data: If True, filter out the query itself from facts.
        verbose_query: Verbosity for query path counting.
        verbose_engine: Verbosity for the unification engine.

    Returns:
        A dictionary containing aggregated counts and per-query results.
    """
    start_time = time.time()
    num_queries = len(queries)
    if num_queries == 0: print("No queries provided."); return None

    print(f"\nAnalyzing {num_queries} queries at depth {target_depth} (Max Atoms: {max_atoms})...")

    # Initialize aggregate results
    total_proven = 0
    total_failed = 0
    total_truncated = 0 # Added aggregate counter
    total_open = 0
    total_errors = 0
    query_details = []

    for i, goal_term in enumerate(queries):
        print(f"\rProcessing query {i+1}/{num_queries}: {goal_term}...", end='')
        current_facts = [fact for fact in facts if fact != goal_term] if is_train_data else facts

        try:
            counts = count_proof_paths_at_depth(
                initial_state=[goal_term],
                target_depth=target_depth,
                max_atoms=max_atoms, # Pass parameter
                rules=rules,
                facts=current_facts,
                verbose=verbose_query
            )

            if counts['error']:
                total_errors += 1
                print(f"\nERROR for query {goal_term}: {counts['error']}")
            else:
                total_proven += counts['proven']
                total_failed += counts['failed']
                total_truncated += counts['truncated'] # Aggregate truncated count
                total_open += counts['open_at_depth']

            query_details.append({ 'query': str(goal_term), **counts })

        except Exception as e:
             total_errors += 1
             print(f"\nCRITICAL ERROR during processing of query {goal_term}: {e}")
             query_details.append({
                 'query': str(goal_term),
                 'proven': 0, 'failed': 0, 'truncated': 0, 'open_at_depth': 0,
                 'error': f'critical_exception: {e}'
             })

    print(f"\nFinished processing {num_queries} queries.")
    end_time = time.time()

    # Final Summary Print
    print(f"\n--- Analysis Summary (Depth {target_depth}, Max Atoms: {max_atoms}) ---")
    print(f"Total queries processed: {num_queries}")
    print(f"Target Depth: {target_depth}")
    print(f"Max Atoms per State: {max_atoms}")
    print("-" * 20)
    print(f"Total Proven Paths (cumulative <= depth {target_depth}): {total_proven}")
    print(f"Total Failed Paths (cumulative <= depth {target_depth}): {total_failed}")
    print(f"Total Truncated Paths (cumulative <= depth {target_depth}): {total_truncated}") # Report total truncated
    print(f"Total Open Paths (unresolved *at* depth {target_depth}): {total_open}")
    print("-" * 20)
    print(f"Queries with errors: {total_errors}")
    print(f"Execution time: {end_time - start_time:.2f} seconds")

    return {
        'target_depth': target_depth,
        'max_atoms': max_atoms,
        'total_queries': num_queries,
        'total_proven_paths': total_proven,
        'total_failed_paths': total_failed,
        'total_truncated_paths': total_truncated, # Include in dict
        'total_open_paths': total_open,
        'total_errors': total_errors,
        'query_details': query_details
    }


# ========================
# ===== Main Execution =====
# ========================
if __name__ == "__main__":

    TARGET_DEPTH = 4  # Set the specific depth
    MAX_ATOMS_PER_STATE = 3 # <<< Set the max number of goals allowed in a state

    # --- (Dataset configuration, File loading, Query loading remains the same) ---
    # ... (Use the __main__ block from the previous response, loading files or dummies) ...
    # 1. Configure dataset and files
    dataset = 'kinship_family'
    prolog_kb_file = './data/kinship_family/kinship_family_no_tables.pl'
    queries_file = './data/kinship_family/train.txt'
    is_train_data = True

    # Load KB (or dummy)
    facts, rules = [], []
    if not os.path.exists(prolog_kb_file):
        print(f"WARNING: KB file not found. Using dummy KB."); facts, rules = get_rules_from_file("dummy_kb")
    else:
        try: facts, rules = get_rules_from_file(prolog_kb_file); print(f"Loaded {len(rules)} rules, {len(facts)} facts.")
        except Exception as e: print(f"ERROR loading KB: {e}"); exit()

    # Load Queries (or dummy)
    queries_terms = []
    if not os.path.exists(queries_file):
        print(f"WARNING: Queries file not found. Using dummy queries.")
        queries_terms = [get_atom_from_string(q) for q in ["grandparent(a,c)", "parent(a,b)", "parent(c,a)"]]
    else:
        try:
            with open(queries_file, 'r') as f: query_strings = [ln.strip() for ln in f if ln.strip()]
            queries_terms = [get_atom_from_string(q_str) for q_str in query_strings]; print(f"Loaded {len(queries_terms)} queries.")
        except Exception as e: print(f"Error loading queries: {e}"); exit()

    # Optional limit
    queries_terms = queries_terms[:5]
    if not queries_terms: print("ERROR: No queries. Exiting."); exit()

    # 4. Run the analysis
    analysis_results = analyze_queries_at_depth(
        queries=queries_terms,
        target_depth=TARGET_DEPTH,
        max_atoms=MAX_ATOMS_PER_STATE, # Pass the new limit
        rules=rules,
        facts=facts,
        is_train_data=is_train_data,
        verbose_query=False, # Keep False for cleaner summary unless debugging
        verbose_engine=False
    )

    # 5. Print detailed results (optional)
    if analysis_results and analysis_results.get('query_details'):
        print(f"\n--- Detailed Results per Query (Depth {TARGET_DEPTH}, Max Atoms: {MAX_ATOMS_PER_STATE}) ---")
        details_to_show = analysis_results['query_details'] # [:20]
        for detail in details_to_show:
            if detail['error']:
                 print(f"Query: {detail['query']:<30} -> ERROR: {detail['error']}")
            else:
                 # Display all counts
                 print(f"Query: {detail['query']:<30} -> Proven: {detail['proven']:<3} Failed: {detail['failed']:<3} Truncated: {detail['truncated']:<3} Open: {detail['open_at_depth']:<3}")
        # if len(analysis_results['query_details']) > len(details_to_show): print(f"... (showing first {len(details_to_show)} details)")
    elif analysis_results:
         print("\nNo detailed query results available.")
    else:
         print("\nAnalysis did not produce results.")