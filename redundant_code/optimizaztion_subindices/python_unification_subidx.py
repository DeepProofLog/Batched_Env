# --- Assume existing imports and Class definitions for Term, Rule ---
# (Imports: re, typing, torch, TensorDict, datetime, os, numpy, ast, defaultdict, deque)
# (Classes: Term, Rule - unchanged from original)
# (Helper: is_variable - unchanged from original)

# --- Modified/New Functions ---

import re
from typing import Dict, Union, List, Any, Tuple, Iterable, Optional, Set
import torch
from tensordict import TensorDict, TensorDictBase
import datetime
import os
import numpy as np
import ast
from collections import defaultdict, deque
from optimizaztion_subindices.utils_v1 import Term, Rule, is_variable
from index_manager_v2 import IndexManager



SubstitutionIdDict = Dict[int, int] # Type alias for clarity {VarID: TermID / ConstantID}

def unify_terms_ids(term1_ids: Tuple[int, ...],
                    term2_ids: Tuple[int, ...],
                    index_manager: IndexManager) -> Optional[SubstitutionIdDict]:
    """Unifies two terms represented by integer IDs. (No changes needed)"""
    # Check predicate and arity
    if term1_ids[0] != term2_ids[0] or len(term1_ids) != len(term2_ids):
        # Special case: allow unifying True() with True() etc. even if arity is 0
        if term1_ids[0] in (index_manager.true_pred_id, index_manager.false_pred_id) and term1_ids == term2_ids:
             pass # Allow matching True/False as singletons
        else:
            return None

    substitutions: SubstitutionIdDict = {}
    queue = deque(zip(term1_ids[1:], term2_ids[1:])) # Start with argument pairs

    while queue:
        arg1_id, arg2_id = queue.popleft()

        # Resolve substitutions until constants or distinct variables are reached
        visited1 = {arg1_id}
        while arg1_id in substitutions:
            arg1_id = substitutions[arg1_id]
            if arg1_id in visited1: return None # Occurs check: Simple cycle detected X=X...
            visited1.add(arg1_id)

        visited2 = {arg2_id}
        while arg2_id in substitutions:
            arg2_id = substitutions[arg2_id]
            if arg2_id in visited2: return None # Occurs check: Simple cycle detected Y=Y...
            visited2.add(arg2_id)


        if arg1_id == arg2_id:
            continue

        is_var1 = index_manager.is_variable_id(arg1_id)
        is_var2 = index_manager.is_variable_id(arg2_id)

        if is_var1:
            # Basic occurs check: Check if var1 is contained within the structure of arg2_id
            # This requires resolving arg2_id if it's a variable. The transitive application below helps.
            # A full occurs check for functional terms (e.g., X = f(X)) is more complex with just IDs.
             if arg2_id in substitutions and substitutions[arg2_id] == arg1_id: return None # Avoid direct X=Y, Y=X cycle before setting
             substitutions[arg1_id] = arg2_id
        elif is_var2:
             if arg1_id in substitutions and substitutions[arg1_id] == arg2_id: return None # Avoid direct X=Y, Y=X cycle before setting
             substitutions[arg2_id] = arg1_id
        elif arg1_id != arg2_id: # Both are constants (or resolved to constants), must be equal
            return None # Constants mismatch

    # Apply substitutions transitively to ensure MGU points to final values/vars
    changed = True
    while changed:
        changed = False
        for var_id, value_id in list(substitutions.items()):
            # Resolve the value further
            visited = {value_id}
            while value_id in substitutions:
                new_value_id = substitutions[value_id]
                if new_value_id == var_id: return None # Occurs check failure (X=...->X)
                if new_value_id in visited : # Cycle detected in value resolution
                    # This can happen with X=Y, Y=Z, Z=X. Let's break cycle arbitrarily or fail.
                    # Failing might be safer, but depends on desired behavior.
                    # For now, let's just stop resolving this chain to avoid infinite loop.
                    # A more robust solution might track full dependency graph.
                    break # Stop resolving this chain if cycle detected
                visited.add(new_value_id)
                value_id = new_value_id

            # Update if the resolved value is different from the current mapping
            if substitutions[var_id] != value_id:
                 substitutions[var_id] = value_id
                 changed = True # Signal that another pass is needed

    return substitutions


def apply_substitution_ids(term_ids: Tuple[int, ...],
                           substitutions: SubstitutionIdDict,
                           index_manager: IndexManager,
                           ) -> Tuple[int, ...]:
    """Applies ID substitutions to a term ID tuple. (No changes needed)"""
    pred_id = term_ids[0]
    new_args = []
    for arg_id in term_ids[1:]:
        original_arg_id = arg_id # Keep track for cycle detection
        visited = {arg_id}
        # Follow substitution chain until non-variable or variable not in subs
        while index_manager.is_variable_id(arg_id) and arg_id in substitutions:
            arg_id = substitutions[arg_id]
            if arg_id in visited : # Avoid infinite loops X=Y, Y=X etc.
                 # print(f"Warning: Cycle detected during substitution application for original ID {original_arg_id}. Breaking loop.")
                 break # Return the ID where cycle detected
            visited.add(arg_id)
        new_args.append(arg_id)
    return (pred_id,) + tuple(new_args)

def unify_with_facts_ids(query_ids: Tuple[int, ...],
                         facts_ids_set: Set[Tuple[int, ...]],
                         index_manager: IndexManager) -> List[SubstitutionIdDict]:
    """Attempts unification with facts represented by integer IDs. (No changes needed)"""
    substitutions_list = []
    query_predicate = query_ids[0]

    # Optimization: If query is ground (no variables), check direct existence
    query_is_ground = not any(index_manager.is_variable_id(arg) for arg in query_ids[1:])
    if query_is_ground:
        if query_ids in facts_ids_set:
            # Signal ground fact match. Return a non-empty list, content can be simple.
            # Using the True mapping helps distinguish from variable bindings.
            return [{index_manager.true_pred_id: index_manager.true_pred_id}]
        else:
            return [] # No match for ground query

    # If query has variables, try unification
    for fact_ids in facts_ids_set:
        # Quick check: predicate must match
        # Facts must be ground (no variables)
        fact_is_ground = not any(index_manager.is_variable_id(arg) for arg in fact_ids[1:])
        if not fact_is_ground:
            # print(f"Warning: Skipping non-ground fact during unification: {index_manager.ids_to_term(fact_ids)}")
            continue # Standard Prolog assumes facts are ground

        if fact_ids[0] == query_predicate and len(fact_ids) == len(query_ids):
            # Unify fact (term1) with query (term2)
            # Subs will map query variables to fact constants
            subs = unify_terms_ids(fact_ids, query_ids, index_manager)
            if subs is not None:
                substitutions_list.append(subs)

    return substitutions_list


def unify_with_rules_ids(query_ids: Tuple[int, ...],
                         rules_as_ids: List[Tuple[Tuple[int, ...], List[Tuple[int, ...]]]],
                         index_manager: IndexManager
                         ) -> List[Tuple[List[Tuple[int, ...]], SubstitutionIdDict]]:
    """
    Unifies a query ID tuple with rule heads, returning substituted bodies. (No changes needed)
    Rule variables use their global IDs; query variables use temporary IDs.
    """
    results = []
    query_predicate = query_ids[0]

    for rule_head_ids, rule_body_ids in rules_as_ids:
        # Check predicate match and arity
        if rule_head_ids[0] == query_predicate and len(rule_head_ids) == len(query_ids):

            # --- Standardize Apart Rule Variables ---
            # Create temporary IDs for rule variables to avoid clashes with query/state vars.
            # This is crucial for correct recursion.
            rule_rename_map: SubstitutionIdDict = {}
            next_temp_var_id_for_rule = index_manager.temp_variable_start_idx # Use manager's global counter conceptually
            # Find all unique variables in the rule (head and body)
            rule_vars = set()
            for arg_id in rule_head_ids[1:]:
                if index_manager.is_global_variable_id(arg_id): rule_vars.add(arg_id)
            for body_term_ids in rule_body_ids:
                for arg_id in body_term_ids[1:]:
                     if index_manager.is_global_variable_id(arg_id): rule_vars.add(arg_id)

            # Need a way to advance the global temp counter. Pass it in and return it?
            # For now, assume we just need *distinct* IDs. Let's use a high offset specific to this call? Risky.
            # Let's try using the main temp counter logic by calling term_to_ids with a temporary map.

            # Simplified approach for now: Assume unification handles scope correctly IF
            # the query variables are already temporary IDs. Let's try *without* explicit renaming here,
            # relying on the fact that query vars passed to unify_terms_ids will have temp IDs >= temp_variable_start_idx
            # and rule vars will have global IDs < temp_variable_start_idx.

            # unify_terms_ids(rule_head, query) -> subs map rule vars to query vars/constants
            subs = unify_terms_ids(rule_head_ids, query_ids, index_manager)

            if subs is not None:
                # Apply the resulting substitutions to the rule body IDs
                # The subs map global rule variable IDs to the appropriate temp state IDs or constants from the query.
                new_body = [apply_substitution_ids(term_ids, subs, index_manager) for term_ids in rule_body_ids]
                results.append((new_body, subs)) # Return the derived body and the substitution map

    # No need to return next_temp_var_id as no new IDs were permanently created here
    return results


# --- Modified get_next_unification_ids ---

def get_next_unification_ids(state_ids: List[Tuple[int, ...]],
                             facts_ids_set: Set[Tuple[int, ...]],
                             rules_as_ids: List[Tuple[Tuple[int, ...], List[Tuple[int, ...]]]],
                             index_manager: IndexManager,
                             next_temp_var_id: int, # Starting ID for NEW temporary vars in this step
                             verbose: int = 0
                             ) -> Tuple[List[List[Tuple[int, ...]]], int]: # Return next states and updated next_temp_var_id
    """
    Core unification logic operating on integer ID tuples.
    Mimics the original python logic:
    1. Tries rule unification on the first goal.
    2. For each state derived from a rule, tries fact unification on ITS first goal.
    3. If no rules matched, tries fact unification on the ORIGINAL first goal.
    4. Simplifies resulting states by removing known ground facts.
    """
    true_id = index_manager.true_pred_id
    false_id = index_manager.false_pred_id

    # --- Initial Checks ---
    if not state_ids: return [[(true_id,)]], next_temp_var_id # Empty state is success
    if any(term_ids == (false_id,) for term_ids in state_ids): return [[(false_id,)]], next_temp_var_id
    current_state_ids = [term_ids for term_ids in state_ids if term_ids != (true_id,)]
    if not current_state_ids: return [[(true_id,)]], next_temp_var_id

    # --- Variable Renaming (State Vars to New Temp Vars) ---
    # Crucial for maintaining correct variable scopes across steps/branches
    state_rename_map: SubstitutionIdDict = {}
    original_to_new_temp_map : Dict[int, int] = {}
    current_vars = set()
    for term_ids in current_state_ids:
        for arg_id in term_ids[1:]:
            if index_manager.is_variable_id(arg_id):
                current_vars.add(arg_id)

    local_next_temp_var_id = next_temp_var_id
    for var_id in sorted(list(current_vars)):
        new_temp_id = local_next_temp_var_id
        state_rename_map[var_id] = new_temp_id
        original_to_new_temp_map[var_id] = new_temp_id
        # Update reverse map for debugging
        original_name = index_manager.variable_idx2str.get(var_id, f"Var{var_id}")
        new_name = f"{original_name}_st{new_temp_id}"
        if new_temp_id not in index_manager.variable_idx2str:
            index_manager.variable_idx2str[new_temp_id] = new_name
        elif index_manager.variable_idx2str[new_temp_id] != new_name and verbose >=3:
            pass # Allow reuse, name might change slightly
        local_next_temp_var_id += 1

    renamed_state_ids = [apply_substitution_ids(term_ids, state_rename_map, index_manager) for term_ids in current_state_ids]
    updated_next_temp_var_id = local_next_temp_var_id # Return the updated counter

    # --- Goal Selection ---
    if not renamed_state_ids:
       print("Warning: No state IDs after initial checks and renaming.")
       return [[(true_id,)]], updated_next_temp_var_id

    query_ids, *remaining_state_ids = renamed_state_ids
    potential_next_states_before_simplification = [] # Collect results before final simplification

    if verbose >= 1: print('\n**********')
    if verbose >= 1: debug_print_state_ids([query_ids], index_manager, "Processing Query IDs (Renamed)")
    if verbose >= 1: debug_print_state_ids(remaining_state_ids, index_manager, "Remaining State IDs (Renamed)")
    if verbose >= 1: print('**********\n')

    # --- Step 1: Unification ONLY with Rules ---
    if verbose >= 1: print('Attempting unification with rules...')
    # rule_results contains tuples: (substituted_body_ids, rule_substitutions)
    rule_results = unify_with_rules_ids(query_ids, rules_as_ids, index_manager)

    # Store intermediate states derived purely from rules
    intermediate_states_from_rules = []
    for body_ids, rule_subs in rule_results:
        # Apply the substitution found during rule unification to the remaining goals
        new_remaining_ids = [apply_substitution_ids(term_ids, rule_subs, index_manager) for term_ids in remaining_state_ids]
        # The new state is the substituted rule body prepended to the substituted remaining goals
        rule_derived_state_ids = body_ids + new_remaining_ids
        intermediate_states_from_rules.append(rule_derived_state_ids)
        if verbose >= 2: debug_print_state_ids(rule_derived_state_ids, index_manager, "Intermediate State (from Rule)")

    # --- Step 2: Apply Fact Unification to First Goal of Intermediate States ---
    if verbose >= 1: print('\nAttempting fact unification on first goal of intermediate states...')
    processed_intermediate_states = False # Flag to track if we processed any rule results

    for state_from_rule in intermediate_states_from_rules:
        processed_intermediate_states = True
        if verbose >= 2: debug_print_state_ids(state_from_rule, index_manager, "  Processing intermediate state")

        if not state_from_rule: # Rule application led directly to success
            if verbose >= 2: print("  Rule led to empty state -> Potential [True]")
            potential_next_states_before_simplification.append([(true_id,)])
            continue

        first_goal_ids = state_from_rule[0]
        rest_of_goals_ids = state_from_rule[1:]

        if verbose >= 1: debug_print_state_ids([first_goal_ids], index_manager, f"    Checking first goal against facts")
        # Attempt to unify the *first goal of the intermediate state* with facts
        fact_substitutions = unify_with_facts_ids(first_goal_ids, facts_ids_set, index_manager)

        fact_match_found_for_goal = False
        if fact_substitutions:
            fact_match_found_for_goal = True
            if verbose >= 1: debug_print_state_ids([first_goal_ids], index_manager, f"    Fact match found for")
            for fact_subs in fact_substitutions:
                if verbose >= 2: print(f"      Applying fact substitution: {fact_subs}")
                current_resulting_state_list_ids = []
                if fact_subs.get(true_id) == true_id: # Ground fact match for first_goal
                    current_resulting_state_list_ids = rest_of_goals_ids # Remaining goals stay as they are
                else: # Fact match with variable binding for first_goal
                    # Apply fact substitution to the *rest* of the goals from the intermediate state
                    current_resulting_state_list_ids = [apply_substitution_ids(term_ids, fact_subs, index_manager) for term_ids in rest_of_goals_ids]

                potential_next_states_before_simplification.append(current_resulting_state_list_ids)
                if verbose >= 2: debug_print_state_ids(current_resulting_state_list_ids, index_manager, "    State after Fact Unification")
        else:
             if verbose >= 1: debug_print_state_ids([first_goal_ids], index_manager, f"    No fact match for")


        # If the first goal of the intermediate state did *not* unify with any fact,
        # then the intermediate state itself (derived from the rule) is a valid next step.
        if not fact_match_found_for_goal:
            potential_next_states_before_simplification.append(state_from_rule)
            if verbose >= 1: debug_print_state_ids(state_from_rule, index_manager, "    Keeping rule-derived state (no fact match for first goal)")


    # --- Fallback: Check Original Query Against Facts (if no rules matched) ---
    if not processed_intermediate_states: # This means rule_results was empty
        if verbose >= 1: print(f"\nNo states derived from rules. Checking original query against facts.")
        direct_fact_subs = unify_with_facts_ids(query_ids, facts_ids_set, index_manager)
        if direct_fact_subs:
            if verbose >= 1: debug_print_state_ids([query_ids], index_manager, "Original query matched facts. Applying to remaining state:")
            if verbose >= 1: debug_print_state_ids(remaining_state_ids, index_manager, "Remaining state was:")
            for fact_subs in direct_fact_subs:
                if verbose >=2: print(f"  Applying fact substitution: {fact_subs}")
                current_resulting_state_list_ids = []
                if fact_subs.get(true_id) == true_id: # Ground query matched ground fact
                    current_resulting_state_list_ids = remaining_state_ids # Keep original remaining state
                else: # Query (with vars) matched fact, apply bindings to remaining goals
                    current_resulting_state_list_ids = [apply_substitution_ids(term_ids, fact_subs, index_manager) for term_ids in remaining_state_ids]

                potential_next_states_before_simplification.append(current_resulting_state_list_ids)
                if verbose >= 1: debug_print_state_ids(current_resulting_state_list_ids, index_manager, "  Resulting State (Direct Fact Match)")
        else:
            if verbose >= 1: debug_print_state_ids([query_ids], index_manager, "Original query also did not match any facts.")
            # No rules matched, no facts matched -> this path fails
            pass # Failure state handled in final processing


    # --- Final Post-processing: Simplify States & Check for Proof ---
    final_next_states_ids = []
    processed_state_tuples = set()

    if verbose >= 1: print("\n--- Post-processing: Simplifying states and checking for proof ---")

    # Handle case where no potential states were generated at all (query failed completely)
    if not potential_next_states_before_simplification:
         if verbose >= 1: print("  No potential next states generated. Returning [False]")
         return [[(false_id,)]], updated_next_temp_var_id

    for state_list_ids in potential_next_states_before_simplification:
        # Handle states that were already marked as True during processing
        if state_list_ids == [(true_id,)]:
            if verbose >= 1: print("  Found pre-resolved [True] state. Proof found!")
            return [[(true_id,)]], updated_next_temp_var_id # Immediate return

        # Check for immediate success (empty list of goals)
        if not state_list_ids:
            if verbose >= 1: print("  Found empty goal list. Proof found!")
            return [[(true_id,)]], updated_next_temp_var_id # Immediate return

        # Simplify by removing known ground facts *within the derived state*
        simplified_state_ids = []
        made_change = False
        if verbose >= 3: debug_print_state_ids(state_list_ids, index_manager, "  Simplifying State")
        for term_ids in state_list_ids:
            # Check if term is ground (predicate + constants)
            is_ground = not any(index_manager.is_variable_id(arg) for arg in term_ids[1:])
            if is_ground and term_ids in facts_ids_set:
                made_change = True
                if verbose >= 1: debug_print_state_ids([term_ids], index_manager, f"    Simplifying: Removing known ground fact")
                continue # Skip this term
            else:
                simplified_state_ids.append(term_ids)

        # Check if simplification resulted in empty list (proof found)
        if not simplified_state_ids:
            if verbose >= 1: print("  State simplified to empty. Proof found!")
            if verbose >= 3: debug_print_state_ids(state_list_ids, index_manager, "  Original state was:")
            return [[(true_id,)]], updated_next_temp_var_id # Immediate return

        # Add unique simplified state
        try:
            # Use tuple of sorted tuples for canonical representation for duplicate check
            state_tuple_for_set = tuple(sorted(simplified_state_ids))
            if state_tuple_for_set not in processed_state_tuples:
                final_next_states_ids.append(simplified_state_ids)
                processed_state_tuples.add(state_tuple_for_set)
                if verbose >= 2: debug_print_state_ids(simplified_state_ids, index_manager, f"  Adding unique simplified state{' (post-simplification)' if made_change else ''}")
            # else:
                # if verbose >= 2: debug_print_state_ids(simplified_state_ids, index_manager, "  Skipping duplicate simplified state")
        except TypeError as e:
            print(f"Warning: Could not create comparable tuple for state {simplified_state_ids} - {e}. Adding without duplicate check.")
            if simplified_state_ids not in final_next_states_ids: # Less efficient check
                final_next_states_ids.append(simplified_state_ids)

    # --- Return final states or False ---
    if not final_next_states_ids:
        # This happens if all paths led to failure or were simplified to True() handled above.
        if verbose >= 1: print("  No viable next states remain after processing rules/facts and simplification.")
        return [[(false_id,)]], updated_next_temp_var_id
    else:
        if verbose >= 1: debug_print_states_list(final_next_states_ids, index_manager, "Final Next States (IDs)")
        if verbose >=1 : print('*************************\n')
        return final_next_states_ids, updated_next_temp_var_id


# --- Wrapper Function (Maintains Original Interface) ---

def get_next_unification_python(state: List[Term],
                                facts: List[Term],
                                rules: List[Rule], # Pass original rules here
                                index_manager: IndexManager, # Needs the manager instance
                                verbose: int = 0
                                ) -> List[List[Term]]:
    """
    Wrapper function to maintain the original interface.
    Converts Terms to IDs, calls the ID-based logic, converts results back.
    """
    if verbose >= 1: print(f"\n--- Optimizing Step ---")
    if verbose >= 2: print(f"Input State: {state}")

    # --- Convert Inputs to IDs ---
    # Convert facts to a set of ID tuples (do this once or ensure manager caches it)
    # Assuming facts are ground, they won't introduce new variables needing temp IDs.
    if not index_manager.facts_as_ids_set: # Populate only if not already done
        if verbose >=3 : print("Converting facts to IDs...")
        facts_var_map = index_manager.variable_str2idx.copy() # Use global map, but shouldn't find vars
        facts_next_temp_id = index_manager.temp_variable_start_idx # Should not be incremented
        for fact in facts:
            try:
                # Facts *should* be ground, so don't assign new vars
                fact_ids, _, facts_next_temp_id = index_manager.term_to_ids(
                    fact, facts_var_map, facts_next_temp_id, assign_new_vars=False
                )
                if any(index_manager.is_variable_id(arg) for arg in fact_ids[1:]):
                     print(f"Warning: Fact '{fact}' appears to contain variables. Treating as ground anyway.")
                index_manager.facts_as_ids_set.add(fact_ids)
            except ValueError as e:
                 print(f"Warning: Skipping fact '{fact}' during conversion: {e}")
        if verbose >=3 : print(f"Facts as IDs: {index_manager.facts_as_ids_set}")


    # Convert current state to ID tuples
    # Variables in the state might be new or existing, assign temp IDs as needed.
    # Start temp ID assignment using the manager's current high-water mark.
    state_var_map = index_manager.variable_str2idx.copy() # Start with global for known vars
    initial_temp_id = index_manager.temp_variable_start_idx # Where to start *new* temp IDs from
    try:
        state_ids, final_var_map, next_temp_id_after_state = index_manager.state_to_ids(
            state, state_var_map, initial_temp_id
        )
    except ValueError as e:
         print(f"Error converting initial state {state} to IDs: {e}")
         return [[Term('False', [])]] # Return failure state


    if verbose >= 2: debug_print_state_ids(state_ids, index_manager, "Input State IDs (Initial Conversion)")
    # if verbose >=3 : print(f"Variable map after state conversion: {final_var_map}")
    # if verbose >=3 : print(f"Next temp ID after state conversion: {next_temp_id_after_state}")


    # --- Call the ID-based function ---
    # Pass the pre-converted rules from the index_manager
    # Pass the *next available* temp ID counter from the state conversion
    next_states_ids, final_temp_id_counter = get_next_unification_ids(
        state_ids,
        index_manager.facts_as_ids_set, # Use pre-converted facts
        index_manager.rules_as_ids,    # Use pre-converted rules
        index_manager,
        next_temp_id_after_state,      # Pass the current counter state
        verbose
    )


    # Update the manager's global counter if needed for subsequent steps, although
    # typically each call to get_next_unification_python starts fresh.
    # index_manager.temp_variable_start_idx = final_temp_id_counter # Optional: Update global counter


    # --- Convert Results back to Terms ---
    final_next_states = []
    for next_state_id_list in next_states_ids:
        try:
             final_next_states.append(index_manager.ids_to_state(next_state_id_list))
        except ValueError as e:
              print(f"Error converting result state {next_state_id_list} back to Terms: {e}")
              # Append a False state to indicate error during conversion
              final_next_states.append([Term('False', [])])

    # order alphabetically by predicate name and constant
    final_next_states = tuple(sorted(final_next_states, key=lambda x: repr(x)))


    if verbose >= 1: print(f"Output States: {final_next_states}")
    if verbose >= 1: print(f"--- End Optimizing Step ---")
    return final_next_states


# --- Debugging Functions (No changes needed) ---

def debug_print_state_ids(state_ids: List[Tuple[int, ...]], index_manager: IndexManager, title: str = "State IDs"):
    """Prints a state represented by ID tuples in a readable format."""
    print(f"--- {title} ---")
    if not state_ids:
        print("[] (Empty State)")
        return
    # Handle special states like [(TrueID,)] or [(FalseID,)]
    # Check if it's a single term tuple with only a predicate ID that isn't a constant/var
    if len(state_ids) == 1 and len(state_ids[0]) == 1:
         pred_id = state_ids[0][0]
         if pred_id == index_manager.true_pred_id or pred_id == index_manager.false_pred_id:
              print(f"[{index_manager.ids_to_term(state_ids[0])}]")
              return
         # Could also be a 0-arity predicate like p() if defined
         # Fall through to general case if unsure

    try:
        terms = [index_manager.ids_to_term(ids) for ids in state_ids]
        print(f"{terms}")
    except ValueError as e:
        print(f"Error converting state {state_ids} to terms for printing: {e}")
        print(f"Raw IDs: {state_ids}")


def debug_print_states_list(states_list_ids: List[List[Tuple[int,...]]], index_manager: IndexManager, title: str = "States List IDs"):
    """Prints a list of states represented by ID tuples."""
    print(f"--- {title} ---")
    if not states_list_ids:
        print("[] (No states)")
        return
    for i, state_ids in enumerate(states_list_ids):
          debug_print_state_ids(state_ids, index_manager, f"State {i}")


# --- Example Test in main (Updated expected IDs) ---

if __name__ == "__main__":
    # --- Define Sample Logic Program Elements ---
    constants = {"a", "b", "c"}
    predicates = {"p", "q", "r", "parent", "grandparent"}
    variables = {"X", "Y", "Z"} # Global variables

    # --- Create Facts (Term objects) ---
    facts = [
        Term("parent", ["a", "b"]), # parent(a, b)
        Term("parent", ["b", "c"]), # parent(b, c)
        Term("p", ["a"]),           # p(a)
    ]

    # --- Create Rules (Rule objects) ---
    rule1_head = Term("grandparent", ["X", "Z"])
    rule1_body = [Term("parent", ["X", "Y"]), Term("parent", ["Y", "Z"])]
    rule1 = Rule(rule1_head, rule1_body)

    rule2_head = Term("q", ["X"])
    rule2_body = [Term("p", ["X"])]
    rule2 = Rule(rule2_head, rule2_body)

    rules = [rule1, rule2]

    # --- Initialize IndexManager ---
    manager = IndexManager(
        constants=constants,
        predicates=predicates,
        variables=variables,
        constant_no=len(constants),     # 3
        predicate_no=len(predicates),   # 5
        variable_no=len(variables),     # 3
        rules=rules,                    # Pass rules for pre-conversion
        max_arity=2
    )

    print("--- Index Manager Mappings ---")
    print(f"ID Offset: {manager.id_offset}")
    print(f"True ID: {manager.true_pred_id}")
    print(f"False ID: {manager.false_pred_id}")
    print(f"Constants: {manager.constant_str2idx} (Range: {manager.constant_start_idx}-{manager.constant_end_idx})")
    print(f"Predicates: {manager.predicate_str2idx} (Range: {manager.predicate_start_idx}-{manager.predicate_end_idx})")
    print(f"Global Variables: {manager.variable_str2idx} (Range: {manager.variable_start_idx}-{manager.variable_end_idx})")
    print(f"Temp Variable Start Index: {manager.temp_variable_start_idx}")


    print("\n--- Rules as IDs ---")
    for i, (h, b) in enumerate(manager.rules_as_ids):
        debug_print_state_ids([h], manager, f"Rule {i} Head")
        debug_print_state_ids(b, manager, f"Rule {i} Body")


    # --- Test Case 1: Simple Fact Resolution ---
    print("\n--- Test Case 1: parent(a, b) ---")
    initial_state_1 = [Term("parent", ["a", "b"])]
    next_states_1 = get_next_unification_python(initial_state_1, facts, rules, manager, verbose=1)
    # Expected: [[True()]] because parent(a,b) is a ground fact, state becomes [] -> True()
    print("Resulting Next States:", next_states_1)

    # --- Test Case 2: Simple Query with Variable ---
    print("\n--- Test Case 2: parent(a, X) ---")
    # X is a query variable -> gets temp ID
    initial_state_2 = [Term("parent", ["a", "X"])]
    next_states_2 = get_next_unification_python(initial_state_2, facts, rules, manager, verbose=1)
    # Expected: [[True()]] because parent(a, X) unifies with parent(a,b) -> X=b, remaining state [], -> True()
    print("Resulting Next States:", next_states_2)

    # --- Test Case 3: Rule Application ---
    print("\n--- Test Case 3: q(a) ---")
    initial_state_3 = [Term("q", ["a"])]
    next_states_3 = get_next_unification_python(initial_state_3, facts, rules, manager, verbose=1)
    # Step 1: q(a) unifies with rule q(X) -> {X:a}, new state [p(X)]{X:a} -> [p(a)]
    # Step 2 (Internal to get_next_unification_ids): Simplified state [p(a)] is ground fact -> [] -> True()
    print("Resulting Next States:", next_states_3) # Expected: [[True()]]

    # --- Test Case 4: Multi-step Rule Application ---
    print("\n--- Test Case 4: grandparent(a, C) ---")
    # C is a query variable -> gets temp ID
    initial_state_4 = [Term("grandparent", ["a", "C"])]
    next_states_4_step1 = get_next_unification_python(initial_state_4, facts, rules, manager, verbose=1)
    # Step 1: grandparent(a, C_temp1) unifies with rule head grandparent(X_global, Z_global) -> {X_global:a, Z_global:C_temp1}
    #         Rule body: [parent(X_global, Y_global), parent(Y_global, Z_global)]
    #         Substituted body: [parent(a, Y_global), parent(Y_global, C_temp1)]
    #         State variables renamed: [parent(a, Y_temp2), parent(Y_temp2, C_temp3)] (Y_global -> Y_temp2, C_temp1 -> C_temp3)
    print("Resulting Next States (Step 1):", next_states_4_step1)
    # Expected: [[parent(a, Y_temp), parent(Y_temp, C_temp)]] with appropriate temp var names/IDs

    # Step 2: Process the first state from step 1
    if next_states_4_step1 and next_states_4_step1[0] != [Term('False', [])] and next_states_4_step1[0] != [Term('True', [])] :
         state_for_step_2 = next_states_4_step1[0]
         print(f"\n--- Processing Step 2 with state: {state_for_step_2} ---")
         next_states_4_step2 = get_next_unification_python(state_for_step_2, facts, rules, manager, verbose=1)
         # Query: parent(a, Y_temp2) -> unify with fact parent(a, b) -> {Y_temp2 : b}
         # Remaining state: [parent(Y_temp2, C_temp3)] -> substituted -> [parent(b, C_temp3)]
         # State variables renamed again: [parent(b, C_temp4)]
         print("Resulting Next States (Step 2):", next_states_4_step2)
         # Expected: [[parent(b, C_temp)]]

         # Step 3: Process the first state from step 2
         if next_states_4_step2 and next_states_4_step2[0] != [Term('False', [])] and next_states_4_step2[0] != [Term('True', [])]:
              state_for_step_3 = next_states_4_step2[0]
              print(f"\n--- Processing Step 3 with state: {state_for_step_3} ---")
              next_states_4_step3 = get_next_unification_python(state_for_step_3, facts, rules, manager, verbose=1)
              # Query: parent(b, C_temp4) -> unify with fact parent(b, c) -> {C_temp4 : c}
              # Remaining state: [] -> True()
              print("Resulting Next States (Step 3):", next_states_4_step3)
              # Expected: [[True()]]