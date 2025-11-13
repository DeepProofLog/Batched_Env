from typing import List, Dict, Set, Tuple, FrozenSet, Optional
from str_based.str_utils import Term, Rule

def is_variable(arg: str) -> bool:
    """Check if an argument is a variable."""
    return arg[0].isupper() or arg[0] == '_'

def unify_terms(term1: Term, term2: Term, verbose: int=0) -> Dict[str, str]:
    """
    Attempts to unify two terms and returns a substitution dictionary if successful.
    Returns None if unification is not possible.
    We unify the first term with the second (term1 with term2)
    
    Args:
        term1: First Term object to unify
        term2: Second Term object to unify
    
    Returns:
        Dictionary mapping variables to their substitutions, or None if unification fails
    """
    assert len(term1.args) == len(term2.args) == 2, 'only support binary predicates'
    print('\nterm1:', term1, 'term2:', term2) if verbose else None
    substitutions = {}
    for arg1, arg2 in zip(term1.args, term2.args):
        print('arg1:', arg1, 'arg2:', arg2) if verbose else None

        # If both are constants, they must be equal
        if not (arg1[0].isupper() or arg1[0] == '_') and not (arg2[0].isupper() or arg2[0] == '_'):
            print('both are constants') if verbose else None
            if arg1 != arg2:
                print('constants are different') if verbose else None
                return None
        elif arg2 in substitutions:
            print('arg2 in substitutions') if verbose else None
            if substitutions[arg2] != arg1:
                print('different substitution') if verbose else None
                return None
        else:
            substitutions[arg2] = arg1
            print('substitutions:', substitutions) if verbose else None
    return substitutions

def unify_with_facts(query: Term, 
                    facts_indexed: Dict[Tuple, Set[Term]], 
                    facts_set: FrozenSet[Term], 
                    excluded_fact: Optional[Term] = None, # Added parameter
                    verbose: int=0) -> List[Dict[str, str]]:
    """
    Attempts to unify a query with a list of facts.
    
    Args:
        query: Term object representing the query
        facts: List of Term objects representing facts
    
    Returns:
        List of successful substitution dictionaries
    """

    substitutions = []
    query_is_ground = not any(arg[0].isupper() or arg[0] == '_' for arg in query.args)

    if query_is_ground:
        if query in facts_set and query != excluded_fact:
            substitutions.append({'True': 'True'})
        print('Ground query:', query, 'Substitutions:', substitutions) if verbose else None
        return substitutions

    # For non-ground queries, use the index
    query_constant_args_with_pos = [(i, arg) for i, arg in enumerate(query.args) if not (arg[0].isupper() or arg[0] == '_')]

    # # Sort constant args by position to create a canonical lookup key
    # sorted_query_constant_args = tuple(sorted(query_constant_args_with_pos, key=lambda x: x[0]))
    # lookup_key = (query.predicate,) + sorted_query_constant_args
    lookup_key = (query.predicate,) + tuple(query_constant_args_with_pos)
    # Retrieve candidate facts using the lookup key (later we remove the excluded_fact)
    candidate_facts = facts_indexed.get(lookup_key, set())

    # Iterate only over the candidate facts
    for fact in candidate_facts:
        if fact == excluded_fact:
            continue
        # Verify predicate and arity match before attempting unification
        if fact.predicate == query.predicate:
            subs = unify_terms(fact, query, verbose=0)
            if subs is not None:
                substitutions.append(subs)
        print('Fact:', fact, 'Substitutions:', subs) if verbose else None
    return substitutions

def unify_with_rules(query: Term, rules_by_pred, verbose: int=0) -> List[Tuple[List[Term], Dict[str, str]]]:
    """
    Attempts to unify a query with the heads of rules and returns their bodies with substitutions.
    
    Args:
        query: Term object representing the query
        rules: List of Rule objects to try unifying with
    
    Returns:
        List of tuples containing (rule body terms, substitution dictionary)
    """
    results = []
    for rule in rules_by_pred.get(query.predicate, ()):
        subs = unify_terms(query, rule.head)
        if subs is not None:
            new_body = [
                (term if (substituted_args := tuple(subs.get(arg, arg) for arg in term.args)) == term.args
                 else Term(term.predicate, substituted_args))
                for term in rule.body
            ]
            print('Rule:', rule, '      Subs:', subs, '     New body:', new_body) if verbose else None
            results.append((new_body, subs))
    
    return results


def _needs_renaming(state: List[Term]) -> bool:
    for t in state:
        for a in t.args:
            if a and (a[0].isupper() or a[0] == '_') and not a.startswith('Var_'):
                return True
    return False

def rename_vars_local(next_states: List[List[Term]],
                      global_next_var_index: int,
                      verbose: int = 0,
                      ) -> Tuple[List[List[Term]], int]:
    """
    Renames variables within each state locally, avoiding collisions with
    pre-existing 'Var_...' variables in that state. The global index counter
    tracks the highest index used across all states.

    Args:
        next_states: A list of states, where each state is a list of Terms.
        global_next_var_index: The starting index suggestion from the global scope.
        verbose: Verbosity level.

    Returns:
        A tuple containing:
        - The list of states with variables renamed locally and safely.
        - The updated global_next_var_index after processing all states.
    """
    if global_next_var_index is None:
        raise ValueError('global_next_var_index cannot be None')

    # Fast path: if no state contains non-canonical variables (uppercase or '_' not starting with 'Var_'),
    # skip all work and return unchanged.
    if not any(_needs_renaming(s) for s in next_states):
        return next_states, global_next_var_index

    renamed_states_outer = []    # Initialize the variable here before the loop
    current_state_var_index = global_next_var_index 
    max_index_seen = global_next_var_index
    for idx, state in enumerate(next_states):
        local_var_mapping: Dict[str, str] = {} # Mapping is local to this state
        renamed_state_inner = [None] * len(state)

        # --- Collision Avoidance Step (only executed if we actually rename) ---
        # Find the max 'Var_k' index already in use in this state
        max_existing_k = -1
        existing_vars_in_state = set()
        for term in state:
            for arg in term.args:
                 if isinstance(arg, str) and arg.startswith('Var_'):
                      existing_vars_in_state.add(arg)
                      try:
                           k = int(arg[4:])
                           max_existing_k = max(max_existing_k, k)
                      except ValueError:
                           raise ValueError(f"Invalid variable format: {arg}")

        local_start_index = max(global_next_var_index, max_existing_k + 1)
        current_state_var_index = local_start_index # Counter for new vars in this state

        # --- Renaming Loop ---
        # Iterate over terms and their arguments to rename variables safely 
        # We only rename variables that are not already 'Var_...'
        # We build new args list only if a change is needed (optimization)
        # This avoids unnecessary object creation
        # We also track if any term changed to avoid unnecessary Term creation

        for i, term in enumerate(state):
            original_args = term.args
            new_args_list = None
            term_changed = False

            for j, arg in enumerate(original_args):
                renamed_arg = arg

                # Only rename non-'Var_' variables
                if arg and (arg[0].isupper() or arg[0] == '_') and not arg.startswith('Var_'):
                    mapped_arg = local_var_mapping.get(arg)
                    if mapped_arg is None:
                        new_var_name = f"Var_{current_state_var_index}"
                        local_var_mapping[arg] = new_var_name
                        current_state_var_index += 1 # Increment index for next new var *in this state*
                        renamed_arg = new_var_name
                        term_changed = True
                    else:
                        renamed_arg = mapped_arg
                        if renamed_arg != arg: # Should always be true if mapped
                            term_changed = True
                # Else: Keep constants and existing 'Var_' variables as they are

                # --- Optimization: Build new args list only if necessary ---
                if term_changed and new_args_list is None:
                    new_args_list = list(original_args[:j])
                if new_args_list is not None:
                    new_args_list.append(renamed_arg)

            if term_changed:
                renamed_state_inner[i] = Term(term.predicate, tuple(new_args_list))
            else:
                renamed_state_inner[i] = term

        renamed_states_outer.append(renamed_state_inner)
        max_index_seen = max(max_index_seen, current_state_var_index)

    if renamed_states_outer != next_states:
        print('\n\nRenamed states:', renamed_states_outer) if verbose else None
        print('Original states:', next_states) if verbose else None
    return renamed_states_outer, max_index_seen



def get_next_unification_python(state: List[Term],
                         facts_set: FrozenSet[Term],
                         facts_indexed: Dict[Tuple, Set[Term]],
                         rules: List[Rule],
                         excluded_fact: Optional[Term] = None,
                         next_var_index: Optional[int] = None,
                         strategy: str = 'complete', # 'complete' or 'rules_then_facts'
                         verbose: int = 0,
                         max_derived_states: int = 500) -> Tuple[List[List[Term]], Optional[int]]:
    """
    Processes a state and returns all possible next states based on a chosen strategy.

    Args:
        state: The current list of goals (Term objects).
        facts_set: A frozenset of all known facts for efficient lookup.
        facts_indexed: A dictionary indexing facts for faster unification.
        rules: A list of all known Rule objects.
        excluded_fact: An optional Term to exclude from fact unification.
        next_var_index: The next available index for renaming variables.
        strategy: The unification strategy to use.
                  'complete': Generates all next states from both facts and rules.
                  'rules_then_facts': A heuristic that prioritizes rule applications
                                      that can be immediately resolved by a fact.
        verbose: Verbosity level for printing debug information.
        max_derived_states: Maximum number of derived states to return. If more states
                           are generated, they will be capped and a warning printed.
                           Default is 500 to match tensor engine behavior.

    Returns:
        A tuple containing:
        - A list of possible next states.
        - The updated next_var_index.
    """
    # --- 1. Initial Checks and Setup ---
    if any(term.predicate == 'False' for term in state): return [[Term('False', ())]], next_var_index
    state = [term for term in state if term.predicate != 'True']
    if not state: return [[Term('True', ())]], next_var_index

    # --- 2. Goal Selection & Optimization ---
    # IMPORTANT: Do NOT sort the state! Sorting breaks variable dependencies between atoms.
    # The goals must be processed in the order they appear to maintain variable bindings.
    # For example, in [neighborOf(albania, X), neighborOf(X, Y), locatedInCR(Y, europe)],
    # X and Y must be bound in order.
    # state.sort(key=lambda term: sum(is_variable(arg) for arg in term.args))
    query = state[0]
    remaining_state = state[1:]

    print(f'\n\n++++++++++++++\nProcessing Query: {query} with strategy: {strategy}') if verbose else None
    print(f'Remaining State: {remaining_state}\n') if verbose else None

    next_states = []

    # --- 3. Generate All Potential Next States (Independent of Strategy) ---

    # --- 3a. Generate states from direct Fact Unification ---
    fact_derived_states = []
    fact_substitutions = unify_with_facts(query, facts_indexed, facts_set, excluded_fact, verbose=0)
    print('\nUnification with facts') if verbose else None
    for subs in fact_substitutions:
        if not remaining_state:
            # If there are no remaining goals, we can return True because we found a fact
            print(f"    True next state and no other goals") if verbose else None
            print('++++++++++++++\n') if verbose else None
            return [[Term('True', ())]], next_var_index
        
        if subs.get('True') == 'True': # Ground query matched a fact
            print(f"    Fact next state for sub {subs}") if verbose else None
            fact_derived_states.append(remaining_state)
        else: # Apply substitutions to the remaining goals
            new_state = [
                (term if (substituted_args := tuple(subs.get(arg, arg) for arg in term.args)) == term.args
                    else Term(term.predicate, substituted_args))
                for term in remaining_state
            ]

            # substitute the facts by True
            # NOTE: We MUST check excluded_fact here! After substitution, a remaining goal
            # might become identical to the original query and should not be treated as a fact.
            for j in range(len(new_state)):
                atom = new_state[j]
                if atom in facts_set and atom != excluded_fact:
                    new_state[j] = Term('True', ())

            # if all atoms in the new state are True, we can return True
            if all(term.predicate == 'True' for term in new_state):
                print(f"    True next state for sub {subs}") if verbose else None
                print('++++++++++++++\n') if verbose else None
                return [[Term('True', ())]], next_var_index
            
            # if not all atoms in the new state are True, filter the True atoms
            new_state = [term for term in new_state if term.predicate != 'True']
            print(f"    New state after filtering facts: {new_state}") if verbose else None

            fact_derived_states.append(new_state)

    # --- 3b. Generate states from Rule Unification ---
    rule_derived_states = []
    print('\nUnification with rules') if verbose else None
    rule_results = unify_with_rules(query, rules, verbose=verbose)
    for body, subs in rule_results:
        new_remaining = [
            (term if (substituted_args := tuple(subs.get(arg, arg) for arg in term.args)) == term.args
                else Term(term.predicate, substituted_args))
            for term in remaining_state
        ]
        new_state = body + new_remaining
        
        # Prune ground facts from rule-derived states
        # NOTE: We MUST check excluded_fact here! After rule application and substitution,
        # an atom might become identical to the original query and should not be treated as a fact.
        for j in range(len(new_state)):
            atom = new_state[j]
            if atom in facts_set and atom != excluded_fact:
                new_state[j] = Term('True', ())
        
        # If all atoms are True, return immediately
        if all(term.predicate == 'True' for term in new_state):
            print(f"    True next state from rule") if verbose else None
            return [[Term('True', ())]], next_var_index
        
        # Filter out True atoms
        new_state = [term for term in new_state if term.predicate != 'True']
        
        rule_derived_states.append(new_state)

    # --- 4. Apply Strategy to Select and Combine States ---

    if strategy == 'complete':
        print('Strategy: complete') if verbose else None
        next_states = fact_derived_states + rule_derived_states

    elif strategy == 'rules_then_facts':
        print('\nStrategy: rules_then_facts') if verbose else None
        # --- 4a. Rule -> Fact Path: Prioritize rules that lead to immediate fact resolution ---
        for r_state in rule_derived_states:

            r_query = r_state[0]
            r_remaining = r_state[1:]
            
            # Try to resolve the new first goal with a fact
            # NOTE: r_query comes from a rule body, not the original query, so we don't exclude anything
            r_fact_substitutions = unify_with_facts(r_query, facts_indexed, facts_set, None, verbose=0)
            print(F"State from rule: {r_state}. \nSubs: {r_fact_substitutions}") if verbose else None

            for subs in r_fact_substitutions:
                if not r_remaining: # The rule->fact step solved the entire sub-problem
                    return [[Term('True', ())]], next_var_index
                
                if subs.get('True') == 'True':
                    new_state = r_remaining
                else:
                    new_state = [ # Apply subs to the rest of the rule-derived state
                        (term if (substituted_args := tuple(subs.get(arg, arg) for arg in term.args)) == term.args
                            else Term(term.predicate, substituted_args))
                        for term in r_remaining
                    ]
                    
                # substitute the facts by True
                # NOTE: We MUST check excluded_fact here! After rule+fact unification and substitution,
                # an atom might become identical to the original query and should not be treated as a fact.
                for j in range(len(new_state)):
                    atom = new_state[j]
                    if atom in facts_set and atom != excluded_fact:
                        new_state[j] = Term('True', ())

                # if all atoms in the new state are True, we can return True
                if all(term.predicate == 'True' for term in new_state):
                    print(f"    True next state for sub {subs}") if verbose else None
                    print('++++++++++++++\n') if verbose else None
                    return [[Term('True', ())]], next_var_index
                
                # if not all atoms in the new state are True, filter the True atoms
                new_state = [term for term in new_state if term.predicate != 'True']
                print(f"    New state after filtering facts: {new_state}") if verbose else None

                next_states.append(new_state)

        # --- 4b. Direct Fact Path: Also include states from direct fact lookups ---
        next_states.extend(fact_derived_states)

        # # --- OPTION 1 --- 
        # # --- 4d. This improves completeness but also more time! Append the rule states in any case
        # next_states.extend(rule_derived_states) 

        # # --- OPTION 2 --- 
        # # --- 4c. Fallback Logic: If no progress was made, use the rule states for the next step ---
        # if not next_states and rule_derived_states:
        #     next_states = rule_derived_states


    # --- 5. Finalize and Return ---
    if not next_states:
        print('No unification possible.') if verbose else None
        return [[Term('False', ())]], next_var_index

    # Apply cap to derived states to match tensor engine behavior
    if len(next_states) > max_derived_states:
        if verbose > 0 or len(next_states) > 500:
            print(f'WARNING: Capping derived states from {len(next_states)} to {max_derived_states}')
        next_states = next_states[:max_derived_states]

    # Rename variables to avoid clashes in subsequent unification steps
    next_states, next_var_index = rename_vars_local(next_states, next_var_index, verbose=0)

    print(f'\nNext states: {len(next_states)}, {next_states}') if verbose else None
    print('++++++++++++++\n') if verbose else None
    return next_states, next_var_index