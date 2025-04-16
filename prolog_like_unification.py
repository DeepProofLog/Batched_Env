import itertools
from typing import List, Dict, Optional, Any, Tuple, Set

# --- Placeholder Data Structures ---
# (Use your actual implementations)

class Term:
    def __init__(self, predicate: str, args: List[Any]):
        self.predicate = predicate
        self.args = args

    def __repr__(self):
        # Use _ for internal variables for slightly better readability maybe
        args_repr = [f"_{a[1:]}" if isinstance(a, str) and a.startswith('_') else str(a) for a in self.args]
        return f"{self.predicate}({', '.join(args_repr)})"
        # Original: return f"{self.predicate}({', '.join(map(str, self.args))})"


    def __eq__(self, other):
        if not isinstance(other, Term):
            return NotImplemented
        return self.predicate == other.predicate and self.args == other.args

    def __hash__(self):
        hashable_args = []
        for arg in self.args:
            if isinstance(arg, list): hashable_args.append(tuple(arg))
            elif isinstance(arg, dict): hashable_args.append(tuple(sorted(arg.items())))
            else: hashable_args.append(arg)
        return hash( (self.predicate, tuple(hashable_args)) )


class Rule:
    def __init__(self, head: Term, body: List[Term]):
        self.head = head
        self.body = body

    def __repr__(self):
        body_str = ", ".join(map(str, self.body)) if self.body else "true"
        return f"{self.head} :- {body_str}"

# --- Utility Functions ---

def is_variable(arg: Any) -> bool:
    """Checks if an argument is a variable (e.g., starts with Upper or _)."""
    # Adjust this if your variable convention is different
    return isinstance(arg, str) and (arg[0].isupper() or arg.startswith('_'))

def find_variables(item: Any) -> Set[str]:
    """Recursively finds all unique variable names in a Term, list, or Rule."""
    variables = set()
    if isinstance(item, Term):
        for arg in item.args:
            variables.update(find_variables(arg))
    elif isinstance(item, list):
        for elem in item:
            variables.update(find_variables(elem))
    elif isinstance(item, Rule):
         variables.update(find_variables(item.head))
         variables.update(find_variables(item.body))
    elif is_variable(item):
        variables.add(item)
    # Ignore constants, numbers, etc.
    return variables

# Unique ID generator for variable standardization
_standardize_counter = itertools.count()

def standardize_rule_variables(rule: Rule) -> Rule:
    """
    Creates a new Rule instance with all variables renamed to be unique.

    Args:
        rule: The rule to standardize.

    Returns:
        A new Rule instance with renamed variables.
    """
    rule_vars = find_variables(rule)
    unique_id = next(_standardize_counter)
    subs = {var: f"_{var}_{unique_id}" for var in rule_vars}

    # Create new terms with substitutions applied
    # Important: Use a robust apply_substitution function here
    new_head = apply_substitution(rule.head, subs)
    new_body = apply_substitution(rule.body, subs)

    return Rule(new_head, new_body)


# --- Placeholder Core Logic Functions ---
# (Use your actual, robust implementations)

def unify(term1: Term, term2: Term) -> Optional[Dict[str, Any]]:
    """Placeholder unification function - NEEDS ROBUST IMPLEMENTATION."""
    # Replace with your actual unification algorithm (e.g., Robinson's)
    # This placeholder is insufficient for real use.
    if term1.predicate != term2.predicate or len(term1.args) != len(term2.args):
        return None
    subs = {}
    # Extremely basic matching, WILL FAIL on complex cases (var=var, etc.)
    for arg1, arg2 in zip(term1.args, term2.args):
        if is_variable(arg1):
            if arg1 in subs and subs[arg1] != arg2: return None
            subs[arg1] = arg2
        elif is_variable(arg2):
             if arg2 in subs and subs[arg2] != arg1: return None
             subs[arg2] = arg1 # Allow query constant matching rule variable
        elif arg1 != arg2:
            return None
    return subs if subs else {'_internal_true_': '_internal_true_'} # Use specific marker


def apply_substitution(item: Any, subs: Dict[str, Any]) -> Any:
    """Placeholder substitution function - NEEDS ROBUST IMPLEMENTATION."""
    # Replace with your actual recursive substitution logic
    if not subs or '_internal_true_' in subs: # Handle empty or marker subs
         return item # No substitution needed

    if isinstance(item, Term):
        # Recursively apply to arguments
        new_args = [apply_substitution(arg, subs) for arg in item.args]
        return Term(item.predicate, new_args)
    elif isinstance(item, list):
        # Recursively apply to list elements
        return [apply_substitution(elem, subs) for elem in item]
    elif is_variable(item):
        # Substitute if variable is in substitution map, otherwise keep original
        # Handle potential chain substitutions if needed (e.g., X->Y, Y->a)
        # This simple version doesn't handle chains.
        return subs.get(item, item)
    else:
        # Constants, numbers, etc., remain unchanged
        return item

# --- State Variable Renaming (Simple version) ---
_state_var_counter = itertools.count()
def rename_variables_in_state(state: List[Term]) -> List[Term]:
    """Renames variables in the state to avoid clashes (simple approach)."""
    state_vars = find_variables(state)
    if not state_vars:
        return state # No variables to rename

    unique_id = next(_state_var_counter)
    subs = {var: f"_{var}_s{unique_id}" for var in state_vars}
    # Use the proper apply_substitution function
    new_state = apply_substitution(state, subs)
    return new_state

def get_next_unification_prolog_like(
    initial_state: List[Term],
    facts: List[Term],
    rules: List[Rule],
    verbose: int = 0
) -> List[List[Term]]:
    """
    Processes a state like Prolog: resolves facts internally at the current
    depth and returns only states resulting from rule applications (next depth).
    Includes proper rule variable standardization.

    Args:
        initial_state: List of Term objects representing the current goals.
        facts: List of Term objects representing known facts.
        rules: List of Rule objects representing inference rules.
        verbose: Verbosity level.

    Returns:
        List of possible next states resulting ONLY from rule applications,
        or [[Term('True', [])]] if solved by facts,
        or [[Term('False', [])]] if failed.
    """
    # --- Initial Checks & State Handling ---
    if not initial_state: return [[Term('True', [])]]
    if any(term.predicate == 'False' for term in initial_state): return [[Term('False', [])]]
    # Remove True terms initially
    current_state = [term for term in initial_state if term.predicate != 'True']
    if not current_state: return [[Term('True', [])]]

    # --- State Variable Standardization ---
    # Ensures state variables have unique names initially for this step
    current_state = rename_variables_in_state(current_state)
    if verbose > 1: print(f"Standardized State: {current_state}")


    # --- Internal Fact Resolution Loop ---
    processed_state = current_state # Start with the standardized state
    while True:
        # Remove 'True' terms and check if state is resolved
        processed_state = [term for term in processed_state if term.predicate != 'True']
        if not processed_state:
            if verbose > 0: print("State resolved by facts.")
            return [[Term('True', [])]] # Success via facts

        if verbose > 1: print(f"Processing State: {processed_state}")

        query, *remaining_state = processed_state
        if verbose > 1: print(f"  Query: {query}, Remaining: {remaining_state}")

        fact_match_found = False
        for fact in facts:
             # In a full system, facts might also need standardization if they contain vars
             subs = unify(query, fact) # Use your real unify function

             # Check for actual substitution, not just marker
             if subs is not None and '_internal_true_' not in subs:
                if verbose > 1: print(f"  Matched Fact: {fact} with Subs: {subs}")
                fact_match_found = True
                new_remaining_state = apply_substitution(remaining_state, subs)
                processed_state = new_remaining_state # Update state for next loop iteration
                if verbose > 1: print(f"  New State after fact: {processed_state}")
                break # Fact found, process the new state from the start of the loop
             elif subs is not None and '_internal_true_' in subs: # Query matched a ground fact exactly
                 if verbose > 1: print(f"  Matched Ground Fact: {fact}")
                 fact_match_found = True
                 processed_state = remaining_state # Just remove the query
                 if verbose > 1: print(f"  New State after ground fact: {processed_state}")
                 break # Fact found, process the new state

        if fact_match_found:
            continue # Go to the start of the while loop with the updated state

        # --- If no fact matched the current query, try rules ---
        if verbose > 0: print(f"Query '{query}' requires rule application.")
        # Store the state *before* rule application needed it
        state_requiring_rule = processed_state
        break # Exit the while loop to proceed to rule expansion


    # --- Rule Expansion ---
    next_states_from_rules = []
    query_for_rules, *remaining_state_for_rules = state_requiring_rule
    if verbose > 0: print(f"Attempting rule unification for: {query_for_rules}")

    for i, original_rule in enumerate(rules):

         # ---> Standardize Rule Variables <---
         rule = standardize_rule_variables(original_rule)
         if verbose > 2: print(f"  Trying Standardized Rule {i}: {rule}")

         subs = unify(query_for_rules, rule.head) # Use standardized head

         if subs is not None:
            if verbose > 0: print(f"  Matched Rule {i}: {original_rule} (as {rule.head}) with Subs: {subs}")

            # Apply substitution to the rule body (using standardized body)
            if '_internal_true_' in subs:
                 new_body = rule.body # Already standardized, no vars in query matched
            else:
                 new_body = apply_substitution(rule.body, subs)

            # Apply substitution to the remaining state goals
            if '_internal_true_' in subs:
                 new_remaining_state = remaining_state_for_rules # Already standardized
            else:
                 new_remaining_state = apply_substitution(remaining_state_for_rules, subs)

            # Combine rule body and remaining state
            new_state = new_body + new_remaining_state
            if verbose > 1: print(f"    Generated New State: {new_state}")

            # Optional: Basic simplification (remove True - can be done here or start of loop)
            new_state_simplified = [term for term in new_state if term.predicate != 'True']

            # Add the resulting state (list of goals) to the results
            # If simplification resulted in empty list, represent as success [[True]]
            # Handle case where body+remaining is empty AFTER substitution
            if not new_state_simplified:
                next_states_from_rules.append([Term('True', [])])
                if verbose > 1: print("    Rule application resulted in immediate True state.")
            else:
                next_states_from_rules.append(new_state_simplified)


    # --- Return Result ---
    if not next_states_from_rules:
        # This means the query couldn't be resolved by facts OR rules
        if verbose > 0: print(f"No facts or rules matched query '{query_for_rules}'. Failing.")
        return [[Term('False', [])]]
    else:
        if verbose > 0: print(f"Returning next states from rules: {next_states_from_rules}")
        # Result is a list of lists of terms [[goal1, goal2], [alt_goal1]]
        return next_states_from_rules

# --- Example Usage (identical to before) ---
# (Include the example usage from the previous response here to test)
if __name__ == '__main__':
    # Define some facts and rules
    facts_db = [
        Term("parent", ["pam", "bob"]),
        Term("parent", ["tom", "bob"]),
        Term("parent", ["tom", "liz"]),
        Term("parent", ["bob", "ann"]),
        Term("parent", ["bob", "pat"]),
        Term("parent", ["pat", "jim"]),
    ]

    rules_db = [
        Rule(
            Term("grandparent", ["X", "Z"]),
            [Term("parent", ["X", "Y"]), Term("parent", ["Y", "Z"])]
        ),
        Rule(
            Term("sibling", ["X", "Y"]),
            [Term("parent", ["P", "X"]), Term("parent", ["P", "Y"])] # Simplified: Needs X \= Y
        )
    ]

    # Reset counters for consistent example runs
    _standardize_counter = itertools.count()
    _state_var_counter = itertools.count()
    print("--- Test 1: Simple Fact ---")
    state1 = [Term("parent", ["pam", "bob"])]
    next1 = get_next_unification_prolog_like(state1, facts_db, rules_db, verbose=1)
    print("Result:", next1) # Expected: [[Term('True', [])]]

    # Reset counters
    _standardize_counter = itertools.count()
    _state_var_counter = itertools.count()
    print("\n--- Test 2: Simple Rule ---")
    state2 = [Term("grandparent", ["tom", "ann"])]
    next2 = get_next_unification_prolog_like(state2, facts_db, rules_db, verbose=1)
    # Expected: [[parent(tom_s0, _Y_0), parent(_Y_0, ann_s0)]] (vars depend on exact standardization)
    print("Result:", next2)

    # Reset counters
    _standardize_counter = itertools.count()
    _state_var_counter = itertools.count()
    print("\n--- Test 3: Fact resolution within rule path (Should be True) ---")
    state3 = [Term("grandparent", ["pam", "ann"])] # pam -> bob -> ann
    next3 = get_next_unification_prolog_like(state3, facts_db, rules_db, verbose=2)
    # Expected: [[Term('True', [])]] because internal fact loop resolves it
    print("Result:", next3)

    # Reset counters
    _standardize_counter = itertools.count()
    _state_var_counter = itertools.count()
    print("\n--- Test 4: Rule needing further steps ---")
    state4 = [Term("sibling", ["ann", "pat"])] # Needs parent(P, ann), parent(P, pat)
    next4 = get_next_unification_prolog_like(state4, facts_db, rules_db, verbose=1)
    # Expected: [[parent(_P_0, ann_s0), parent(_P_0, pat_s0)]]
    print("Result:", next4)


    # Reset counters
    _standardize_counter = itertools.count()
    _state_var_counter = itertools.count()
    print("\n--- Test 5: Failure ---")
    state5 = [Term("parent", ["pam", "jim"])]
    next5 = get_next_unification_prolog_like(state5, facts_db, rules_db, verbose=1)
    print("Result:", next5) # Expected: [[Term('False', [])]]

    # Reset counters
    _standardize_counter = itertools.count()
    _state_var_counter = itertools.count()
    print("\n--- Test 6: Multiple Goals, first is fact ---")
    state6 = [Term("parent", ["tom", "liz"]), Term("parent", ["bob", "pat"])]
    next6 = get_next_unification_prolog_like(state6, facts_db, rules_db, verbose=2)
    # Expected: [[Term('True', [])]] because internal loop resolves both facts
    print("Result:", next6)

    # Reset counters
    _standardize_counter = itertools.count()
    _state_var_counter = itertools.count()
    print("\n--- Test 7: Multiple Goals, first needs rule ---")
    state7 = [Term("grandparent", ["pam", "A"]), Term("parent", ["tom", "liz"])]
    next7 = get_next_unification_prolog_like(state7, facts_db, rules_db, verbose=1)
    # Expected: [[parent(pam_s0, _Y_0), parent(_Y_0, A_s0), parent(tom, liz)]]
    print("Result:", next7)