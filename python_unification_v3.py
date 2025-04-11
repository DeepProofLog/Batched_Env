import torch
from typing import Tuple, Optional, Union, List, Set, Dict, NamedTuple
from env_v4 import IndexManager
# --- Constants ---
PADDING_VALUE = -999 # Value for padding tensors
UNBOUND_VAR = -998   # Placeholder for unbound variables (primarily for formatting/debugging)

# --- Data Structures for Clarity ---
class UnificationResult(NamedTuple):
    substitutions: torch.Tensor  # Shape: (bs, k, max_subs, 2)
    target_indices: torch.Tensor # Shape: (bs, k) - Index of fact or rule
    valid_mask: torch.Tensor     # Shape: (bs, k) - Indicates valid unifications
    target_bodies: Optional[torch.Tensor] = None # Shape: (bs, k, body_len, 3) - Only for rules

# --- Helper Functions (Mostly Unchanged) ---

# --- Unification Functions (Keep _get_var_set_and_tensor, formatters, apply_substitutions_vectorized, batch_unify_with_facts as is) ---
def _get_var_set_and_tensor(
    vars_idx: Union[torch.Tensor, List[int], Set[int], IndexManager],
    device: torch.device
) -> Tuple[Set[int], torch.Tensor]:
    """Ensures vars_idx is a set and a tensor on the correct device."""
    if isinstance(vars_idx, IndexManager):
        return vars_idx.vars_idx_set, vars_idx.vars_idx_tensor.to(device)

    if isinstance(vars_idx, torch.Tensor):
        vars_idx_tensor = vars_idx.to(device=device, dtype=torch.long).detach()
        vars_idx_set = set(vars_idx_tensor.cpu().tolist())
    elif isinstance(vars_idx, (list, set)):
        vars_idx_set = set(vars_idx)
        vars_idx_tensor = torch.tensor(list(vars_idx_set), dtype=torch.long, device=device)
    else:
        raise TypeError("vars_idx must be an IndexManager, tensor, list, or set")
    if vars_idx_tensor.dim() == 0:
        vars_idx_tensor = vars_idx_tensor.unsqueeze(0)
    if not torch.all(vars_idx_tensor < 0):
         logging.warning(f"Variable tensor contains non-negative values: {vars_idx_tensor[vars_idx_tensor >= 0]}.")
    return vars_idx_set, vars_idx_tensor

def _format_term(
    term_idx: int,
    vars_idx_set: Set[int],
    constant_idx2str: Optional[Dict[int, str]] = None,
    variable_idx2str: Optional[Dict[int, str]] = None
) -> str:
    """Formats a single term (subject or object)."""
    if term_idx == PADDING_VALUE: return "[pad]"
    if term_idx == UNBOUND_VAR: return "[unbound]"
    if term_idx in vars_idx_set:
        var_str = variable_idx2str.get(term_idx, f"VAR_IDX({term_idx})") if variable_idx2str else f"VAR({term_idx})"
        return var_str
    if constant_idx2str and term_idx in constant_idx2str:
        return constant_idx2str[term_idx]
    if variable_idx2str and term_idx in variable_idx2str: # Should not happen if vars_idx_set is correct
        return variable_idx2str[term_idx] + "[?]" # Mark as unexpected if found here
    return f"IDX({term_idx})"

def _format_atom(
    atom_tensor: torch.Tensor,
    predicate_idx2str: Optional[Dict[int, str]] = None,
    constant_idx2str: Optional[Dict[int, str]] = None,
    vars_idx_set: Optional[Set[int]] = None,
    variable_idx2str: Optional[Dict[int, str]] = None
) -> str:
    """Formats an atom tensor (shape [3]) into a readable string."""
    if atom_tensor.is_cuda: atom_tensor = atom_tensor.cpu()

    if not isinstance(atom_tensor, torch.Tensor) or atom_tensor.dim() < 1 or atom_tensor.shape[-1] != 3:
        return f"[invalid atom shape: {atom_tensor.shape}]"
    if atom_tensor.dim() > 1: atom_tensor = atom_tensor.view(-1, 3)[0]
    if atom_tensor.numel() != 3: return f"[invalid atom numel: {atom_tensor.numel()}]"

    p_idx, s_idx, o_idx = atom_tensor.tolist()

    is_padding_atom = (p_idx == PADDING_VALUE and s_idx == PADDING_VALUE and o_idx == PADDING_VALUE)
    if is_padding_atom: return "[padding_atom]"

    p_str = predicate_idx2str.get(p_idx, f"PRED_IDX({p_idx})") if predicate_idx2str else f"P({p_idx})"

    if p_str in ['True', 'False', 'End']:
        # Check if S and O are padding for these special predicates
        if s_idx != PADDING_VALUE or o_idx != PADDING_VALUE:
             # Return the index representation if args are not padding
             s_str_err = _format_term(s_idx, vars_idx_set or set(), constant_idx2str, variable_idx2str)
             o_str_err = _format_term(o_idx, vars_idx_set or set(), constant_idx2str, variable_idx2str)
             return f"({p_str}, {s_str_err}, {o_str_err})[!]" # Mark as unexpected format
        else:
             return f"({p_str})" # Correct format

    vars_set = vars_idx_set if isinstance(vars_idx_set, set) else set()
    s_str = _format_term(s_idx, vars_set, constant_idx2str, variable_idx2str)
    o_str = _format_term(o_idx, vars_set, constant_idx2str, variable_idx2str)

    return f"({p_str}, {s_str}, {o_str})"

def _format_substitution(
    sub_pair: torch.Tensor, # Shape (2,) : [var_idx, value_idx]
    constant_idx2str: Optional[Dict[int, str]] = None,
    vars_idx_set: Optional[Set[int]] = None,
    variable_idx2str: Optional[Dict[int, str]] = None
) -> str:
    """Formats a substitution pair tensor into a readable string."""
    if sub_pair.is_cuda: sub_pair = sub_pair.cpu()

    if not isinstance(sub_pair, torch.Tensor) or sub_pair.shape != (2,):
        return f"[invalid sub pair: {sub_pair}]"

    var_idx, value_idx = sub_pair.tolist()

    if var_idx == PADDING_VALUE: return "[no_sub]"

    vars_set = vars_idx_set if isinstance(vars_idx_set, set) else set()

    var_str = "[var?]"
    if var_idx < 0 :
        var_str = variable_idx2str.get(var_idx, f"VAR_IDX({var_idx})") if variable_idx2str else f"VAR({var_idx})"
    else:
        var_str = f"NON_VAR({var_idx})" # Should not happen

    value_str = _format_term(value_idx, vars_set, constant_idx2str, variable_idx2str)
    return f"{var_str} -> {value_str}"


def apply_substitutions_vectorized(
    atoms: torch.Tensor,              # Shape: (..., num_atoms, 3)
    substitutions: torch.Tensor,      # Shape: (..., max_subs, 2), [var_idx, value_idx]
    vars_idx_tensor: torch.Tensor,    # 1D Tensor of variable indices
    max_iterations: int = 10,         # Limit recursion/iteration depth
) -> torch.Tensor:
    """
    Applies substitutions (Var -> Value) to a tensor of atoms iteratively using vectorized operations.
    Handles Var -> Const and Var -> Var substitutions until no changes occur or max_iterations.
    Assumes `substitutions` can broadcast to the leading dimensions of `atoms`.
    """
    if atoms.numel() == 0: return atoms.clone()
    if atoms.shape[-1] != 3: raise ValueError(f"Last dimension of atoms must be 3, but got shape {atoms.shape}")
    if substitutions.numel() > 0 and substitutions.shape[-1] != 2: raise ValueError(f"Last dimension of substitutions must be 2, but got shape {substitutions.shape}")

    device = atoms.device
    substituted_atoms = atoms.clone()

    if substitutions.numel() == 0: return substituted_atoms

    # Ensure substitutions are on the correct device
    substitutions = substitutions.to(device)

    valid_subs_mask = (substitutions[..., 0] != PADDING_VALUE) # Shape: (..., max_subs)
    if not torch.any(valid_subs_mask): return substituted_atoms

    changed = True
    iterations = 0
    while changed and iterations < max_iterations:
        changed = False
        iterations += 1

        max_subs_dim = substitutions.shape[-2]
        for sub_idx in range(max_subs_dim):
            current_sub = substitutions[..., sub_idx, :] # Shape (..., 2)
            current_mask = valid_subs_mask[..., sub_idx] # Shape (...)

            if not torch.any(current_mask): continue

            # Use PADDING_VALUE as placeholder for inactive substitutions
            var_to_replace = torch.where(current_mask, current_sub[..., 0], PADDING_VALUE)
            replacement_value = torch.where(current_mask, current_sub[..., 1], PADDING_VALUE)

            # --- Broadcasting Setup ---
            # Ensure var_to_replace and replacement_value can broadcast to atoms shape
            num_atom_dims = substituted_atoms.dim()
            # Shape to broadcast: (..., 1, 1) to match (..., num_atoms, 3)
            view_shape_list = list(var_to_replace.shape)
            # Add singleton dims for num_atoms and the P/S/O dimension
            view_shape_list.extend([1] * (num_atom_dims - len(view_shape_list) - 1))
            view_shape_list.append(1) # For the S/O dimension comparison
            view_shape = tuple(view_shape_list)

            var_to_replace_b = var_to_replace.view(view_shape)
            replacement_value_b = replacement_value.view(view_shape)
            # Also broadcast the mask
            current_mask_b = current_mask.view(view_shape[:-1] + (1,)) # Broadcast mask needs shape (..., 1) or similar

            # --- Apply Substitution to Subject (Column 1) ---
            # Check where S matches the variable to replace, AND the substitution is active
            s_match_mask = (substituted_atoms[..., 1:2] == var_to_replace_b) & current_mask_b
            if torch.any(s_match_mask):
                current_s_values = substituted_atoms[..., 1:2]
                # Only apply if the value actually changes
                apply_s_mask = s_match_mask & (current_s_values != replacement_value_b)
                if torch.any(apply_s_mask):
                    substituted_atoms[..., 1:2] = torch.where(apply_s_mask, replacement_value_b, current_s_values)
                    changed = True

            # --- Apply Substitution to Object (Column 2) ---
            # Check where O matches the variable to replace, AND the substitution is active
            o_match_mask = (substituted_atoms[..., 2:3] == var_to_replace_b) & current_mask_b
            if torch.any(o_match_mask):
                current_o_values = substituted_atoms[..., 2:3]
                 # Only apply if the value actually changes
                apply_o_mask = o_match_mask & (current_o_values != replacement_value_b)
                if torch.any(apply_o_mask):
                    substituted_atoms[..., 2:3] = torch.where(apply_o_mask, replacement_value_b, current_o_values)
                    changed = True

    if iterations >= max_iterations and changed:
        logging.warning(f"Max substitution iterations ({max_iterations}) reached.")

    return substituted_atoms


def batch_unify_with_facts(
    states_idx: torch.Tensor,             # Shape: (bs, 1, n_padding_atoms, 3)
    facts: torch.Tensor,                  # Shape: (n_facts, 3)
    vars_idx: Union[torch.Tensor, List[int], Set[int], IndexManager], # Variable indices or IndexManager
    k: int,                               # Max number of unifications to return
    device: Optional[torch.device] = None,
    verbose: bool = False,
    predicate_idx2str: Optional[Dict[int, str]] = None,
    constant_idx2str: Optional[Dict[int, str]] = None,
    variable_idx2str: Optional[Dict[int, str]] = None
) -> UnificationResult:
    """
    Performs batched unification for the first query against FACTS. (Vectorized)
    Returns simple substitutions (Query Var -> Fact Const) for the *first* K unifying facts found per query.
    Detects conflicts where a query like (P, VarX, VarX) tries to unify with (P, A, B).
    MODIFIED: Selects the first k valid unifications without sorting all potential ones.
    """
    if k <= 0: raise ValueError("k must be a positive integer")

    effective_device = device if device else states_idx.device
    states_idx = states_idx.to(effective_device)
    facts = facts.to(effective_device)
    vars_idx_set, vars_idx_tensor = _get_var_set_and_tensor(vars_idx, effective_device)

    bs, _, n_padding_atoms, _ = states_idx.shape
    n_facts = facts.shape[0]

    # Determine actual k based on available facts
    actual_k = min(k, n_facts) if n_facts > 0 else 0

    # Initialize output tensors with padding
    # Max 2 substitutions needed (S -> S_fact, O -> O_fact)
    substitutions_out = torch.full((bs, k, 2, 2), PADDING_VALUE, dtype=torch.long, device=effective_device)
    fact_indices_out = torch.full((bs, k), PADDING_VALUE, dtype=torch.long, device=effective_device)
    valid_mask_out = torch.zeros((bs, k), dtype=torch.bool, device=effective_device)

    # Handle edge cases: no batch items, no facts, or no query atoms
    if bs == 0 or n_facts == 0 or n_padding_atoms == 0:
        return UnificationResult(substitutions_out, fact_indices_out, valid_mask_out)

    # --- Extract First Query Atom ---
    first_queries = states_idx[:, 0, 0, :] # Shape: (bs, 3)
    # Check if the first query is just padding
    is_query_valid = ~torch.all(first_queries == PADDING_VALUE, dim=-1) # Shape: (bs,)

    # --- Prepare for Broadcasting ---
    queries_expanded = first_queries.unsqueeze(1) # (bs, 1, 3)
    facts_expanded = facts.unsqueeze(0)           # (1, n_facts, 3)

    # --- Check Basic Unification Conditions (Vectorized) ---
    # Predicate match
    pred_match = (queries_expanded[:, :, 0] == facts_expanded[:, :, 0]) # (bs, n_facts)

    # Subject match: Query S is variable OR Query S == Fact S
    is_query_var_s = torch.isin(queries_expanded[:, :, 1], vars_idx_tensor.view(-1)) # (bs, 1) -> broadcasts
    subj_match = is_query_var_s | (queries_expanded[:, :, 1] == facts_expanded[:, :, 1]) # (bs, n_facts)

    # Object match: Query O is variable OR Query O == Fact O
    is_query_var_o = torch.isin(queries_expanded[:, :, 2], vars_idx_tensor.view(-1)) # (bs, 1) -> broadcasts
    obj_match = is_query_var_o | (queries_expanded[:, :, 2] == facts_expanded[:, :, 2]) # (bs, n_facts)

    # Initial unification mask (predicate, subject, object must match/be variable)
    # Also ensure the query itself is valid (not padding)
    unifies_mask_initial = pred_match & subj_match & obj_match & is_query_valid.unsqueeze(1) # (bs, n_facts)

    # If no potential matches or k is 0, return padded results
    if actual_k == 0 or not torch.any(unifies_mask_initial):
        return UnificationResult(substitutions_out, fact_indices_out, valid_mask_out)

    # --- Find Indices of the First K Potential Matches (No Sorting) ---
    # Calculate cumulative sum of True values along the fact dimension
    cum_true = torch.cumsum(unifies_mask_initial.long(), dim=1) # (bs, n_facts)

    # Create a mask for elements that are True in the initial mask AND are within the first 'actual_k' True elements
    first_k_mask = (cum_true <= actual_k) & (cum_true > 0) & unifies_mask_initial # (bs, n_facts)

    # Get the coordinates (batch_idx, fact_idx) of these first k potential matches
    first_k_coords = torch.nonzero(first_k_mask, as_tuple=True)
    n_potential_k = len(first_k_coords[0]) # Total number of potential matches found across all batches (up to bs*k)

    # Initialize final masks and outputs for the top-k slice (up to actual_k)
    final_valid_mask_slice = torch.zeros((bs, actual_k), dtype=torch.bool, device=effective_device)
    gathered_subs = torch.full((bs, actual_k, 2, 2), PADDING_VALUE, dtype=torch.long, device=effective_device)
    gathered_indices = torch.full((bs, actual_k), PADDING_VALUE, dtype=torch.long, device=effective_device)

    if n_potential_k > 0:
        potential_bs_indices, potential_fact_indices_flat = first_k_coords # (n_potential_k,), (n_potential_k,)

        # Determine the 'k-th' index (0 to k-1) for each potential match within its batch row
        potential_k_indices = cum_true[first_k_coords] - 1 # (n_potential_k,)

        # Gather the corresponding queries and facts for these potential matches
        potential_queries_flat = first_queries[potential_bs_indices] # Shape: (n_potential_k, 3)
        potential_facts_flat = facts[potential_fact_indices_flat]    # Shape: (n_potential_k, 3)

        # --- Conflict Check: Query(P, X, X) vs Fact(P, A, B) where A != B ---
        # Check if query S and O are the *same* variable
        is_query_s_var_flat = torch.isin(potential_queries_flat[:, 1], vars_idx_tensor.view(-1))
        is_query_o_var_flat = torch.isin(potential_queries_flat[:, 2], vars_idx_tensor.view(-1))
        is_same_query_var_flat = is_query_s_var_flat & is_query_o_var_flat & \
                                 (potential_queries_flat[:, 1] == potential_queries_flat[:, 2]) # (n_potential_k,)

        # Check if fact S and O are different constants/terms
        is_diff_fact_const_flat = (potential_facts_flat[:, 1] != potential_facts_flat[:, 2]) # (n_potential_k,)

        # Conflict occurs if query has same var but fact has different terms
        is_conflict_flat = is_same_query_var_flat & is_diff_fact_const_flat # (n_potential_k,)

        # Update the slice of the final valid mask - only non-conflicting ones are truly valid
        # Use scatter/indexing to place the non-conflict results at the correct (bs, k) location
        final_valid_mask_slice[potential_bs_indices, potential_k_indices] = ~is_conflict_flat

        # --- Gather Substitutions for Valid, Non-Conflicting Unifications ---
        # Find coordinates of the final valid unifications within the (bs, actual_k) slice
        final_valid_coords = torch.nonzero(final_valid_mask_slice, as_tuple=True)
        n_final_valid = len(final_valid_coords[0])

        if n_final_valid > 0:
            # Filter down to the non-conflicting entries among the potentials
            non_conflict_mask_flat = ~is_conflict_flat # (n_potential_k,)

            # Get the queries, facts, and fact indices for the final valid set
            # Filter the flat tensors using the non_conflict_mask_flat
            final_queries = potential_queries_flat[non_conflict_mask_flat] # (n_final_valid, 3)
            final_facts = potential_facts_flat[non_conflict_mask_flat]     # (n_final_valid, 3)
            final_fact_idxs_out = potential_fact_indices_flat[non_conflict_mask_flat] # (n_final_valid,)

            # Get the (bs, k) coordinates for these final valid entries
            final_bs_indices, final_k_indices = final_valid_coords

            # Place the fact indices into the gathered output tensor at the correct (bs, k) coords
            gathered_indices[final_bs_indices, final_k_indices] = final_fact_idxs_out

            # --- Create Substitution Pairs (Var -> Const) ---
            # Check if query S/O are variables in the final valid set
            is_s_var_final = torch.isin(final_queries[:, 1], vars_idx_tensor.view(-1)) # (n_final_valid,)
            is_o_var_final = torch.isin(final_queries[:, 2], vars_idx_tensor.view(-1)) # (n_final_valid,)

            # Subject substitution: [Query S Var, Fact S Value] if Query S is var, else [PAD, PAD]
            s_sub_pairs = torch.stack([
                torch.where(is_s_var_final, final_queries[:, 1], PADDING_VALUE),
                torch.where(is_s_var_final, final_facts[:, 1], PADDING_VALUE)
            ], dim=1) # (n_final_valid, 2)

            # Object substitution: [Query O Var, Fact O Value] if Query O is var, else [PAD, PAD]
            o_sub_pairs = torch.stack([
                torch.where(is_o_var_final, final_queries[:, 2], PADDING_VALUE),
                torch.where(is_o_var_final, final_facts[:, 2], PADDING_VALUE)
            ], dim=1) # (n_final_valid, 2)

            # Place the substitution pairs into the gathered output tensor at the correct (bs, k) coords
            gathered_subs[final_bs_indices, final_k_indices, 0, :] = s_sub_pairs
            gathered_subs[final_bs_indices, final_k_indices, 1, :] = o_sub_pairs

    # --- Populate Final Output Tensors ---
    # Copy the gathered results (up to actual_k) into the full output tensors (size k)
    substitutions_out[:, :actual_k] = gathered_subs
    fact_indices_out[:, :actual_k] = gathered_indices
    valid_mask_out[:, :actual_k] = final_valid_mask_slice

    if verbose:
        # (Add more detailed printing here if needed for debugging)
        print("-" * 40 + "\nDEBUG: batch_unify_with_facts (End)\n" + "-" * 40)

    return UnificationResult(substitutions_out, fact_indices_out, valid_mask_out)

# --- CORRECTED/REVISED batch_unify_with_rules ---

# Helper function for `batch_unify_with_rules`
def get_binding(var_idx: int,
                subs_dict: Dict[int, int],
                visited: Set[int],
                vars_idx_set: Set[int],
                max_depth: int = 20 # Increased depth limit slightly
                ) -> Optional[int]:
    """
    Finds the ultimate binding for a term index, detecting cycles.
    Returns the final term index (could be original var_idx, another var, a const) or None if cycle detected.
    """
    # Base case: If it's not a variable OR it's a variable not currently bound
    if var_idx not in vars_idx_set or var_idx not in subs_dict:
        return var_idx

    # Cycle detection / Depth limit
    if var_idx in visited:
        # logging.debug(f"Cycle detected for var {var_idx} in get_binding. Visited: {visited}")
        return None
    if len(visited) > max_depth:
        # logging.warning(f"Max depth ({max_depth}) exceeded in get_binding for var {var_idx}.")
        return None

    visited.add(var_idx)
    next_val = subs_dict[var_idx]
    # Recursively find the binding of the next value
    result = get_binding(next_val, subs_dict, visited, vars_idx_set, max_depth)
    # Backtrack: remove from visited set *after* the recursive call returns
    # This correctly handles graph traversal style cycle detection
    visited.remove(var_idx)

    return result

# Helper function for `batch_unify_with_rules`
def occurs_check(var_idx: int,
                 term_idx: int,
                 subs_dict: Dict[int, int],
                 vars_idx_set: Set[int],
                 max_depth: int = 20) -> bool:
    """
    Simplified occurs check for PSO representation.
    Checks if var_idx is reachable from term_idx following substitutions.
    Returns True if var_idx occurs in term_idx (or its bindings), False otherwise.
    """
    # If term_idx is not a variable, var_idx cannot occur in it (in PSO)
    if term_idx not in vars_idx_set:
        return False

    # Check the binding chain of term_idx
    current = term_idx
    visited_check = set()
    depth = 0
    while current in vars_idx_set and current in subs_dict and depth < max_depth:
        if current == var_idx:
            return True # Found the variable in the chain
        if current in visited_check:
            # logging.debug(f"Cycle detected during occurs_check from {term_idx} for {var_idx}.")
            return True # Treat cycle as occurrence for safety
        visited_check.add(current)
        current = subs_dict[current]
        depth += 1

    # After the loop, check the final resolved value
    return current == var_idx


def batch_unify_with_rules(
    states_idx: torch.Tensor,             # Shape: (bs, 1, n_padding_atoms, 3)
    rules: torch.Tensor,                  # Shape: (n_rules, max_rule_atoms, 3) Head=0, Body=1:
    vars_idx: Union[torch.Tensor, List[int], Set[int], IndexManager], # Variable indices or IndexManager
    k: int,                               # Max number of rule unifications
    max_subs_per_rule: int = 5,           # Max substitutions tracked per unification
    device: Optional[torch.device] = None,
    verbose: bool = False, # Main env verbose flag (used below for higher level debug)
    prover_verbose: int = 0, # <<< Controls detail level of unification debugging prints
    predicate_idx2str: Optional[Dict[int, str]] = None,
    constant_idx2str: Optional[Dict[int, str]] = None,
    variable_idx2str: Optional[Dict[int, str]] = None
) -> UnificationResult:
    """
    Performs batched unification for the first query against RULE HEADS. (Corrected)
    Calculates the Most General Unifier (MGU) substitutions needed.
    Handles Var->Const, Const->Var, Var->Var bindings and detects conflicts/cycles.
    NOTE: The core MGU calculation per (query, rule_head) pair remains iterative.
    """
    if k <= 0: raise ValueError("k must be a positive integer")
    if max_subs_per_rule <= 0: raise ValueError("max_subs_per_rule must be positive")

    effective_device = device if device else states_idx.device
    states_idx = states_idx.to(effective_device)
    rules = rules.to(effective_device)
    vars_idx_set, vars_idx_tensor = _get_var_set_and_tensor(vars_idx, effective_device)

    bs, _, n_padding_atoms, _ = states_idx.shape
    if rules.numel() == 0:
        n_rules, max_rule_atoms = 0, 0
    else:
        n_rules, max_rule_atoms, _ = rules.shape
    rule_body_len = max(0, max_rule_atoms - 1)

    actual_k = min(k, n_rules) if n_rules > 0 else 0

    # --- Initialize Output Tensors ---
    substitutions_out = torch.full((bs, k, max_subs_per_rule, 2), PADDING_VALUE, dtype=torch.long, device=effective_device)
    rule_indices_out = torch.full((bs, k), PADDING_VALUE, dtype=torch.long, device=effective_device)
    rule_bodies_out_shape = (bs, k, rule_body_len, 3)
    rule_bodies_out = torch.full(rule_bodies_out_shape, PADDING_VALUE, dtype=torch.long, device=effective_device)
    valid_mask_out = torch.zeros((bs, k), dtype=torch.bool, device=effective_device)

    if bs == 0 or n_rules == 0 or n_padding_atoms == 0 or max_rule_atoms <= 0:
        return UnificationResult(substitutions_out, rule_indices_out, valid_mask_out, rule_bodies_out)

    # --- Prepare Inputs ---
    first_queries = states_idx[:, 0, 0, :] # (bs, 3)
    is_query_valid = ~torch.all(first_queries == PADDING_VALUE, dim=-1) # (bs,)

    rule_heads = rules[:, 0, :]      # (n_rules, 3)
    rule_bodies = rules[:, 1:, :]    # (n_rules, rule_body_len, 3)

    # --- Initial Predicate Match ---
    queries_expanded = first_queries.unsqueeze(1) # (bs, 1, 3)
    heads_expanded = rule_heads.unsqueeze(0)      # (1, n_rules, 3)
    # Predicate must match and query must be valid (not padding)
    pred_match = (queries_expanded[:, :, 0] == heads_expanded[:, :, 0]) & is_query_valid.unsqueeze(1) # (bs, n_rules)

    # Precompute which query/head args are variables (for efficiency in loop)
    is_query_s_var = torch.isin(first_queries[:, 1:2], vars_idx_tensor.view(-1)) # (bs, 1)
    is_query_o_var = torch.isin(first_queries[:, 2:3], vars_idx_tensor.view(-1)) # (bs, 1)
    is_head_s_var = torch.isin(rule_heads[:, 1:2], vars_idx_tensor.view(-1))     # (n_rules, 1)
    is_head_o_var = torch.isin(rule_heads[:, 2:3], vars_idx_tensor.view(-1))     # (n_rules, 1)

    # --- Perform MGU Calculation (Iterative per Query-Rule Pair) ---
    potential_subs_list = [[[] for _ in range(n_rules)] for _ in range(bs)]
    potential_valid = torch.zeros((bs, n_rules), dtype=torch.bool, device=effective_device)
    max_binding_depth = max_subs_per_rule * 2 + 10 # Safety depth for get_binding

    for b in range(bs):
        if not is_query_valid[b]: continue # Skip padding queries
        if not torch.any(pred_match[b]): continue # Skip if no predicate matches for this query

        query = first_queries[b]
        q_p, q_s, q_o = query.tolist()
        is_q_s_var_b = is_query_s_var[b].item()
        is_q_o_var_b = is_query_o_var[b].item()

        # --- DEBUG --- Print initial query for the batch item if verbose
        if prover_verbose > 0 and b == 0: # Print only for first batch item
             print(f"--- MGU Start B={b}: Query = {_format_atom(query, predicate_idx2str, constant_idx2str, vars_idx_set, variable_idx2str)} ---")


        for r in range(n_rules):
            if not pred_match[b, r]: continue # Skip if predicate doesn't match

            head = rule_heads[r]
            h_p, h_s, h_o = head.tolist()
            is_h_s_var_r = is_head_s_var[r].item()
            is_h_o_var_r = is_head_o_var[r].item()

            # --- DEBUG --- Print Rule Head being attempted
            # Print only for first batch item for readability
            if prover_verbose > 1 and b == 0 :
                print(f"\nDEBUG: Unify B={b}, R={r} [Head: {_format_atom(head, predicate_idx2str, constant_idx2str, vars_idx_set, variable_idx2str)}]")

            temp_subs = {} # Substitution dictionary for this query-rule pair
            possible = True # Flag to track if unification is still possible
            term_pairs = [(q_s, h_s, is_q_s_var_b, is_h_s_var_r),
                          (q_o, h_o, is_q_o_var_b, is_h_o_var_r)]

            # --- Unify Subject and Object pairs ---
            term_pair_idx = 0 # Track S/O pair
            for q_term, h_term, is_q_var, is_h_var in term_pairs:
                if not possible: break
                pair_label = "Subject" if term_pair_idx == 0 else "Object"

                # --- DEBUG --- Print current pair
                if prover_verbose > 1 and b == 0:
                    q_term_str = _format_term(q_term, vars_idx_set, constant_idx2str, variable_idx2str)
                    h_term_str = _format_term(h_term, vars_idx_set, constant_idx2str, variable_idx2str)
                    print(f"  [{pair_label} Pair]: q={q_term_str}({q_term}), h={h_term_str}({h_term}), is_q_var={is_q_var}, is_h_var={is_h_var}")

                # 1. Find ultimate bindings using the *current* substitutions
                q_final = get_binding(q_term, temp_subs, set(), vars_idx_set, max_binding_depth) if is_q_var else q_term
                h_final = get_binding(h_term, temp_subs, set(), vars_idx_set, max_binding_depth) if is_h_var else h_term

                # Check for cycles detected during resolution
                if q_final is None or h_final is None:
                    if prover_verbose > 1 and b == 0: print(f"    Cycle/Depth detected during resolution!")
                    possible = False; break

                # Check if they are already the same after resolution
                if q_final == h_final:
                    if prover_verbose > 1 and b == 0: print(f"    Resolved: q_final={q_final}, h_final={h_final}. Terms match, continue.")
                    term_pair_idx += 1
                    continue

                is_q_final_var = q_final in vars_idx_set
                is_h_final_var = h_final in vars_idx_set

                # --- DEBUG --- Print resolved terms
                if prover_verbose > 1 and b == 0:
                    q_final_str = _format_term(q_final, vars_idx_set, constant_idx2str, variable_idx2str)
                    h_final_str = _format_term(h_final, vars_idx_set, constant_idx2str, variable_idx2str)
                    print(f"    Resolved: q_final={q_final_str}({q_final}), h_final={h_final_str}({h_final})")
                    print(f"    is_q_final_var={is_q_final_var}, is_h_final_var={is_h_final_var}")

                # 2. Handle unification based on variable/constant status
                if is_q_final_var: # Case 1: Query term resolves to a variable
                    # Occurs Check: Ensure q_final does not occur in h_final
                    if occurs_check(q_final, h_final, temp_subs, vars_idx_set, max_binding_depth):
                        if prover_verbose > 1 and b==0: print(f"    Occurs Check Failed: {_format_term(q_final, vars_idx_set, None, variable_idx2str)} in {_format_term(h_final, vars_idx_set, constant_idx2str, variable_idx2str)}")
                        possible = False; break
                    if prover_verbose > 1 and b==0: print(f"    Binding q_final:{_format_term(q_final, vars_idx_set, None, variable_idx2str)}({q_final}) -> h_final:{_format_term(h_final, vars_idx_set, constant_idx2str, variable_idx2str)}({h_final})")
                    temp_subs[q_final] = h_final # Bind query var to head term resolution

                elif is_h_final_var: # Case 2: Head term resolves to a variable (and query term didn't)
                    # Occurs Check: Ensure h_final does not occur in q_final
                    if occurs_check(h_final, q_final, temp_subs, vars_idx_set, max_binding_depth):
                        if prover_verbose > 1 and b==0: print(f"    Occurs Check Failed: {_format_term(h_final, vars_idx_set, None, variable_idx2str)} in {_format_term(q_final, vars_idx_set, constant_idx2str, variable_idx2str)}")
                        possible = False; break
                    if prover_verbose > 1 and b==0: print(f"    Binding h_final:{_format_term(h_final, vars_idx_set, None, variable_idx2str)}({h_final}) -> q_final:{_format_term(q_final, vars_idx_set, constant_idx2str, variable_idx2str)}({q_final})")
                    temp_subs[h_final] = q_final # Bind head var to query term resolution

                else: # Case 3: Both resolve to constants, but they are different
                    if prover_verbose > 1 and b==0: print(f"    Constant Mismatch: {_format_term(q_final, vars_idx_set, constant_idx2str, variable_idx2str)} != {_format_term(h_final, vars_idx_set, constant_idx2str, variable_idx2str)}")
                    possible = False; break

                # --- DEBUG --- Print current state after pair processing
                if prover_verbose > 1 and b == 0 :
                    subs_str = {_format_term(k, vars_idx_set, None, variable_idx2str): _format_term(v, vars_idx_set, constant_idx2str, variable_idx2str) for k,v in temp_subs.items()}
                    print(f"    -> Possible: {possible}, New temp_subs: {subs_str}")

                term_pair_idx += 1
            # --- END Inner Loop ---

            # --- DEBUG --- Print before finalization
            if prover_verbose > 1 and b == 0 :
                print(f"  End Pairs Loop. Possible: {possible}")

            # --- Finalize Substitutions for this pair ---
            if possible:
                final_subs_list = []
                processed_vars = set()
                possible_final = True
                # Iterate through variables that got *initially* bound in temp_subs
                current_keys = list(temp_subs.keys()) # Capture keys before modification? Or just use temp_subs? Use temp_subs directly.

                if prover_verbose > 1 and b == 0: print(f"  Attempting Finalization. Initial Keys: {list(temp_subs.keys())}")

                # Iterate through the variables that were actually assigned something in temp_subs
                for var_start in list(temp_subs.keys()): # Iterate over potentially changing dict keys
                    if var_start not in vars_idx_set: continue # Should not happen if keys are vars

                    if var_start in processed_vars: continue

                    # Find the *ultimate* binding using the *final* temp_subs
                    final_val = get_binding(var_start, temp_subs, set(), vars_idx_set, max_binding_depth)

                    # --- DEBUG --- Print finalization step
                    if prover_verbose > 1 and b == 0:
                         var_start_str = variable_idx2str.get(var_start,"?")
                         final_val_str = _format_term(final_val, vars_idx_set, constant_idx2str, variable_idx2str) if final_val is not None else "None"
                         print(f"    Finalizing Key: {var_start_str}({var_start}) -> Resolved: {final_val_str}({final_val})")


                    if final_val is None: # Cycle detected during finalization
                        if prover_verbose > 1 and b == 0: print(f"      Cycle/Depth detected during finalization!")
                        possible_final = False; break

                    # Only add substitution if the var ultimately binds to something different
                    if var_start != final_val:
                        if prover_verbose > 1 and b == 0: print(f"      Adding Substitution: [{var_start}, {final_val}]")
                        final_subs_list.append([var_start, final_val])

                    processed_vars.add(var_start)
                    # Mark the final value as processed too if it's a variable that might be a key
                    if final_val in vars_idx_set:
                        processed_vars.add(final_val)

                # Check substitution count limit *after* creating the minimal set
                if possible_final and len(final_subs_list) <= max_subs_per_rule:
                    if prover_verbose > 1 and b == 0: print(f"  Finalization SUCCEEDED for R={r}. Num Subs: {len(final_subs_list)}")
                    potential_valid[b, r] = True
                    potential_subs_list[b][r] = final_subs_list
                # Optional: Log why it failed if verbose
                elif prover_verbose > 1 and b == 0:
                    reason = "Cycle/Depth" if not possible_final else f"Too many subs ({len(final_subs_list)} > {max_subs_per_rule})"
                    print(f"  Finalization FAILED for R={r}. Reason: {reason}")
            # --- END Finalization Step ---

        # --- DEBUG --- Print end of rule checks for the batch item
        if prover_verbose > 0 and b == 0: # Print only for first batch item
             valid_rules_indices = torch.where(potential_valid[b])[0].tolist()
             print(f"--- MGU End B={b}: Valid Rule Indices = {valid_rules_indices} ---")


    # --- Select Top K Valid Unifications ---
    if actual_k == 0: # No rules to unify with, or k=0
        return UnificationResult(substitutions_out, rule_indices_out, valid_mask_out, rule_bodies_out)

    # Score valid unifications (1.0) higher than invalid ones (-inf)
    scores = torch.where(potential_valid,
                         torch.tensor(1.0, device=effective_device),
                         torch.tensor(float('-inf'), device=effective_device))

    # Handle case where there are fewer valid unifications than k
    num_valid_per_batch = potential_valid.sum(dim=1)
    if torch.all(num_valid_per_batch == 0): # No valid unifications found at all
        return UnificationResult(substitutions_out, rule_indices_out, valid_mask_out, rule_bodies_out)

    # Ensure k does not exceed the number of potential rules
    k_for_topk = min(actual_k, n_rules)
    if k_for_topk == 0: # Should be caught above, but safety
        return UnificationResult(substitutions_out, rule_indices_out, valid_mask_out, rule_bodies_out)

    try:
        # Find the indices of the top k valid rules for each batch item
        top_scores, top_indices_in_rules = torch.topk(scores, k=k_for_topk, dim=1)
    except RuntimeError as e:
        logging.error(f"Error during rule topk: {e}. Scores shape: {scores.shape}, k_for_topk: {k_for_topk}, n_rules: {n_rules}")
        # Fallback if k > n_rules somehow slips through
        if scores.shape[1] < k_for_topk:
            k_for_topk = scores.shape[1]
            if k_for_topk == 0: return UnificationResult(substitutions_out, rule_indices_out, valid_mask_out, rule_bodies_out)
            top_scores, top_indices_in_rules = torch.topk(scores, k=k_for_topk, dim=1)
        else:
            return UnificationResult(substitutions_out, rule_indices_out, valid_mask_out, rule_bodies_out) # Unexpected error

    # Mask indicating which of the top-k slots correspond to actual valid unifications
    final_valid_k_mask = (top_scores > float('-inf')) # Shape: (bs, k_for_topk)

    # --- Gather Results for Top K ---
    # Get batch and k coordinates of the final valid unifications
    final_valid_coords = torch.nonzero(final_valid_k_mask, as_tuple=True)
    n_final_valid = len(final_valid_coords[0])

    if n_final_valid > 0:
        final_bs_indices, final_k_indices = final_valid_coords
        # Get the original rule indices corresponding to these top-k slots
        final_rule_indices = top_indices_in_rules[final_valid_coords] # Shape: (n_final_valid,)

        # --- Populate Output Tensors ---
        # 1. Rule Indices
        rule_indices_out[final_bs_indices, final_k_indices] = final_rule_indices

        # 2. Rule Bodies (if they exist)
        if rule_body_len > 0:
            # Gather the bodies corresponding to the chosen rules
            gathered_bodies = rule_bodies[final_rule_indices] # Shape: (n_final_valid, rule_body_len, 3)
            # Place them into the output tensor at the correct (bs, k) locations
            rule_bodies_out[final_bs_indices, final_k_indices] = gathered_bodies

        # 3. Substitutions
        # Create a temporary tensor to hold substitutions before scattering
        temp_subs_tensor = torch.full((n_final_valid, max_subs_per_rule, 2), PADDING_VALUE, dtype=torch.long, device=effective_device)
        for i in range(n_final_valid):
            b_final = final_bs_indices[i].item()
            rule_idx_final = final_rule_indices[i].item()
            # Retrieve the pre-calculated substitution list for this valid unification
            subs_list = potential_subs_list[b_final][rule_idx_final] # Use final indices b_final, rule_idx_final
            num_subs = len(subs_list)
            if num_subs > 0:
                # Convert list to tensor and place in the temporary tensor
                subs_tensor_entry = torch.tensor(subs_list, dtype=torch.long, device=effective_device)
                temp_subs_tensor[i, :num_subs, :] = subs_tensor_entry

        # Scatter the substitutions into the output tensor at the correct (bs, k) locations
        substitutions_out[final_bs_indices, final_k_indices] = temp_subs_tensor

    # Populate the final output mask (up to k_for_topk)
    valid_mask_out[:, :k_for_topk] = final_valid_k_mask

    # --- DEBUG --- Print final valid mask for batch 0
    if prover_verbose > 1 and b == 0:
        print(f"DEBUG B=0 Final valid_mask_out[0, :k_for_topk]: {valid_mask_out[0, :k_for_topk]}")


    # Keep original verbose flag for this higher-level debug message
    if verbose and prover_verbose > 0: # Use main verbose flag for this summary
        print("-" * 40 + f"\nDEBUG: batch_unify_with_rules (End - B={bs}) - Valid Counts: {potential_valid.sum(dim=1).tolist()}\n" + "=" * 40)

    return UnificationResult(substitutions_out, rule_indices_out, valid_mask_out, rule_bodies_out)



# --- Modified apply_substitutions_and_create_next_state (Keep as is) ---
class NextStateResult(NamedTuple):
    """Structure to hold results from creating next states."""
    next_states: torch.Tensor      # Shape: (bs, k, new_max_atoms, 3)
    is_proven_mask: torch.Tensor   # Shape: (bs, k) - True if the state represents a proof

def apply_substitutions_and_create_next_state(
    states_idx: torch.Tensor,             # Shape: (bs, 1, n_padding_atoms, 3)
    unification_result: UnificationResult,# Result from fact or rule unification
    vars_idx_tensor: torch.Tensor,        # 1D Tensor of variable indices
    new_max_atoms: Optional[int] = None,  # Max atoms in the output state
    max_sub_iterations: int = 10          # Iteration limit for apply_substitutions_vectorized
) -> NextStateResult: # Modified return type
    """
    Generates next goal states based on successful unifications (fact or rule).
    Applies substitutions vectorially to remaining original goals and (if applicable) rule bodies.
    Concatenates results, pads/truncates, and determines if a state represents a full proof.
    Output shapes in NextStateResult.
    """
    bs, _, n_padding_atoms, _ = states_idx.shape
    k = unification_result.substitutions.shape[1]
    device = states_idx.device

    substitutions = unification_result.substitutions # (bs, k, max_subs, 2)
    valid_mask = unification_result.valid_mask       # (bs, k)
    rule_bodies = unification_result.target_bodies   # (bs, k, rule_body_len, 3) or None

    has_rule_bodies = rule_bodies is not None and rule_bodies.numel() > 0 and rule_bodies.shape[-1] == 3
    rule_body_len = rule_bodies.shape[2] if has_rule_bodies else 0

    # Determine size of remaining original goals
    num_remaining_atoms_orig = max(0, n_padding_atoms - 1)
    if new_max_atoms is None: # Default output size based on inputs
        new_max_atoms = num_remaining_atoms_orig + rule_body_len
    if new_max_atoms < 0: new_max_atoms = 0 # Ensure non-negative

    # Initialize output tensors
    next_states_out = torch.full((bs, k, new_max_atoms, 3), PADDING_VALUE, dtype=torch.long, device=device)
    is_proven_mask_out = torch.zeros((bs, k), dtype=torch.bool, device=device) # Initialize proven mask

    # Extract remaining goals (all but the first atom) from the original state
    # Handle case where there are no remaining goals (n_padding_atoms <= 1)
    remaining_queries_base = states_idx[:, 0, 1:, :] if n_padding_atoms > 1 else torch.empty((bs, 0, 3), dtype=torch.long, device=device)

    # --- Process only valid unifications ---
    valid_coords = torch.nonzero(valid_mask, as_tuple=True)
    n_valid = len(valid_coords[0])

    if n_valid == 0: # No valid unifications, return empty/padded results
        return NextStateResult(next_states_out, is_proven_mask_out)

    # Gather data corresponding to valid unifications
    valid_bs_indices, valid_k_indices = valid_coords
    valid_subs = substitutions[valid_bs_indices, valid_k_indices] # (n_valid, max_subs, 2)
    valid_rem_q = remaining_queries_base[valid_bs_indices]        # (n_valid, num_remaining_atoms_orig, 3)
    # Gather rule bodies if they exist, otherwise use empty tensor
    valid_bodies = rule_bodies[valid_bs_indices, valid_k_indices] if has_rule_bodies else torch.empty((n_valid, 0, 3), dtype=torch.long, device=device) # (n_valid, rule_body_len, 3)

    # --- Apply Substitutions (Vectorized) ---
    # Unsqueeze substitutions to allow broadcasting over the atoms dimension
    # Shape required by apply_substitutions_vectorized: (..., n_atoms, 3) for atoms
    # Shape required for substitutions: (..., max_subs, 2) - needs to broadcast over leading dims of atoms
    # Here, atoms have shape (n_valid, n_atoms, 3), subs have (n_valid, max_subs, 2)
    # No unsqueezing needed if apply_substitutions_vectorized handles broadcasting correctly.

    # Apply subs to remaining original query atoms
    subst_rem_q_flat = apply_substitutions_vectorized(
        valid_rem_q, valid_subs, vars_idx_tensor, max_sub_iterations
    ) # Shape: (n_valid, num_remaining_atoms_orig, 3)

    # Apply subs to rule body atoms (if any)
    subst_body_flat = torch.empty((n_valid, 0, 3), dtype=torch.long, device=device)
    if rule_body_len > 0:
        subst_body_flat = apply_substitutions_vectorized(
            valid_bodies, valid_subs, vars_idx_tensor, max_sub_iterations
        ) # Shape: (n_valid, rule_body_len, 3)

    # --- Filter Padding, Concatenate, Check Proven, Pad/Truncate (Iterative Part) ---
    # Need to iterate through the valid results because filtering padding and concatenating
    # changes the number of atoms per result, making pure vectorization difficult.
    is_proven_flat = torch.zeros(n_valid, dtype=torch.bool, device=device) # Track proof status for valid items

    for i in range(n_valid):
        b = valid_bs_indices[i].item()    # Original batch index
        k_idx = valid_k_indices[i].item() # Original k index

        # Get substituted atoms for this specific valid unification
        current_body = subst_body_flat[i] # (rule_body_len, 3)
        current_rem_q = subst_rem_q_flat[i] # (num_remaining_atoms_orig, 3)

        # Filter out padding atoms introduced by substitution or original padding
        body_keep_mask = torch.any(current_body != PADDING_VALUE, dim=-1) # (rule_body_len,)
        rem_q_keep_mask = torch.any(current_rem_q != PADDING_VALUE, dim=-1)# (num_remaining_atoms_orig,)

        subst_body_filtered = current_body[body_keep_mask]    # (num_actual_body_atoms, 3)
        subst_rem_q_filtered = current_rem_q[rem_q_keep_mask] # (num_actual_rem_q_atoms, 3)

        # Concatenate the filtered atoms: new goals = substituted body + substituted remaining query
        concatenated_atoms = torch.cat((subst_body_filtered, subst_rem_q_filtered), dim=0)

        # --- Check if this state represents a proof ---
        # Proof is complete if there are no remaining atoms after concatenation
        if concatenated_atoms.shape[0] == 0:
            is_proven_flat[i] = True # Mark this unification result as leading to a proof

        # --- Pad or Truncate to fit `new_max_atoms` ---
        current_len = concatenated_atoms.shape[0]
        len_to_copy = min(current_len, new_max_atoms)
        if len_to_copy > 0:
            # Copy the resulting atoms into the output tensor at the correct (b, k_idx) location
            next_states_out[b, k_idx, :len_to_copy, :] = concatenated_atoms[:len_to_copy, :]
        # If current_len > new_max_atoms, truncation occurs implicitly.
        # If current_len < new_max_atoms, the rest remains PADDING_VALUE.

    # Scatter the proof status back to the output mask using the original valid coordinates
    is_proven_mask_out[valid_bs_indices, valid_k_indices] = is_proven_flat

    return NextStateResult(next_states_out, is_proven_mask_out)



# --- Example Usage (Updated to use vectorized functions) ---
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu') # Force CPU for debugging if needed
    print(f"Using device: {device}")

    K_FACTS = 3 # Max unifications from facts
    K_RULES = 2 # Max unifications from rules
    MAX_SUBS_PER_RULE = 5 # Limit substitutions stored for rule MGU
    OUTPUT_MAX_ATOMS = 5 # Define a fixed max size for generated states for consistency
    MAX_SUB_ITER = 5 # Max iterations for substitution propagation
    PROVER_VERBOSE_LEVEL = 2 # <<< Set Rule unification verbosity here (0=off, 1=basic, 2=detailed)

    print(f"Using PADDING_VALUE: {PADDING_VALUE}")
    print(f"Using UNBOUND_VAR: {UNBOUND_VAR}")

    # --- Mappings & Variables ---
    predicate_map = {1: "relatedTo", 2: "typeOf", 3: "hasProperty", 4: "parent", 5: "ancestor"}
    constant_map = {10: "ObjA", 11: "ObjB", 12: "ObjC", 13: "ObjD", 14: "ObjE", 15: "ObjF",
                    20: "PropX", 21: "PropY", 22: "PropZ", 30: "Person1", 31: "Person2", 32: "Person3"}
    vars_idx_list = [-1, -2, -3, -10, -11, -12] # Combined query and rule vars
    vars_set, vars_tensor = _get_var_set_and_tensor(vars_idx_list, device)
    # Create reverse map for variables for better printing
    variable_map = {idx: f"Var({idx})" for idx in vars_idx_list} # Simple map for demo

    # --- States ---
    # Shape: (bs, 1, n_atoms, 3) - Add the singleton dimension expected by functions
    states = torch.tensor([
        [[[1, -1, 10], [2, 11, -2], [PADDING_VALUE]*3, [PADDING_VALUE]*3]], # S0: (relatedTo, V1, A), (typeOf, B, V2)
        [[[5, 30, -1], [1, -1, 12], [PADDING_VALUE]*3, [PADDING_VALUE]*3]], # S1: (ancestor, P1, V1), (relatedTo, V1, C)
        [[[2, -1, -2], [1, -1, 10], [PADDING_VALUE]*3, [PADDING_VALUE]*3]], # S2: (typeOf, V1, V2), (relatedTo, V1, A)
        [[[3, -1, -1], [2, -1, 12], [PADDING_VALUE]*3, [PADDING_VALUE]*3]], # S3: (hasProp, V1, V1), (typeOf, V1, C) -> Conflict check test
        [[[PADDING_VALUE]*3, [PADDING_VALUE]*3, [PADDING_VALUE]*3, [PADDING_VALUE]*3]]  # S4: Padding query
    ], dtype=torch.long, device=device)
    # Replace 0,0,0 with padding for clarity
    states[states == 0] = PADDING_VALUE
    bs, _, n_atoms_state, _ = states.shape
    print(f"State shape: {states.shape}")


    # --- Facts ---
    facts_tensor = torch.tensor([
        [1, 11, 10], # F0: (relatedTo, B, A) -> Unifies S0 {-1: B}
        [1, 12, 13], # F1: (relatedTo, C, D)
        [1, 14, 10], # F2: (relatedTo, E, A) -> Unifies S0 {-1: E}
        [2, 10, 11], # F3: (typeOf, A, B)   -> Unifies S2 {-1: A, -2: B}
        [2, 12, 12], # F4: (typeOf, C, C)   -> Unifies S2 {-1: C, -2: C}
        [1, 15, 10], # F5: (relatedTo, F, A) -> Unifies S0 {-1: F}
        [3, 20, 20], # F6: (hasProp, X, X)   -> Unifies S3 {-1: X}
        [3, 21, 22], # F7: (hasProp, Y, Z)   -> Conflicts S3 (VarX=Y vs VarX=Z) -> Should be filtered by fact unify conflict check
        [4, 30, 31], # F8: (parent, P1, P2) -> Used by rules
        [5, 31, 32]  # F9: (ancestor, P2, P3) -> Base case for rule R1
    ], dtype=torch.long, device=device)
    print(f"Facts shape: {facts_tensor.shape}")

    # --- Rules ---
    # Padded to same number of atoms (Head + Max Body)
    rules_tensor = torch.tensor([
        # R0: ancestor(X, Y) :- parent(X, Y)
        [[5, -10, -11], [4, -10, -11], [PADDING_VALUE, PADDING_VALUE, PADDING_VALUE]], # Padded to 3 atoms
        # R1: ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z)
        [[5, -10, -12], [4, -10, -11], [5, -11, -12]]
    ], dtype=torch.long, device=device)
    _, n_atoms_rule, _ = rules_tensor.shape
    print(f"Rules shape: {rules_tensor.shape}")


    # --- Run Fact Unification ---
    print("\n" + "#" * 20 + " FACT UNIFICATION " + "#" * 20)
    fact_unification_result = batch_unify_with_facts(
        states, facts_tensor, vars_tensor, k=K_FACTS, device=device,
        verbose=False, # Keep basic verbose off for fact step unless needed
        predicate_idx2str=predicate_map,
        constant_idx2str=constant_map,
        variable_idx2str=variable_map # Pass var map
    )

    # --- Apply Fact Substitutions ---
    print("\n" + "#" * 20 + " APPLYING FACT SUBSTITUTIONS " + "#" * 20)
    next_states_from_facts = apply_substitutions_and_create_next_state(
        states,
        fact_unification_result, # Pass the whole result object
        vars_tensor,
        new_max_atoms=OUTPUT_MAX_ATOMS, # Pad/truncate result
        max_sub_iterations=MAX_SUB_ITER
    )

    # --- Run Rule Unification ---
    print("\n" + "#" * 20 + " RULE UNIFICATION " + "#" * 20)
    rule_unification_result = batch_unify_with_rules(
        states, rules_tensor, vars_tensor, k=K_RULES,
        max_subs_per_rule=MAX_SUBS_PER_RULE,
        device=device,
        verbose=False, # Keep basic verbose off for rule step unless needed
        prover_verbose=PROVER_VERBOSE_LEVEL, # <<< Use detailed verbosity level here
        predicate_idx2str=predicate_map,
        constant_idx2str=constant_map,
        variable_idx2str=variable_map # Pass var map
    )

    # --- Apply Rule Substitutions ---
    print("\n" + "#" * 20 + " APPLYING RULE SUBSTITUTIONS " + "#" * 20)
    next_states_from_rules = apply_substitutions_and_create_next_state(
        states,
        rule_unification_result, # Pass the whole result object
        vars_tensor,
        new_max_atoms=OUTPUT_MAX_ATOMS, # Pad/truncate result
        max_sub_iterations=MAX_SUB_ITER
    )

    # ==========================================================================
    # START OF MODIFIED SECTION
    # ==========================================================================
    print("\n" + "=" * 20 + " INTERPRETED RESULTS " + "=" * 20)
    for b in range(bs):
        query_atom = states[b, 0, 0, :]
        query_str = _format_atom(query_atom, predicate_map, constant_map, vars_set, variable_map)
        print(f"\n--- State {b}: Original First Query = {query_str} ---")

        # Print original remaining queries
        print(f"  Original Remaining Queries:")
        has_rem_query = False
        for atom_idx in range(1, n_atoms_state):
            rem_atom = states[b, 0, atom_idx, :]
            # Use only PADDING_VALUE check
            if torch.any(rem_atom != PADDING_VALUE):
                print(f"     {_format_atom(rem_atom, predicate_map, constant_map, vars_set, variable_map)}")
                has_rem_query = True
        if not has_rem_query: print("     [None]")

        is_query_padding = torch.all(query_atom == PADDING_VALUE)
        if is_query_padding:
            print("  Original query is padding. Skipping results.")
            continue

        # --- Print Fact Results ---
        print(f"\n  Fact Unification Results (Top {K_FACTS}):")
        found_fact_unification = False
        if b < fact_unification_result.valid_mask.shape[0]:
            for k_idx in range(min(K_FACTS, fact_unification_result.valid_mask.shape[1])):
                if fact_unification_result.valid_mask[b, k_idx]:
                    found_fact_unification = True
                    fact_index = fact_unification_result.target_indices[b, k_idx].item()
                    subs_tensor = fact_unification_result.substitutions[b, k_idx] # Shape (2, 2)

                    print(f"   - Fact Slot {k_idx}:")
                    if fact_index != PADDING_VALUE and 0 <= fact_index < len(facts_tensor):
                        fact_atom = facts_tensor[fact_index]
                        fact_str = _format_atom(fact_atom, predicate_map, constant_map, set(), variable_map) # Facts have no vars
                        print(f"     Unified with Fact {fact_index}: {fact_str}")
                    else:
                        print(f"     Unified with Fact Index: {fact_index} [Invalid or Padding]")

                    print(f"     Substitutions:")
                    subs_found = False
                    for sub_pair in subs_tensor: # Iterate through the pairs [Var, Val]
                        if sub_pair[0].item() != PADDING_VALUE:
                            sub_str = _format_substitution(sub_pair, constant_map, vars_set, variable_map)
                            print(f"       {sub_str}")
                            subs_found = True
                    if not subs_found: print("       [None]")

                    # Print the resulting next state from facts
                    print(f"     Next State Queries:")
                    has_next_query = False
                    if b < next_states_from_facts.next_states.shape[0] and k_idx < next_states_from_facts.next_states.shape[1]:
                        for atom_idx in range(next_states_from_facts.next_states.shape[2]):
                            next_atom = next_states_from_facts.next_states[b, k_idx, atom_idx, :]
                            if torch.any(next_atom != PADDING_VALUE):
                                print(f"       {_format_atom(next_atom, predicate_map, constant_map, vars_set, variable_map)}")
                                has_next_query = True
                    if not has_next_query: print("       [Proven or Empty State]")

            if not found_fact_unification:
                print("   [None found]")

        # --- Print Rule Results ---
        print(f"\n  Rule Unification Results (Top {K_RULES}):")
        found_rule_unification = False
        if b < rule_unification_result.valid_mask.shape[0]:
            for k_idx in range(min(K_RULES, rule_unification_result.valid_mask.shape[1])):
                if rule_unification_result.valid_mask[b, k_idx]:
                    found_rule_unification = True
                    rule_index = rule_unification_result.target_indices[b, k_idx].item()
                    subs_tensor = rule_unification_result.substitutions[b, k_idx] # Shape (max_subs_per_rule, 2)

                    print(f"   - Rule Slot {k_idx}:")
                    if rule_index != PADDING_VALUE and 0 <= rule_index < len(rules_tensor):
                        rule_head_atom = rules_tensor[rule_index, 0, :] # Get head atom
                        rule_head_str = _format_atom(rule_head_atom, predicate_map, constant_map, vars_set, variable_map)
                        print(f"     Unified with Rule {rule_index} Head: {rule_head_str}")
                    else:
                        print(f"     Unified with Rule Index: {rule_index} [Invalid or Padding]")

                    print(f"     Substitutions (MGU):")
                    subs_found = False
                    for sub_pair in subs_tensor: # Iterate through the pairs [Var, Val]
                        if sub_pair[0].item() != PADDING_VALUE:
                            sub_str = _format_substitution(sub_pair, constant_map, vars_set, variable_map)
                            print(f"       {sub_str}")
                            subs_found = True
                    if not subs_found: print("       [None]")


                    # Print the resulting next state from rules
                    print(f"     Next State Queries:")
                    has_next_query = False
                    if b < next_states_from_rules.next_states.shape[0] and k_idx < next_states_from_rules.next_states.shape[1]:
                        for atom_idx in range(next_states_from_rules.next_states.shape[2]):
                            next_atom = next_states_from_rules.next_states[b, k_idx, atom_idx, :]
                            if torch.any(next_atom != PADDING_VALUE):
                                print(f"       {_format_atom(next_atom, predicate_map, constant_map, vars_set, variable_map)}")
                                has_next_query = True
                    if not has_next_query: print("       [Proven or Empty State]")

            if not found_rule_unification:
                print("   [None found]")
    # ==========================================================================
    # END OF MODIFIED SECTION
    # ==========================================================================