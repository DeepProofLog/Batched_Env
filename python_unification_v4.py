import torch
from typing import Tuple, Optional, Union, List, Set, Dict, NamedTuple
import logging

# --- Constants and Dummy Classes ---
PADDING_VALUE = -999
UNBOUND_VAR = -998 # Although not used in this specific substitution logic, good to keep defined
class IndexManager: pass # Dummy

class SimplifiedFactResult(NamedTuple):
    """Holds the results of fact unification structured per k-slot."""
    valid_mask: torch.Tensor      # Shape: (bs, k) - True if a valid unification exists for this slot
    fact_indices: torch.Tensor    # Shape: (bs, k) - Index of the fact for this slot (or PAD)
    substitutions: torch.Tensor   # Shape: (bs, k, 2, 2) - Substitutions [var, val] for S/O terms (or PAD)


# --- Helper Function ---
def _get_var_set_and_tensor(
    vars_idx: Union[torch.Tensor, List[int], Set[int], IndexManager],
    device: torch.device
) -> Tuple[Set[int], torch.Tensor]:
    """Ensures vars_idx is a set and a tensor on the correct device."""
    # --- Placeholder Implementation ---
    if isinstance(vars_idx, IndexManager):
        vars_idx_list = [-1, -2] # Example
        vars_idx_set = set(vars_idx_list)
        vars_idx_tensor = torch.tensor(vars_idx_list, dtype=torch.long, device=device)
        return vars_idx_set, vars_idx_tensor

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
    # Ensure padding/unbound are not treated as variables for substitution lookup
    vars_idx_set = {v for v in vars_idx_set if v not in (PADDING_VALUE, UNBOUND_VAR)}
    vars_idx_tensor = vars_idx_tensor[(vars_idx_tensor != PADDING_VALUE) & (vars_idx_tensor != UNBOUND_VAR)]
    if vars_idx_tensor.numel() == 0: # Handle empty var list after filtering
        vars_idx_set = set()
    return vars_idx_set, vars_idx_tensor
# --- Core Term Unification Function (No Top-K) ---
class TermUnificationResult(NamedTuple):
    unification_mask: torch.Tensor # Shape: (bs, n_facts) - True if terms unify (incl. conflict check)
    all_substitutions: torch.Tensor # Shape: (bs, n_facts, m, 2) - Potential [var, val] subs for all pairs

def check_term_unification(
    t1: torch.Tensor,             # Shape: (bs, n_facts, m) - Query side terms, broadcasted
    t2: torch.Tensor,             # Shape: (bs, n_facts, m) - Fact side terms, broadcasted
    vars_idx_tensor: torch.Tensor,# Shape: (num_vars,) - Tensor of variable indices
    padding_value: int = PADDING_VALUE,
) -> TermUnificationResult:
    """
    Checks unification between two sets of terms t1 and t2 across bs and n_facts dimensions.
    Calculates a mask indicating valid unifications and all potential substitutions.
    Does NOT perform top-k selection.

    Args:
        t1: Tensor of query-side terms, shape (bs, n_facts, m).
        t2: Tensor of fact-side terms, shape (bs, n_facts, m).
        vars_idx_tensor: 1D Tensor containing indices that represent variables.
        padding_value: Integer value used for padding.

    Returns:
        TermUnificationResult containing:
            unification_mask: Boolean tensor (bs, n_facts).
            all_substitutions: Tensor (bs, n_facts, m, 2) of [var, val] pairs.
    """
    if t1.shape != t2.shape:
        raise ValueError(f"t1 and t2 must have the same shape, got {t1.shape} and {t2.shape}")
    if t1.dim() != 3:
        raise ValueError(f"t1 and t2 must have 3 dimensions (bs, n_facts, m), got {t1.dim()}")
    if vars_idx_tensor.dim() != 1:
         raise ValueError(f"vars_idx_tensor must be 1D, got {vars_idx_tensor.dim()}")

    bs, n_facts, m = t1.shape
    device = t1.device

    # --- 1. Check if t1 terms are variables ---
    # is_var_mask has shape (bs, n_facts, m)
    is_var_mask = torch.isin(t1, vars_idx_tensor)

    # --- 2. Check element-wise match condition ---
    # Either t1 is a variable OR t1 equals t2
    # element_match_mask has shape (bs, n_facts, m)
    element_match_mask = is_var_mask | (t1 == t2)

    # --- 3. Check overall term match (all m elements must match) ---
    # term_match_mask has shape (bs, n_facts)
    term_match_mask = torch.all(element_match_mask, dim=-1)

    # --- 4. Check for conflicts ---
    # A conflict occurs if the same variable appears in different positions 'i' and 'j' in t1,
    # but the corresponding values in t2 are different.
    is_conflict = torch.zeros((bs, n_facts), dtype=torch.bool, device=device)
    if m > 1:
        # Iterate through unique pairs of term positions (i, j)
        for i in range(m):
            for j in range(i + 1, m):
                t1_i, t1_j = t1[:, :, i], t1[:, :, j]
                t2_i, t2_j = t2[:, :, i], t2[:, :, j]
                is_var_i = is_var_mask[:, :, i]
                is_var_j = is_var_mask[:, :, j]

                # Conflict for this pair (i, j)
                conflict_ij = (
                    is_var_i & is_var_j &          # Both t1[i] and t1[j] must be variables
                    (t1_i == t1_j) &               # They must be the *same* variable
                    (t2_i != t2_j)                 # But the corresponding t2 values must differ
                )
                is_conflict = is_conflict | conflict_ij # Accumulate conflicts

    # --- 5. Final unification mask ---
    # Terms unify if they match element-wise AND there's no conflict
    final_unification_mask = term_match_mask & ~is_conflict

    # --- 6. Calculate all potential substitutions ---
    # If t1[i] is a variable, substitution is [t1[i], t2[i]]
    # Otherwise, it's [pad, pad]
    # sub_var/sub_val have shape (bs, n_facts, m)
    sub_var = torch.where(is_var_mask, t1, torch.tensor(padding_value, dtype=torch.long, device=device))
    sub_val = torch.where(is_var_mask, t2, torch.tensor(padding_value, dtype=torch.long, device=device))

    # all_substitutions has shape (bs, n_facts, m, 2)
    all_substitutions = torch.stack([sub_var, sub_val], dim=-1)

    # Mask substitutions where the term unification itself failed
    # Expand final_unification_mask to (bs, n_facts, 1, 1) for broadcasting
    all_substitutions = torch.where(
        final_unification_mask.unsqueeze(-1).unsqueeze(-1),
        all_substitutions,
        torch.tensor(padding_value, dtype=torch.long, device=device)
    )


    return TermUnificationResult(
        unification_mask=final_unification_mask,
        all_substitutions=all_substitutions
    )
# --- END Core Term Unification Function ---


def get_first_k_fact_unifications_generalized(
    queries_idx: torch.Tensor,            # Shape: (bs, 3)
    facts: torch.Tensor,                  # Shape: (n_facts, 3)
    vars_idx: Union[torch.Tensor, List[int], Set[int], IndexManager], # Variable indices or IndexManager
    k: int,                               # Max number of unifications to return
    device: Optional[torch.device] = None,
    padding_value: int = PADDING_VALUE,
) -> SimplifiedFactResult:
    """
    Finds the first K unifying, non-conflicting facts for given queries (P, S, O)
    and returns the results structured in (bs, k) tensors. Uses generalized term unification internally.
    Includes fix for broadcasting error in top-k index finding. Removes internal prints.

    Args:
        queries_idx: Tensor of queries, shape (bs, 3).
        facts: Tensor of facts, shape (n_facts, 3).
        vars_idx: Indices representing variables.
        k: Max number of unifications per query.
        device: Torch device.
        padding_value: Value used for padding.

    Returns:
        SimplifiedFactResult containing valid_mask (bs, k), fact_indices (bs, k),
        and substitutions (bs, k, 2, 2) corresponding to S/O terms.
    """
    if k <= 0: raise ValueError("k must be a positive integer")
    if queries_idx.dim() != 2 or queries_idx.shape[1] != 3:
        raise ValueError(f"queries_idx must have shape (bs, 3), but got {queries_idx.shape}")
    if facts.dim() != 2 or facts.shape[1] != 3:
        raise ValueError(f"facts must have shape (n_facts, 3), but got {facts.shape}")


    effective_device = device if device else queries_idx.device
    queries_idx = queries_idx.to(effective_device)
    facts = facts.to(effective_device)
    # Assume _get_var_set_and_tensor is defined elsewhere and works correctly
    vars_idx_set, vars_idx_tensor = _get_var_set_and_tensor(vars_idx, effective_device)

    bs = queries_idx.shape[0]
    n_facts = facts.shape[0]
    actual_k = min(k, n_facts) if n_facts > 0 else 0 # Cannot retrieve more facts than available

    # --- Initialize Output Tensors (Final Size k) ---
    substitutions_out = torch.full((bs, k, 2, 2), padding_value, dtype=torch.long, device=effective_device) # m=2 for S/O
    fact_indices_out = torch.full((bs, k), padding_value, dtype=torch.long, device=effective_device)
    valid_mask_out = torch.zeros((bs, k), dtype=torch.bool, device=effective_device)

    # --- Handle Edge Cases ---
    if bs == 0 or k == 0: # If n_facts is 0, actual_k will be 0 and handled later
        return SimplifiedFactResult(valid_mask_out, fact_indices_out, substitutions_out)

    # --- Prepare Terms for Broadcasting ---
    is_query_valid = ~torch.all(queries_idx == padding_value, dim=-1) # Shape: (bs,)
    q_p, q_s, q_o = queries_idx[:, 0], queries_idx[:, 1], queries_idx[:, 2] # (bs,)

    # Handle case where there are no facts
    if n_facts == 0:
         return SimplifiedFactResult(valid_mask_out, fact_indices_out, substitutions_out)

    f_p, f_s, f_o = facts[:, 0], facts[:, 1], facts[:, 2]                   # (n_facts,)

    # --- 1. Check Predicate Match ---
    pred_match_mask = (q_p.unsqueeze(1) == f_p.unsqueeze(0)) # (bs, n_facts)

    # --- 2. Prepare S/O terms for check_term_unification ---
    query_so = torch.stack([q_s, q_o], dim=-1) # (bs, 2)
    facts_so = torch.stack([f_s, f_o], dim=-1) # (n_facts, 2)

    t1 = query_so.unsqueeze(1).expand(-1, n_facts, -1) # (bs, n_facts, 2)
    t2 = facts_so.unsqueeze(0).expand(bs, -1, -1)      # (bs, n_facts, 2)

    # --- 3. Check S/O Term Unification ---
    # Assume check_term_unification is defined elsewhere and works correctly
    term_result = check_term_unification(
        t1=t1,
        t2=t2,
        vars_idx_tensor=vars_idx_tensor,
        padding_value=padding_value
    ) # .unification_mask (bs, n_facts), .all_substitutions (bs, n_facts, 2, 2)

    # --- 4. Combine Predicate match with S/O term unification mask ---
    final_unifies_mask = pred_match_mask & term_result.unification_mask & is_query_valid.unsqueeze(1) # (bs, n_facts)

    # --- 5. Early Exit if No Matches or actual_k=0 ---
    if actual_k == 0 or not torch.any(final_unifies_mask):
        return SimplifiedFactResult(valid_mask_out, fact_indices_out, substitutions_out)

    # --- 6. Perform Top-K Selection on final_unifies_mask ---
    # --- 6.0. Create a mask for valid unifications ---
    cum_true = torch.cumsum(final_unifies_mask.long(), dim=1) # (bs, n_facts)
    num_valid_per_batch = cum_true[:, -1] # (bs,)

    k_indices_range = torch.arange(actual_k, device=effective_device) # (actual_k,)
    final_valid_mask_slice = num_valid_per_batch.unsqueeze(1) > k_indices_range.unsqueeze(0) # (bs, actual_k)

    # --- 6.1. Find the first k indices where unification is valid ---
    target_k_vals = torch.arange(1, actual_k + 1, device=effective_device).unsqueeze(0) # (1, actual_k)
    match_k_val = (cum_true.unsqueeze(1) >= target_k_vals.unsqueeze(-1)) # Shape: (bs, actual_k, n_facts)

    final_fact_indices_slice = torch.argmax(match_k_val.long(), dim=2) # (bs, actual_k)
    final_fact_indices_slice = torch.where(final_valid_mask_slice, final_fact_indices_slice, padding_value) # (bs, actual_k)

    # --- 7. Gather Substitutions for Top-K results ---
    bs_coords, k_coords = torch.meshgrid(torch.arange(bs, device=effective_device),
                                         torch.arange(actual_k, device=effective_device),
                                         indexing='ij') # (bs, actual_k)

    safe_fact_indices = torch.where(final_valid_mask_slice, final_fact_indices_slice, 0)
    gathered_subs_slice = term_result.all_substitutions[bs_coords, safe_fact_indices] # (bs, actual_k, 2, 2)

    gathered_subs_slice = torch.where(
        final_valid_mask_slice.unsqueeze(-1).unsqueeze(-1), # (bs, actual_k, 1, 1)
        gathered_subs_slice,
        torch.tensor(padding_value, device=effective_device, dtype=torch.long)
    )

    # --- 8. Populate Final Output Tensors (up to size k) ---
    valid_mask_out[:, :actual_k] = final_valid_mask_slice
    fact_indices_out[:, :actual_k] = final_fact_indices_slice
    substitutions_out[:, :actual_k] = gathered_subs_slice

    return SimplifiedFactResult(valid_mask_out, fact_indices_out, substitutions_out)
# --- END MODIFIED Main Unification Function ---

# --- Substitution Application Function (Unchanged Internally) ---
def apply_substitutions_to_states(
    states_idx: torch.Tensor,               # Shape: (bs, 1, n_padding_atoms, 3) <- Still needs full state
    unification_result: SimplifiedFactResult, # Result from unification step
    padding_value: int = PADDING_VALUE
) -> torch.Tensor:
    """
    Applies the substitutions found in `unification_result` to the remaining atoms
    (atoms 1 to n) in `states_idx`.

    Args:
        states_idx: The original state tensor (bs, 1, n_atoms, 3).
        unification_result: The SimplifiedFactResult containing valid_mask and substitutions
                           obtained by unifying the *first* atom of states_idx.
        padding_value: The value used for padding.

    Returns:
        A tensor of shape (bs, k, n_padding_atoms - 1, 3) representing
        the next states after applying substitutions for each of the k
        valid unifications found for the first atom. Invalid slots
        (where unification_result.valid_mask is False) are filled with padding_value.
    """
    bs, _, n_padding_atoms, _ = states_idx.shape
    if not isinstance(unification_result, SimplifiedFactResult):
        raise TypeError("unification_result must be a SimplifiedFactResult NamedTuple")
    k = unification_result.valid_mask.shape[1]
    device = states_idx.device

    # --- Handle edge case: No remaining atoms ---
    if n_padding_atoms <= 1:
        return torch.empty((bs, k, 0, 3), dtype=torch.long, device=device)

    # --- Extract remaining atoms and expand for k ---
    remaining_atoms = states_idx[:, 0, 1:, :] # Shape: (bs, n_padding_atoms - 1, 3)
    # Expand -> Shape: (bs, k, n_padding_atoms - 1, 3)
    # Need to unsqueeze remaining_atoms first to add the k dimension placeholder
    expanded_atoms = remaining_atoms.unsqueeze(1).expand(bs, k, n_padding_atoms - 1, 3)

    # --- Prepare substitutions ---
    subs = unification_result.substitutions # Alias, shape: (bs, k, 2, 2)

    # --- Apply substitutions iteratively (vectorized over bs, k, atoms, terms) ---
    next_states = expanded_atoms.clone()

    # Apply first substitution pair [var1, val1]
    var1 = subs[:, :, 0, 0].unsqueeze(-1).unsqueeze(-1) # (bs, k, 1, 1)
    val1 = subs[:, :, 0, 1].unsqueeze(-1).unsqueeze(-1) # (bs, k, 1, 1)
    is_valid_sub1 = (var1 != padding_value)             # (bs, k, 1, 1)
    match1 = (next_states == var1) & is_valid_sub1      # (bs, k, n-1, 3)
    next_states = torch.where(match1, val1, next_states)

    # Apply second substitution pair [var2, val2]
    var2 = subs[:, :, 1, 0].unsqueeze(-1).unsqueeze(-1) # (bs, k, 1, 1)
    val2 = subs[:, :, 1, 1].unsqueeze(-1).unsqueeze(-1) # (bs, k, 1, 1)
    is_valid_sub2 = (var2 != padding_value)             # (bs, k, 1, 1)
    match2 = (next_states == var2) & is_valid_sub2      # (bs, k, n-1, 3)
    next_states = torch.where(match2, val2, next_states)

    # --- Mask invalid k-slots ---
    valid_mask_expanded = unification_result.valid_mask.unsqueeze(-1).unsqueeze(-1) # (bs, k, 1, 1)
    final_next_states = torch.where(
        valid_mask_expanded,
        next_states,
        torch.tensor(padding_value, dtype=torch.long, device=device)
    )

    return final_next_states

# --- Helper formatting functions (based on user's example structure) ---
def _format_atom(atom_tensor, pred_map, const_map, var_set, var_map, padding_value=PADDING_VALUE):
    """Formats a (3,) tensor atom into a string."""
    atom_tensor_cpu = atom_tensor.cpu()
    if torch.all(atom_tensor_cpu == padding_value):
        return "[Padding Atom]"
    p, s, o = atom_tensor_cpu.tolist()
    p_str = pred_map.get(p, f"P({p})") # Use P() prefix for unknown predicates

    # Check if s is a known variable *index*
    s_str = var_map.get(s) if s in var_set else const_map.get(s, f"C({s})") # Use C() prefix for unknown constants
    # Check if o is a known variable *index*
    o_str = var_map.get(o) if o in var_set else const_map.get(o, f"C({o})")

    # Handle padding within atom terms specifically
    if s == padding_value: s_str = "PAD"
    if o == padding_value: o_str = "PAD"
    if p == padding_value: p_str = "PAD" # Should not happen in valid atoms, but good practice

    return f"{p_str}({s_str}, {o_str})"

def _format_substitution(sub_pair_tensor, const_map, var_set, var_map, padding_value=PADDING_VALUE):
    """Formats a (2,) substitution tensor [var, val] into a string."""
    sub_pair_tensor_cpu = sub_pair_tensor.cpu()
    var_idx, val_idx = sub_pair_tensor_cpu.tolist()

    # Check if the variable slot itself is padded (meaning no substitution here)
    if var_idx == padding_value:
        return None # Return None to signify no actual substitution string to print

    var_str = var_map.get(var_idx) # Assume var_idx is always a valid variable if not padding
    if not var_str: # Should not happen with valid inputs, but safety check
        var_str = f"Var({var_idx})?"

    # Value can be a variable or a constant or padding
    if val_idx == padding_value:
        val_str = "PAD"
    elif val_idx in var_set:
        val_str = var_map.get(val_idx, f"V({val_idx})?") # Should be found in var_map if in var_set
    else:
        val_str = const_map.get(val_idx, f"C({val_idx})") # Use C() prefix for unknown constants

    return f"{var_str} -> {val_str}"
# --- END Helper formatting functions ---


# --- START MODIFIED Main execution block ---
if __name__ == '__main__':
    # --- Setup with More Queries ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    K_FACTS = 2 # Max unifications from facts to display/consider
    BS = 4      # << Increased Batch size to 4
    N_ATOMS = 3 # Number of atoms per state (1 query + 2 next)
    N_FACTS = 6 # << Increased Number of facts in the knowledge base

    print(f"Settings: K_FACTS={K_FACTS}, Batch Size={BS}, Atoms/State={N_ATOMS}, Num Facts={N_FACTS}")
    print(f"Using PADDING_VALUE: {PADDING_VALUE}")

    # --- Updated Mappings & Variables ---
    predicate_map = {1: "parent", 2: "knows", 3:"likes"} # Added likes
    constant_map = {10: "Alice", 11: "Bob", 12: "Charlie", 13: "David", 14:"Eve", 15:"Frank"} # Added Frank
    vars_idx_list = [-1, -2] # V1 = -1, V2 = -2
    vars_set, vars_tensor = _get_var_set_and_tensor(vars_idx_list, device)
    variable_map = {idx: f"V{abs(idx)}" for idx in vars_idx_list} # V1, V2

    # --- State Data (BS=4, N_ATOMS=3) ---
    # Shape: (bs, 1, n_atoms, 3)
    states_idx = torch.full((BS, 1, N_ATOMS, 3), PADDING_VALUE, device=device, dtype=torch.long)

    # State 0: Query: parent(V1, Bob), Next1: knows(V1, Charlie), Next2: likes(Alice, V1)
    states_idx[0, 0, 0, :] = torch.tensor([1, -1, 11], device=device) # parent(V1, Bob)
    states_idx[0, 0, 1, :] = torch.tensor([2, -1, 12], device=device) # knows(V1, Charlie)
    states_idx[0, 0, 2, :] = torch.tensor([3, 10, -1], device=device) # likes(Alice, V1)

    # State 1: Query: knows(V1, V1), Next1: parent(David, V1), Next2: [PAD]
    states_idx[1, 0, 0, :] = torch.tensor([2, -1, -1], device=device) # knows(V1, V1)
    states_idx[1, 0, 1, :] = torch.tensor([1, 13, -1], device=device) # parent(David, V1)
    # states_idx[1, 0, 2, :] remains PADDING

    # State 2: Query: likes(V1, V2), Next1: knows(V2, V1), Next2: parent(V1, Frank)
    states_idx[2, 0, 0, :] = torch.tensor([3, -1, -2], device=device) # likes(V1, V2)
    states_idx[2, 0, 1, :] = torch.tensor([2, -2, -1], device=device) # knows(V2, V1)
    states_idx[2, 0, 2, :] = torch.tensor([1, -1, 15], device=device) # parent(V1, Frank)

    # State 3: Query: parent(Eve, Frank) -> No variables, Next1: likes(Eve, Bob), Next2: [PAD]
    states_idx[3, 0, 0, :] = torch.tensor([1, 14, 15], device=device) # parent(Eve, Frank)
    states_idx[3, 0, 1, :] = torch.tensor([3, 14, 11], device=device) # likes(Eve, Bob)
    # states_idx[3, 0, 2, :] remains PADDING


    # --- Facts Data (N_FACTS=6) ---
    # Shape: (n_facts, 3)
    facts_tensor = torch.full((N_FACTS, 3), PADDING_VALUE, device=device, dtype=torch.long)
    # Fact 0: parent(Alice, Bob) -> Unifies S0 {V1->Alice}
    facts_tensor[0, :] = torch.tensor([1, 10, 11], device=device)
    # Fact 1: knows(Charlie, Charlie) -> Unifies S1 {V1->Charlie}
    facts_tensor[1, :] = torch.tensor([2, 12, 12], device=device)
    # Fact 2: parent(David, Bob) -> Unifies S0 {V1->David}
    facts_tensor[2, :] = torch.tensor([1, 13, 11], device=device)
    # Fact 3: knows(Alice, Bob) -> Conflict for S1 query (V1==V1 requires S==O)
    facts_tensor[3, :] = torch.tensor([2, 10, 11], device=device)
    # Fact 4: likes(Alice, Bob) -> Unifies S2 {V1->Alice, V2->Bob}
    facts_tensor[4, :] = torch.tensor([3, 10, 11], device=device)
    # Fact 5: parent(Eve, Frank) -> Unifies S3 {} (no vars)
    facts_tensor[5, :] = torch.tensor([1, 14, 15], device=device)
    # Note: We could add more facts like `likes(David, Charlie)` etc.


    print("\n--- Initial Data ---")
    print(f"Variable Map: {variable_map}")
    print(f"Facts Tensor (Indices 0 to {N_FACTS-1}):\n", facts_tensor.cpu().numpy())
    # Print states more readably
    print("Initial States:")
    for b in range(BS):
         print(f" State {b}:")
         for i in range(N_ATOMS):
             atom_tensor = states_idx[b,0,i,:]
             atom_str = _format_atom(atom_tensor, predicate_map, constant_map, vars_set, variable_map)
             # Also print raw indices for the state atoms
             print(f"  Atom {i}: {atom_str}  (Indices: {atom_tensor.cpu().numpy().tolist()})")
    print("-" * 30)

    # --- Step 1: Extract First Atoms (Queries) ---
    first_atoms_queries = states_idx[:, 0, 0, :].clone() # Shape: (bs, 3)

    # --- Step 2: Get Unifications for the First Atoms ---
    unification_result = get_first_k_fact_unifications_generalized(
        queries_idx=first_atoms_queries,
        facts=facts_tensor,
        vars_idx=vars_tensor, # Pass the tensor containing var indices
        k=K_FACTS,
        device=device,
        padding_value=PADDING_VALUE
    )

    # --- Step 3: Apply Substitutions to Remaining Atoms ---
    next_states_idx = apply_substitutions_to_states(
        states_idx=states_idx, # Pass the original full state
        unification_result=unification_result,
        padding_value=PADDING_VALUE
    ) # Shape (bs, K_FACTS, n_atoms - 1, 3)


    # --- Step 4: Interpret and Print Results with Indices ---
    print("\n" + "=" * 20 + " INTERPRETED RESULTS " + "=" * 20)

    for b in range(BS): # Iterate through Batch items
        query_atom = states_idx[b, 0, 0, :]
        query_str = _format_atom(query_atom, predicate_map, constant_map, vars_set, variable_map)
        print(f"\n--- Processing Batch Item b={b} ---")
        # Print original query with raw indices
        print(f"  Original Query: {query_str}  (Indices: {query_atom.cpu().numpy().tolist()})")

        print(f"  Looking for Top {K_FACTS} Fact Unifications:")

        found_any_unification_for_b = False
        # Iterate through the K slots provided by the unification result
        for k_idx in range(K_FACTS):
            # Check if this k-slot is valid for this batch item
            if b < unification_result.valid_mask.shape[0] and k_idx < unification_result.valid_mask.shape[1] \
               and unification_result.valid_mask[b, k_idx]:

                found_any_unification_for_b = True
                fact_index = unification_result.fact_indices[b, k_idx].item()
                subs_tensor_bk = unification_result.substitutions[b, k_idx] # Shape (2, 2)

                print(f"\n  -> Result Slot k={k_idx}:") # Identify the K-slot

                # Print the Fact it unified with, including raw indices
                if fact_index != PADDING_VALUE and 0 <= fact_index < len(facts_tensor):
                    fact_atom = facts_tensor[fact_index]
                    fact_str = _format_atom(fact_atom, predicate_map, constant_map, set(), {})
                    print(f"     Unified with Fact Index fact_idx={fact_index}: {fact_str}  (Indices: {fact_atom.cpu().numpy().tolist()})")
                else:
                     print(f"     Unified with Fact Index: {fact_index} [INVALID or PADDING - Check Logic]")

                # Print Substitutions for this (b, k) pair, including raw indices
                print(f"     Substitutions Generated (Raw: {subs_tensor_bk.cpu().numpy().tolist()}):")
                subs_found_in_slot = False
                for i in range(subs_tensor_bk.shape[0]): # Iterate through the 2 potential substitution pairs [Var, Val]
                    sub_pair = subs_tensor_bk[i]
                    sub_str = _format_substitution(sub_pair, constant_map, vars_set, variable_map)
                    if sub_str is not None:
                        print(f"       {sub_str}")
                        subs_found_in_slot = True
                if not subs_found_in_slot:
                    print("       [None]")

                # Print the Resulting Next State for this (b, k) branch, including raw indices
                print(f"     Resulting Next State Queries:")
                has_next_query = False
                # Check bounds for next_states_idx before accessing
                if b < next_states_idx.shape[0] and k_idx < next_states_idx.shape[1]:
                    # Iterate through remaining atoms in the next state (N_ATOMS - 1)
                    for atom_idx in range(next_states_idx.shape[2]): # next_states_idx.shape[2] == N_ATOMS - 1
                        next_atom = next_states_idx[b, k_idx, atom_idx, :]
                        # Check if the atom is not just padding
                        if torch.any(next_atom.cpu() != PADDING_VALUE):
                             next_atom_str = _format_atom(next_atom, predicate_map, constant_map, vars_set, variable_map)
                             # Print next state atom with raw indices
                             print(f"       Atom {atom_idx+1}: {next_atom_str}  (Indices: {next_atom.cpu().numpy().tolist()})")
                             has_next_query = True
                if not has_next_query:
                     print("       [State Proven or No Remaining Queries]")

            # Optional: Print a message if a k-slot is invalid/padded
            # else:
            #    # Check bounds before printing this message too
            #    if b < unification_result.valid_mask.shape[0] and k_idx < unification_result.valid_mask.shape[1]:
            #         print(f"\n  -> Result Slot k={k_idx}: [No valid unification found for this slot]")


        if not found_any_unification_for_b:
            # Check if the original query was padding to differentiate reasons for no unification
            if torch.all(query_atom.cpu() == PADDING_VALUE):
                 print("  -> Original query was padding. No unification attempted.")
            else:
                 print("  -> No matching facts found or all potential matches resulted in conflict.")


    print("\n" + "=" * 50)
# --- END MODIFIED Main execution block ---