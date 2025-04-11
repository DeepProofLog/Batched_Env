import torch
from typing import Tuple, Optional, Union, List, Set, Dict

# Define a suitable padding value (must not be a valid index, variable, or constant)
# Use -999 instead of -1 to avoid collision with variable index -1
PADDING_VALUE = -999
# Define a value to represent an unbound variable in substitution results
UNBOUND_VAR = -998 # Needs to be distinct from PADDING_VALUE and valid indices

def _format_atom(
    atom_tensor: torch.Tensor,
    predicate_idx2str: Optional[Dict[int, str]] = None,
    constant_idx2str: Optional[Dict[int, str]] = None,
    vars_idx: Optional[Union[torch.Tensor, List[int], Set[int]]] = None
) -> str:
    """Helper function to format an atom tensor into a readable string."""
    # Check if input is a tensor of the correct shape
    if not isinstance(atom_tensor, torch.Tensor) or atom_tensor.dim() == 0 or atom_tensor.numel() != 3:
        return f"[invalid atom: {atom_tensor}]" # Fallback for unexpected input

    p_idx, s_idx, o_idx = atom_tensor.tolist()

    # Convert vars_idx to a set for efficient lookup if needed
    vars_idx_set = set()
    if vars_idx is not None:
        if isinstance(vars_idx, torch.Tensor):
            vars_idx_set = set(vars_idx.tolist())
        elif isinstance(vars_idx, (list, set)):
            vars_idx_set = set(vars_idx)
        # else: it remains an empty set


    # Format predicate
    p_str = str(p_idx)
    if predicate_idx2str and p_idx in predicate_idx2str:
        p_str = predicate_idx2str[p_idx]
    # Check for the specific padding atom [0, 0, 0] or fully padded [-999, -999, -999]
    elif (p_idx == 0 and s_idx == 0 and o_idx == 0) or \
         (p_idx == PADDING_VALUE and s_idx == PADDING_VALUE and o_idx == PADDING_VALUE):
        return "[padding_atom]" # Changed label for clarity

    # --- Helper to format Subject/Object ---
    def format_term(term_idx, is_var_func):
        term_str = str(term_idx)
        is_term_var = is_var_func(term_idx) # Check against the set of variable indices
        if is_term_var:
            term_str = f"VAR({term_idx})"
        elif constant_idx2str and term_idx in constant_idx2str:
            term_str = constant_idx2str[term_idx]
        # Check against the NEW PADDING_VALUE
        elif term_idx == PADDING_VALUE:
            term_str = "[pad_val]" # Changed label for clarity
        # Handle 0 constant if predicate is not 0 (to distinguish from padding atom)
        elif term_idx == 0 and p_idx != 0 and p_idx != PADDING_VALUE:
             term_str = str(term_idx) # Treat 0 as a constant
        elif term_idx == UNBOUND_VAR:
             term_str = "[unbound]"
        return term_str

    # Check if an index is a variable based on the provided set
    is_variable = lambda idx: idx in vars_idx_set

    s_str = format_term(s_idx, is_variable)
    o_str = format_term(o_idx, is_variable)

    return f"({p_str}, {s_str}, {o_str})"

def _format_substitution(
    sub_pair: torch.Tensor,
    constant_idx2str: Optional[Dict[int, str]] = None,
    vars_idx: Optional[Union[torch.Tensor, List[int], Set[int]]] = None # Added vars_idx
) -> str:
    """Helper function to format a substitution pair."""
    # Check input type and shape
    if not isinstance(sub_pair, torch.Tensor) or sub_pair.shape != (2,):
        return f"[invalid sub pair: {sub_pair}]"

    var_idx, value_idx = sub_pair.tolist()

    # Fallback check against the NEW PADDING_VALUE
    if var_idx == PADDING_VALUE:
        return "[no_sub]"

    # Convert vars_idx to a set for efficient lookup if needed
    vars_idx_set = set()
    if vars_idx is not None:
        if isinstance(vars_idx, torch.Tensor):
            vars_idx_set = set(vars_idx.tolist())
        elif isinstance(vars_idx, (list, set)):
            vars_idx_set = set(vars_idx)

    # Format the value part of the substitution
    value_str = str(value_idx)
    is_value_var = value_idx in vars_idx_set
    if is_value_var:
         value_str = f"VAR({value_idx})"
    elif constant_idx2str and value_idx in constant_idx2str:
        value_str = constant_idx2str[value_idx]
    # Check against the NEW PADDING_VALUE
    elif value_idx == PADDING_VALUE:
        value_str = "[pad_val]"
    elif value_idx == UNBOUND_VAR:
        value_str = "[unbound]"
    # Handle 0 constant
    elif value_idx == 0:
        value_str = str(value_idx)


    return f"VAR({var_idx}) -> {value_str}"


# --- Fact Unification (Original function renamed) ---
def batch_unify_with_facts(
    states_idx: torch.Tensor,      # Shape: (bs, 1, n_padding_atoms, 3)
    facts: torch.Tensor,           # Shape: (n_facts, 3)
    vars_idx: Union[torch.Tensor, List[int], Set[int]], # Variable indices
    k: int,                        # Max number of unifications to return
    device: Optional[str] = None,
    verbose: bool = False,         # Flag to enable debug printing
    predicate_idx2str: Optional[Dict[int, str]] = None, # Optional mapping for predicates
    constant_idx2str: Optional[Dict[int, str]] = None   # Optional mapping for constants/entities
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Performs batched unification for the first query (atom at index 0) against FACTS,
    returning substitutions for up to K unifying facts.
    Substitutions bind variables in the QUERY to constants in the FACT.
    (Function body omitted for brevity as it's unchanged and working)
    """
    # --- Setup ---
    if k <= 0:
        raise ValueError("k must be a positive integer")

    if device is None:
        device = states_idx.device
    else:
        states_idx = states_idx.to(device)
        facts = facts.to(device)

    if isinstance(vars_idx, (list, set)):
        vars_idx_tensor = torch.tensor(list(vars_idx), dtype=torch.long, device=device)
    elif isinstance(vars_idx, torch.Tensor):
        vars_idx_tensor = vars_idx.to(device)
    else:
        raise TypeError("vars_idx must be a tensor, list, or set")
    vars_idx_set = set(vars_idx_tensor.tolist())


    bs, _, n_padding_atoms, _ = states_idx.shape
    n_facts = facts.shape[0]

    if verbose:
        print("-" * 40)
        print(f"DEBUG: batch_unify_with_facts (Batch Element 0)")
        print("-" * 40)
        print(f"Batch size: {bs}, Num padding atoms: {n_padding_atoms}, Num facts: {n_facts}, k: {k}")
        print(f"Device: {device}")
        print(f"Variable indices: {vars_idx_set}")
        print(f"Using PADDING_VALUE: {PADDING_VALUE}")

    actual_k = min(k, n_facts) if n_facts > 0 else 0

    substitutions_out = torch.full((bs, k, 2, 2), PADDING_VALUE, dtype=torch.long, device=device)
    fact_indices_out = torch.full((bs, k), PADDING_VALUE, dtype=torch.long, device=device)
    valid_mask_out = torch.zeros((bs, k), dtype=torch.bool, device=device)

    if bs == 0 or n_facts == 0 or n_padding_atoms == 0:
        if verbose: print("DEBUG: Early exit (no batch items, facts, or atoms in state).")
        return substitutions_out, fact_indices_out, valid_mask_out

    # --- Simplified First Query Extraction ---
    first_queries = states_idx[:, 0, 0, :] # Shape: (bs, 3)
    has_valid_query = ~torch.all(first_queries == 0, dim=-1) & \
                      ~torch.all(first_queries == PADDING_VALUE, dim=-1) # Shape: (bs,)


    if verbose:
        # Ensure query0 is on CPU for formatting if needed, handle potential empty first_queries
        if bs > 0:
            query0 = first_queries[0]
            query0_str = _format_atom(query0.cpu(), predicate_idx2str, constant_idx2str, vars_idx_set)
            print(f"\nDEBUG: First Query [0]: {query0_str} (Raw: {query0.tolist()})")
            print(f"DEBUG: Has Valid Query [0]: {has_valid_query[0].item()}")
            if not has_valid_query[0].item():
                print("DEBUG: First query is padding, skipping unification for this element.")
        else:
             print("\nDEBUG: Batch size is 0, no first query.")


    # --- Calculate Unification Mask ---
    # Ensure tensors are not empty before unsqueezing
    if bs > 0 and n_facts > 0:
        queries_expanded = first_queries.unsqueeze(1) # Shape: (bs, 1, 3)
        facts_expanded = facts.unsqueeze(0)           # Shape: (1, n_facts, 3)

        is_query_var_s = torch.isin(queries_expanded[:, :, 1], vars_idx_tensor) # Shape: (bs, 1)
        is_query_var_o = torch.isin(queries_expanded[:, :, 2], vars_idx_tensor) # Shape: (bs, 1)
        is_query_var_s_expanded = is_query_var_s.expand(bs, n_facts) # Shape: (bs, n_facts)
        is_query_var_o_expanded = is_query_var_o.expand(bs, n_facts) # Shape: (bs, n_facts)

        pred_match = (queries_expanded[:, :, 0] == facts_expanded[:, :, 0]) # Shape: (bs, n_facts)

        subj_match = is_query_var_s_expanded | (
            (~is_query_var_s_expanded) & (queries_expanded[:, :, 1] == facts_expanded[:, :, 1])
        )
        obj_match = is_query_var_o_expanded | (
            (~is_query_var_o_expanded) & (queries_expanded[:, :, 2] == facts_expanded[:, :, 2])
        )

        unifies_mask = pred_match & subj_match & obj_match
        unifies_mask = unifies_mask & has_valid_query.unsqueeze(1) # Shape: (bs, n_facts)
    else:
        # Handle empty case: no unification possible
         unifies_mask = torch.zeros((bs, n_facts), dtype=torch.bool, device=device)


    if verbose and bs > 0:
        print("\nDEBUG: Fact Unification Check [Batch Element 0]:")
        unifies_mask_0 = unifies_mask[0]
        # Handle case where unifies_mask_0 might be empty if n_facts is 0
        if unifies_mask_0.numel() > 0:
            unifying_fact_indices_0 = torch.nonzero(unifies_mask_0).squeeze(-1)
            if unifying_fact_indices_0.dim() == 0: unifying_fact_indices_0 = unifying_fact_indices_0.unsqueeze(0)
            unifying_fact_indices_0 = unifying_fact_indices_0.tolist()
            print(f"  Potentially Unifying Fact Indices: {unifying_fact_indices_0}")
            for fact_idx in unifying_fact_indices_0:
                 # Check index validity before accessing facts
                 if 0 <= fact_idx < n_facts:
                     fact_str = _format_atom(facts[fact_idx].cpu(), predicate_idx2str, constant_idx2str, vars_idx_set)
                     print(f"       - Fact {fact_idx}: {fact_str} (Raw: {facts[fact_idx].tolist()})")
                 else:
                      print(f"       - Invalid Fact Index: {fact_idx}")
            if not unifying_fact_indices_0: print("       No facts potentially unify.")
        else:
            print("       No facts to check against.")


    # --- Select Arbitrary K Unifying Facts ---
    # Ensure scores tensor is created correctly even if unifies_mask is empty
    if unifies_mask.numel() > 0:
         scores = torch.where(unifies_mask, torch.tensor(1.0, device=device), torch.tensor(-1.0, device=device))
    else:
         scores = torch.full((bs, n_facts), -1.0, device=device) # All invalid


    if actual_k == 0:
        if verbose: print("\nDEBUG: actual_k is 0, skipping topk.")
        return substitutions_out, fact_indices_out, valid_mask_out

    # topk requires k > 0 and input dim size >= k. Handled by actual_k logic.
    # Also requires input dim size > 0 if k > 0.
    if n_facts == 0 and actual_k > 0:
         if verbose: print("\nDEBUG: No facts available, skipping topk.")
         return substitutions_out, fact_indices_out, valid_mask_out # Already padded

    # Ensure scores has a non-zero dimension size for topk if actual_k > 0
    if scores.shape[1] > 0:
        top_scores, top_fact_indices = torch.topk(scores, k=actual_k, dim=1) # (bs, actual_k)
        initial_valid_k_mask = (top_scores > -1.0) # (bs, actual_k)
        top_fact_indices = torch.where(initial_valid_k_mask, top_fact_indices, torch.tensor(PADDING_VALUE, dtype=torch.long, device=device))
    else: # Handle case where n_facts < actual_k (should be covered by actual_k=min(k, n_facts)) or n_facts=0
        top_fact_indices = torch.full((bs, actual_k), PADDING_VALUE, dtype=torch.long, device=device)
        initial_valid_k_mask = torch.zeros((bs, actual_k), dtype=torch.bool, device=device)


    if verbose and bs > 0:
        print(f"\nDEBUG: Top-{actual_k} Fact Selection [Batch Element 0]:")
        # Check if top_fact_indices exists and has data for batch 0
        if top_fact_indices.shape[0] > 0:
             print(f"  Top Fact Indices (Raw, Padded): {top_fact_indices[0, :].tolist()}")
             print(f"  Initial Valid K Mask: {initial_valid_k_mask[0, :].tolist()}")
        else:
             print("  No top fact indices generated (likely bs=0 or n_facts=0).")


    # --- Gather Data and Check Conflicts ---
    valid_k_coords = torch.nonzero(initial_valid_k_mask, as_tuple=True)
    n_valid_topk = len(valid_k_coords[0])
    final_valid_mask_slice = torch.zeros((bs, actual_k), dtype=torch.bool, device=device)

    if n_valid_topk > 0:
        valid_bs_indices = valid_k_coords[0]
        valid_k_indices = valid_k_coords[1]
        # Ensure valid_fact_indices are within bounds before gathering
        valid_fact_indices_raw = top_fact_indices[valid_k_coords]
        valid_fact_indices_mask = (valid_fact_indices_raw >= 0) & (valid_fact_indices_raw < n_facts)

        # Filter coordinates and indices based on valid fact indices
        valid_k_coords = (valid_bs_indices[valid_fact_indices_mask], valid_k_indices[valid_fact_indices_mask])
        valid_fact_indices = valid_fact_indices_raw[valid_fact_indices_mask]
        n_valid_topk = len(valid_k_coords[0]) # Update count

        if n_valid_topk > 0: # Proceed only if valid indices remain
            valid_bs_indices = valid_k_coords[0] # Update bs indices

            top_k_queries_flat = first_queries[valid_bs_indices] # Shape: (n_valid_topk, 3)
            top_k_facts_flat = facts[valid_fact_indices]         # Shape: (n_valid_topk, 3)

            top_k_is_s_var_flat = is_query_var_s.squeeze(1)[valid_bs_indices] # Shape: (n_valid_topk,)
            top_k_is_o_var_flat = is_query_var_o.squeeze(1)[valid_bs_indices] # Shape: (n_valid_topk,)

            # Conflict: Same variable in query S/O, different constants in fact S/O
            is_same_query_var_flat = top_k_is_s_var_flat & top_k_is_o_var_flat & (top_k_queries_flat[:, 1] == top_k_queries_flat[:, 2])
            is_diff_fact_const_flat = top_k_facts_flat[:, 1] != top_k_facts_flat[:, 2]
            is_conflict_flat = is_same_query_var_flat & is_diff_fact_const_flat # Shape: (n_valid_topk,)

            # Place non-conflicting results into the mask using the filtered coordinates
            final_valid_mask_slice[valid_k_coords] = ~is_conflict_flat

            if verbose and bs > 0:
                print("\nDEBUG: Fact Conflict Check [Batch Element 0]:")
                indices_for_batch_0 = (valid_bs_indices == 0).nonzero().squeeze(-1)
                if indices_for_batch_0.dim() == 0: indices_for_batch_0 = indices_for_batch_0.unsqueeze(0)
                if len(indices_for_batch_0) > 0:
                    k_indices_0 = valid_k_coords[1][indices_for_batch_0] # Use k indices from filtered coords
                    fact_indices_0 = valid_fact_indices[indices_for_batch_0]
                    conflicts_0 = is_conflict_flat[indices_for_batch_0]
                    for i in range(len(indices_for_batch_0)):
                        flat_idx = indices_for_batch_0[i].item()
                        k_slot = k_indices_0[i].item()
                        fact_idx = fact_indices_0[i].item()
                        is_conflict = conflicts_0[i].item()
                        query_str = _format_atom(top_k_queries_flat[flat_idx].cpu(), predicate_idx2str, constant_idx2str, vars_idx_set)
                        fact_str = _format_atom(top_k_facts_flat[flat_idx].cpu(), predicate_idx2str, constant_idx2str, vars_idx_set)
                        print(f"       - K-Slot {k_slot}, Fact {fact_idx}: Q={query_str}, F={fact_str} -> Conflict: {is_conflict}")
                else: print("       No facts in Top K for batch 0 to check (after filtering).")
                print(f"  Final Valid Mask Slice [0]: {final_valid_mask_slice[0].tolist()}")


            # --- Populate Output Tensors ---
            # Use the final_valid_mask_slice to decide which top_fact_indices to keep
            fact_indices_out[:, :actual_k] = torch.where(final_valid_mask_slice, top_fact_indices[:,:actual_k], torch.tensor(PADDING_VALUE, dtype=torch.long, device=device))
            valid_mask_out[:, :actual_k] = final_valid_mask_slice

            # --- Calculate and Populate Substitutions ---
            final_valid_coords = torch.nonzero(final_valid_mask_slice, as_tuple=True)
            n_final_valid = len(final_valid_coords[0])

            if n_final_valid > 0:
                 # We need to map the final valid (bs, k) coordinates back to the indices
                 # within the filtered 'flat' arrays (top_k_queries_flat, etc.)
                 # Create a temporary map using the valid_k_coords (filtered ones)
                 temp_map_flat = torch.full_like(final_valid_mask_slice, -1, dtype=torch.long)
                 temp_map_flat[valid_k_coords] = torch.arange(n_valid_topk, device=device) # Map (bs,k) -> flat_idx (filtered)

                 # Select using the final valid coordinates
                 final_valid_indices_in_flat = temp_map_flat[final_valid_coords]

                 # Filter out potential -1 indices (shouldn't happen with correct logic)
                 valid_flat_mask = final_valid_indices_in_flat != -1
                 final_valid_coords = (final_valid_coords[0][valid_flat_mask], final_valid_coords[1][valid_flat_mask])
                 final_valid_indices_in_flat = final_valid_indices_in_flat[valid_flat_mask]
                 n_final_valid = len(final_valid_indices_in_flat) # Update count


                 if n_final_valid > 0:
                    # Gather data only for the final valid substitutions from the filtered flat arrays
                    s_var_indices = top_k_queries_flat[final_valid_indices_in_flat, 1]
                    s_const_indices = top_k_facts_flat[final_valid_indices_in_flat, 1]
                    o_var_indices = top_k_queries_flat[final_valid_indices_in_flat, 2]
                    o_const_indices = top_k_facts_flat[final_valid_indices_in_flat, 2]

                    s_sub_needed_final = top_k_is_s_var_flat[final_valid_indices_in_flat]
                    o_sub_needed_final = top_k_is_o_var_flat[final_valid_indices_in_flat]

                    # Subject substitutions
                    s_sub_final_coords = (final_valid_coords[0][s_sub_needed_final], final_valid_coords[1][s_sub_needed_final])
                    if len(s_sub_final_coords[0]) > 0:
                        s_vars_to_assign = s_var_indices[s_sub_needed_final]
                        s_consts_to_assign = s_const_indices[s_sub_needed_final]
                        substitutions_out[s_sub_final_coords[0], s_sub_final_coords[1], 0, 0] = s_vars_to_assign
                        substitutions_out[s_sub_final_coords[0], s_sub_final_coords[1], 0, 1] = s_consts_to_assign

                    # Object substitutions
                    o_sub_final_coords = (final_valid_coords[0][o_sub_needed_final], final_valid_coords[1][o_sub_needed_final])
                    if len(o_sub_final_coords[0]) > 0:
                        o_vars_to_assign = o_var_indices[o_sub_needed_final]
                        o_consts_to_assign = o_const_indices[o_sub_needed_final]
                        substitutions_out[o_sub_final_coords[0], o_sub_final_coords[1], 1, 0] = o_vars_to_assign
                        substitutions_out[o_sub_final_coords[0], o_sub_final_coords[1], 1, 1] = o_consts_to_assign

                    if verbose and bs > 0:
                        print("\nDEBUG: Final Populated Fact Substitutions [Batch Element 0]:")
                        final_indices_for_batch_0 = (final_valid_coords[0] == 0).nonzero().squeeze(-1)
                        if final_indices_for_batch_0.dim() == 0: final_indices_for_batch_0 = final_indices_for_batch_0.unsqueeze(0)
                        if len(final_indices_for_batch_0) > 0:
                            k_indices_final_0 = final_valid_coords[1][final_indices_for_batch_0]
                            for i in range(len(final_indices_for_batch_0)):
                                k_slot = k_indices_final_0[i].item()
                                s_sub_pair = substitutions_out[0, k_slot, 0, :]
                                o_sub_pair = substitutions_out[0, k_slot, 1, :]
                                s_sub_str = _format_substitution(s_sub_pair.cpu(), constant_idx2str, vars_idx_set)
                                o_sub_str = _format_substitution(o_sub_pair.cpu(), constant_idx2str, vars_idx_set)
                                fact_idx = fact_indices_out[0, k_slot].item()
                                print(f"       - K-Slot {k_slot} (Fact {fact_idx}): Sub S: {s_sub_str}, Sub O: {o_sub_str}")
                        else: print("       No final valid fact substitutions for batch 0.")


    if verbose:
        print("-" * 40)
        print("DEBUG: End Fact Unification Verbose Output [0]")
        print("-" * 40)

    return substitutions_out, fact_indices_out, valid_mask_out


# --- Fact Substitution Application (Original function) ---
def apply_fact_substitutions_to_rest(
    states_idx: torch.Tensor,      # Shape: (bs, 1, n_padding_atoms, 3)
    substitutions: torch.Tensor,   # Shape: (bs, k, 2, 2) - Output from batch_unify_with_facts
    valid_mask: torch.Tensor,      # Shape: (bs, k) - Output from batch_unify_with_facts
    PADDING_VALUE: int = -999
) -> torch.Tensor:
    """
    Applies the FACT substitutions (Var_Query -> Const_Fact) to the remaining queries
    in each state for each valid unification.
    (Function body omitted for brevity as it's unchanged and working)
    """
    bs, _, n_padding_atoms, _ = states_idx.shape
    k = substitutions.shape[1]
    device = states_idx.device

    if n_padding_atoms <= 1:
        return torch.empty((bs, k, 0, 3), dtype=torch.long, device=device)

    # --- Prepare Tensors ---
    remaining_queries_base = states_idx[:, 0, 1:, :] # (bs, n_padding_atoms - 1, 3)
    remaining_queries = remaining_queries_base.unsqueeze(1).expand(bs, k, n_padding_atoms - 1, 3) # (bs, k, n_pad-1, 3)

    # --- Extract Substitution Components ---
    s_var = substitutions[:, :, 0, 0].unsqueeze(-1) # (bs, k, 1)
    s_const = substitutions[:, :, 0, 1].unsqueeze(-1)
    o_var = substitutions[:, :, 1, 0].unsqueeze(-1)
    o_const = substitutions[:, :, 1, 1].unsqueeze(-1)

    s_sub_valid = (s_var != PADDING_VALUE) # (bs, k, 1)
    o_sub_valid = (o_var != PADDING_VALUE)

    # --- Apply Substitutions Vectorized ---
    applied_queries = remaining_queries.clone()
    rq_s = remaining_queries[:, :, :, 1] # (bs, k, n_pad-1)
    rq_o = remaining_queries[:, :, :, 2]

    new_s = rq_s
    new_s = torch.where((rq_s == s_var) & s_sub_valid, s_const, new_s)
    new_s = torch.where((rq_s == o_var) & o_sub_valid, o_const, new_s)

    new_o = rq_o
    new_o = torch.where((rq_o == s_var) & s_sub_valid, s_const, new_o)
    new_o = torch.where((rq_o == o_var) & o_sub_valid, o_const, new_o)

    applied_queries[:, :, :, 1] = new_s
    applied_queries[:, :, :, 2] = new_o

    # --- Mask the Results ---
    next_states = torch.full_like(applied_queries, PADDING_VALUE, dtype=torch.long, device=device)
    valid_mask_expanded = valid_mask.unsqueeze(-1).unsqueeze(-1).expand_as(applied_queries)
    next_states = torch.where(valid_mask_expanded, applied_queries, next_states)

    return next_states


# --- NEW: General Substitution Application ---
def apply_substitutions_general(
    atoms: torch.Tensor,           # Shape: (num_atoms, 3)
    substitutions: torch.Tensor,   # Shape: (max_subs, 2), [var_idx, value_idx]
    vars_idx_tensor: torch.Tensor, # Tensor of variable indices
    max_iterations: int = 10,      # Limit recursion/iteration depth
    PADDING_VALUE: int = -999,
    UNBOUND_VAR: int = -998
) -> torch.Tensor:
    """
    Applies a list of substitutions (Var -> Value) to a tensor of atoms.
    Handles Var -> Const and Var -> Var substitutions iteratively.
    Value can be a constant index or another variable index.
    (Function body omitted for brevity as it's unchanged)
    """
    if atoms.numel() == 0 or substitutions.numel() == 0:
        return atoms.clone() # No atoms or substitutions to apply

    device = atoms.device
    substituted_atoms = atoms.clone()
    num_atoms = atoms.shape[0]

    # Filter out padding substitutions and create a lookup map
    valid_subs_mask = (substitutions[:, 0] != PADDING_VALUE)
    valid_subs = substitutions[valid_subs_mask] # Shape (n_valid_subs, 2)
    n_valid_subs = valid_subs.shape[0]

    if n_valid_subs == 0:
        return substituted_atoms # No valid substitutions

    # --- Iterative Substitution Application ---
    changed = True
    iterations = 0
    while changed and iterations < max_iterations:
        changed = False
        iterations += 1
        current_atoms = substituted_atoms.clone() # Work on a copy in this iteration

        # Iterate through each valid substitution rule
        for i in range(n_valid_subs):
            var_to_replace = valid_subs[i, 0]
            replacement_value = valid_subs[i, 1]

            # Find where this variable appears in Subject and Object columns
            s_match_mask = (current_atoms[:, 1] == var_to_replace) # Shape (num_atoms,)
            o_match_mask = (current_atoms[:, 2] == var_to_replace) # Shape (num_atoms,)

            # Apply substitution if found
            if torch.any(s_match_mask):
                substituted_atoms[s_match_mask, 1] = replacement_value
                changed = True
            if torch.any(o_match_mask):
                substituted_atoms[o_match_mask, 2] = replacement_value
                changed = True

        # Optional: Check for simple cycles (X->Y, Y->X) - more complex cycles need occurs check
        # This basic check might not catch longer chains easily in tensor format

    if iterations >= max_iterations and changed:
        print(f"Warning: Max substitution iterations ({max_iterations}) reached. Potential cycle or slow convergence.")
        # Decide how to handle this: return current state or raise error?
        # Returning current state might leave some variables unbound.

    return substituted_atoms


# --- NEW: Rule Unification ---
def batch_unify_with_rules(
    states_idx: torch.Tensor,      # Shape: (bs, 1, n_padding_atoms, 3)
    rules: torch.Tensor,           # Shape: (n_rules, max_rule_atoms, 3) Head=rules[:,0,:], Body=rules[:,1:,:]
    vars_idx: Union[torch.Tensor, List[int], Set[int]], # Variable indices (used by BOTH query and rules)
    k: int,                        # Max number of rule unifications to return
    max_subs_per_rule: int = 5,    # Max substitutions tracked per unification
    device: Optional[str] = None,
    verbose: bool = False,
    predicate_idx2str: Optional[Dict[int, str]] = None,
    constant_idx2str: Optional[Dict[int, str]] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Performs batched unification for the first query against RULE HEADS.
    Calculates substitutions needed to make query and head identical.
    Handles Var->Const, Const->Var, Var->Var bindings.
    """
    # --- Setup ---
    if k <= 0: raise ValueError("k must be a positive integer")
    if max_subs_per_rule <= 0: raise ValueError("max_subs_per_rule must be positive")

    if device is None: device = states_idx.device
    else:
        states_idx = states_idx.to(device)
        rules = rules.to(device)

    if isinstance(vars_idx, (list, set)):
        vars_idx_tensor = torch.tensor(list(vars_idx), dtype=torch.long, device=device)
    elif isinstance(vars_idx, torch.Tensor):
        vars_idx_tensor = vars_idx.to(device)
    else: raise TypeError("vars_idx must be a tensor, list, or set")
    vars_idx_set = set(vars_idx_tensor.tolist())

    bs, _, n_padding_atoms, _ = states_idx.shape
    n_rules, max_rule_atoms, _ = rules.shape
    rule_body_len = max_rule_atoms - 1

    if verbose:
        print("\n" + "=" * 40)
        print(f"DEBUG: batch_unify_with_rules (Batch Element 0 or State 1)")
        print("=" * 40)
        print(f"BS: {bs}, N_Rules: {n_rules}, Max Rule Atoms: {max_rule_atoms}, K: {k}, Max Subs: {max_subs_per_rule}")
        print(f"Device: {device}, Vars: {vars_idx_set}, Padding: {PADDING_VALUE}")

    actual_k = min(k, n_rules) if n_rules > 0 else 0

    substitutions_out = torch.full((bs, k, max_subs_per_rule, 2), PADDING_VALUE, dtype=torch.long, device=device)
    rule_indices_out = torch.full((bs, k), PADDING_VALUE, dtype=torch.long, device=device)
    rule_bodies_out_shape = (bs, k, max(0, rule_body_len), 3)
    rule_bodies_out = torch.full(rule_bodies_out_shape, PADDING_VALUE, dtype=torch.long, device=device)
    valid_mask_out = torch.zeros((bs, k), dtype=torch.bool, device=device)

    if bs == 0 or n_rules == 0 or n_padding_atoms == 0 or max_rule_atoms <= 0:
        if verbose: print("DEBUG: Early exit (no batch items, rules, atoms in state, or rule atoms).")
        return substitutions_out, rule_indices_out, rule_bodies_out, valid_mask_out

    first_queries = states_idx[:, 0, 0, :]
    has_valid_query = ~torch.all(first_queries == 0, dim=-1) & \
                      ~torch.all(first_queries == PADDING_VALUE, dim=-1)

    rule_heads = rules[:, 0, :]
    rule_bodies = rules[:, 1:, :]

    if verbose and bs > 0:
        b_idx_print = 1 if bs > 1 else 0
        query_print = first_queries[b_idx_print]
        query_print_str = _format_atom(query_print.cpu(), predicate_idx2str, constant_idx2str, vars_idx_set)
        print(f"\nDEBUG: First Query [b={b_idx_print}]: {query_print_str} (Raw: {query_print.tolist()})")
        print(f"DEBUG: Has Valid Query [b={b_idx_print}]: {has_valid_query[b_idx_print].item()}")
        if not has_valid_query[b_idx_print].item(): print(f"DEBUG: Query [b={b_idx_print}] is padding.")
        if n_rules > 0:
             print(f"DEBUG: First Rule Head [r=0]: {_format_atom(rule_heads[0].cpu(), predicate_idx2str, constant_idx2str, vars_idx_set)}")
        else:
             print("DEBUG: No rules provided.")

    if bs > 0 and n_rules > 0:
        queries_expanded = first_queries.unsqueeze(1)
        heads_expanded = rule_heads.unsqueeze(0)
        pred_match = (queries_expanded[:, :, 0] == heads_expanded[:, :, 0])
    else:
        pred_match = torch.zeros((bs, n_rules), dtype=torch.bool, device=device)

    potential_subs = []
    potential_bodies = []
    potential_valid = torch.zeros((bs, n_rules), dtype=torch.bool, device=device)

    for b in range(bs):
        batch_subs_for_b = []
        batch_bodies_for_b = []
        if not has_valid_query[b]:
            potential_subs.append(batch_subs_for_b)
            potential_bodies.append(batch_bodies_for_b)
            continue

        query = first_queries[b]
        q_p, q_s, q_o = query.tolist()
        is_q_s_var = q_s in vars_idx_set
        is_q_o_var = q_o in vars_idx_set

        for r in range(n_rules):
            is_target_state = (b == 1) # Debug flag for state 1

            if not pred_match[b, r]: continue

            head = rule_heads[r]
            h_p, h_s, h_o = head.tolist()
            is_h_s_var = h_s in vars_idx_set
            is_h_o_var = h_o in vars_idx_set

            temp_subs = {}
            possible = True

            # --- Corrected get_binding function ---
            def get_binding(var, subs_dict):
                """Finds the ultimate binding for a variable, handling chains and cycles."""
                # If var is not a variable or not in subs_dict, return var itself.
                if var not in vars_idx_set or var not in subs_dict:
                    return var

                current_var = var
                visited = {current_var}
                count = 0
                max_depth = max_subs_per_rule * 2 # Limit depth

                while current_var in subs_dict and count < max_depth:
                    next_val = subs_dict.get(current_var)
                    if next_val == current_var: return None # Direct cycle X->X
                    if next_val in visited: return None  # Longer cycle detected
                    # If next_val is not a variable or not further bound, return it
                    if next_val not in vars_idx_set or next_val not in subs_dict:
                        return next_val
                    # Continue chain
                    visited.add(next_val)
                    current_var = next_val
                    count += 1

                if count >= max_depth:
                     print(f"Warning: get_binding max depth reached for var {var}")
                return current_var # Return last var in chain (or original if unbound)
            # --- End of corrected get_binding ---


            term_pairs_info = [(q_s, h_s, is_q_s_var, is_h_s_var, "Subject"), (q_o, h_o, is_q_o_var, is_h_o_var, "Object")]

            for q_term, h_term, is_q_var, is_h_var, term_name in term_pairs_info:
                if not possible: break

                q_final = get_binding(q_term, temp_subs) if is_q_var else q_term
                h_final = get_binding(h_term, temp_subs) if is_h_var else h_term

                if q_final is None or h_final is None: # Cycle detected by get_binding
                     possible = False; break

                is_q_final_var = q_final in vars_idx_set
                is_h_final_var = h_final in vars_idx_set

                # --- MODIFIED Unification Cases ---
                if q_final == h_final:
                    # Match (constants or same variable)
                    continue

                # Case 1: Both are variables -> Bind H_var -> Q_var (User Request)
                elif is_q_final_var and is_h_final_var:
                    # Check for conflicts before binding H -> Q
                    h_binding_check = get_binding(h_final, temp_subs) # Should return h_final if unbound
                    q_binding_check = get_binding(q_final, temp_subs) # Should return q_final if unbound
                    # If H is already bound to something else (that isn't Q) -> Conflict
                    if h_binding_check != h_final and h_binding_check != q_final: possible = False; break
                    # If Q is already bound to something else (that isn't H) -> Conflict
                    if q_binding_check != q_final and q_binding_check != h_final: possible = False; break
                    # Bind H_var -> Q_var
                    temp_subs[h_final] = q_final

                # Case 2: Query is variable, Head is constant -> Bind Q_var -> H_Const
                elif is_q_final_var and not is_h_final_var:
                    q_binding_check = get_binding(q_final, temp_subs)
                    # Conflict if Q already bound to a DIFFERENT constant
                    if q_binding_check not in vars_idx_set and q_binding_check != h_final: possible = False; break
                    # Conflict if Q already bound to a variable H' which resolves to a different constant
                    if q_binding_check in vars_idx_set:
                         q_binding_final = get_binding(q_binding_check, temp_subs)
                         if q_binding_final is not None and q_binding_final not in vars_idx_set and q_binding_final != h_final: possible = False; break
                    temp_subs[q_final] = h_final

                # Case 3: Head is variable, Query is constant -> Bind H_var -> Q_Const
                elif not is_q_final_var and is_h_final_var:
                    h_binding_check = get_binding(h_final, temp_subs)
                     # Conflict if H already bound to a DIFFERENT constant
                    if h_binding_check not in vars_idx_set and h_binding_check != q_final: possible = False; break
                    # Conflict if H already bound to a variable Q' which resolves to a different constant
                    if h_binding_check in vars_idx_set:
                         h_binding_final = get_binding(h_binding_check, temp_subs)
                         if h_binding_final is not None and h_binding_final not in vars_idx_set and h_binding_final != q_final: possible = False; break
                    temp_subs[h_final] = q_final

                # Case 4: Both are constants (and already checked they are not equal)
                else:
                    possible = False
                    break
                # --- End of MODIFIED Unification Cases ---


            if possible:
                 # Final conflict check (redundant checks removed, rely on get_binding)
                 final_bindings = {}
                 possible_final = True
                 for var in temp_subs:
                     final_val = get_binding(var, temp_subs)
                     if final_val is None: possible_final = False; break # Cycle detected
                     # Check for direct conflict: var already finalized to something different
                     if var in final_bindings and final_bindings[var] != final_val:
                          possible_final = False; break
                     final_bindings[var] = final_val

                 if possible_final:
                     subs_list = [[var, val if val is not None else UNBOUND_VAR] for var, val in final_bindings.items()]
                     actual_subs_len = len(final_bindings)
                     if len(subs_list) > max_subs_per_rule:
                         subs_list = subs_list[:max_subs_per_rule]
                     while len(subs_list) < max_subs_per_rule:
                         subs_list.append([PADDING_VALUE, PADDING_VALUE])

                     if is_target_state and verbose: print(f"    [b=1, r={r}] Storing SUCCESSFUL unification. Subs: {subs_list[:actual_subs_len]}")

                     potential_valid[b, r] = True
                     batch_subs_for_b.append((r, torch.tensor(subs_list, dtype=torch.long, device=device)))
                     body = rule_bodies[r] if rule_body_len > 0 else torch.empty((0, 3), dtype=torch.long, device=device)
                     batch_bodies_for_b.append((r, body))


        potential_subs.append(batch_subs_for_b)
        potential_bodies.append(batch_bodies_for_b)

    # --- Minimal Debug Print ---
    if verbose:
        print("\nDEBUG: After Rule Unification Loop:")
        print(f"  potential_valid:\n{potential_valid.int()}")
        if bs > 1: print(f"  potential_valid[1]: {potential_valid[1].int()}")


    if potential_valid.numel() > 0:
         scores = torch.where(potential_valid, torch.tensor(1.0, device=device), torch.tensor(-1.0, device=device))
    else:
         scores = torch.full((bs, n_rules), -1.0, device=device)

    if actual_k == 0:
        if verbose: print("\nDEBUG: actual_k is 0 for rules, skipping topk.")
        return substitutions_out, rule_indices_out, rule_bodies_out, valid_mask_out

    if n_rules == 0 and actual_k > 0:
         if verbose: print("\nDEBUG: No rules available, skipping topk.")
         return substitutions_out, rule_indices_out, rule_bodies_out, valid_mask_out

    if scores.shape[1] > 0 :
        top_scores, top_indices_in_rules = torch.topk(scores, k=actual_k, dim=1)
        final_valid_k_mask = (top_scores > -1.0)
        # --- Minimal Debug Print ---
        if verbose and bs > 1:
            print("\nDEBUG: TopK Results for State 1 (b=1):")
            print(f"  Scores[1]: {scores[1].tolist()}")
            print(f"  Top Scores[1]: {top_scores[1].tolist()}")
            print(f"  Top Indices[1]: {top_indices_in_rules[1].tolist()}")
            print(f"  Final Valid Mask[1]: {final_valid_k_mask[1].tolist()}")
    else:
        top_indices_in_rules = torch.full((bs, actual_k), PADDING_VALUE, dtype=torch.long, device=device)
        final_valid_k_mask = torch.zeros((bs, actual_k), dtype=torch.bool, device=device)


    selected_rule_indices = torch.full((bs, actual_k), PADDING_VALUE, dtype=torch.long, device=device)
    selected_subs = torch.full((bs, actual_k, max_subs_per_rule, 2), PADDING_VALUE, dtype=torch.long, device=device)
    selected_bodies = torch.full((bs, actual_k, max(0, rule_body_len), 3), PADDING_VALUE, dtype=torch.long, device=device)

    if len(potential_subs) != bs or len(potential_bodies) != bs:
         print(f"ERROR DEBUG: Length mismatch! bs={bs}, len(potential_subs)={len(potential_subs)}, len(potential_bodies)={len(potential_bodies)}")
         return substitutions_out, rule_indices_out, rule_bodies_out, valid_mask_out

    for b in range(bs):
        if b >= len(potential_subs) or b >= len(potential_bodies):
             if verbose: print(f"ERROR DEBUG: Index b={b} is out of range for potential_subs/bodies during mapping.")
             continue

        current_batch_subs = potential_subs[b]
        current_batch_bodies = potential_bodies[b]
        rule_to_list_idx_map = {item[0]: idx for idx, item in enumerate(current_batch_subs)} if current_batch_subs else {}
        body_rule_to_list_idx_map = {item[0]: idx for idx, item in enumerate(current_batch_bodies)} if current_batch_bodies else {}

        for k_idx in range(actual_k):
             # Check bounds for final_valid_k_mask and top_indices_in_rules
            if b < final_valid_k_mask.shape[0] and k_idx < final_valid_k_mask.shape[1] and final_valid_k_mask[b, k_idx]:
                if b < top_indices_in_rules.shape[0] and k_idx < top_indices_in_rules.shape[1]:
                    rule_idx = top_indices_in_rules[b, k_idx].item()
                    selected_rule_indices[b, k_idx] = rule_idx

                    if rule_idx in rule_to_list_idx_map:
                        list_idx = rule_to_list_idx_map[rule_idx]
                        if 0 <= list_idx < len(current_batch_subs):
                             selected_subs[b, k_idx] = current_batch_subs[list_idx][1]
                        elif verbose: print(f"DEBUG WARNING: Mapped list_idx {list_idx} invalid for rule {rule_idx} in potential_subs[{b}]")

                    if rule_idx in body_rule_to_list_idx_map:
                         body_list_idx = body_rule_to_list_idx_map[rule_idx]
                         if 0 <= body_list_idx < len(current_batch_bodies):
                             body_tensor = current_batch_bodies[body_list_idx][1]
                             if body_tensor.shape[0] == selected_bodies.shape[2]:
                                 selected_bodies[b, k_idx] = body_tensor
                             elif verbose: print(f"DEBUG WARNING: Body tensor shape mismatch for rule {rule_idx} in potential_bodies[{b}]")
                         elif verbose: print(f"DEBUG WARNING: Mapped body_list_idx {body_list_idx} invalid for rule {rule_idx} in potential_bodies[{b}]")
                else:
                     if verbose: print(f"DEBUG WARNING: Index out of bounds for top_indices_in_rules[{b}, {k_idx}]")
            # else: unification not valid for this k_idx, leave as padding


    substitutions_out[:, :actual_k] = selected_subs[:, :actual_k]
    rule_indices_out[:, :actual_k] = selected_rule_indices[:, :actual_k]
    rule_bodies_out[:, :actual_k] = selected_bodies[:, :actual_k]
    valid_mask_out[:, :actual_k] = final_valid_k_mask[:, :actual_k]

    # Verbose Output section (remains the same)
    if verbose and bs > 0:
        b_idx_print = 1 if bs > 1 else 0
        print(f"\nDEBUG: Rule Top-{actual_k} Selection & Results [Batch Element {b_idx_print}]:")
        if final_valid_k_mask.shape[0] > b_idx_print and top_indices_in_rules.shape[0] > b_idx_print:
            valid_indices_print = top_indices_in_rules[b_idx_print][final_valid_k_mask[b_idx_print]].tolist()
            print(f"  Selected Rule Indices (Valid in Top K): {valid_indices_print}")

            for k_idx in range(actual_k):
                if valid_mask_out[b_idx_print, k_idx]:
                    rule_idx = rule_indices_out[b_idx_print, k_idx].item()
                    subs_tensor = substitutions_out[b_idx_print, k_idx]
                    body_tensor = rule_bodies_out[b_idx_print, k_idx]

                    rule_head_str = "[Rule head invalid index]"
                    if 0 <= rule_idx < n_rules:
                         rule_head_str = _format_atom(rule_heads[rule_idx].cpu(), predicate_idx2str, constant_idx2str, vars_idx_set)
                    print(f"  - K-Slot {k_idx}: Unifies with Rule {rule_idx} (Head: {rule_head_str})")

                    subs_strs = []
                    for sub_idx in range(max_subs_per_rule):
                        sub_pair = subs_tensor[sub_idx]
                        if sub_pair[0].item() != PADDING_VALUE:
                            subs_strs.append(_format_substitution(sub_pair.cpu(), constant_idx2str, vars_idx_set))
                    print(f"    Subs: {{{', '.join(subs_strs)}}}")

                    print(f"    Rule Body Atoms:")
                    has_body = False
                    if body_tensor.shape[0] > 0:
                        for atom_idx in range(body_tensor.shape[0]):
                            body_atom = body_tensor[atom_idx]
                            if body_atom[0].item() != PADDING_VALUE and not torch.all(body_atom == 0):
                                 print(f"      {_format_atom(body_atom.cpu(), predicate_idx2str, constant_idx2str, vars_idx_set)}")
                                 has_body = True
                    if not has_body:
                        print("      [No body atoms or all padding]")
        else:
             print(f"  No results to display for batch {b_idx_print} (likely bs too small or k=0).")


        print("-" * 40)
        print(f"DEBUG: End Rule Unification Verbose Output [{b_idx_print}]")
        print("=" * 40)


    return substitutions_out, rule_indices_out, rule_bodies_out, valid_mask_out


# --- NEW: Rule Unification Application ---
def generate_next_states_from_rules(
    states_idx: torch.Tensor,          # Shape: (bs, 1, n_padding_atoms, 3)
    rule_substitutions: torch.Tensor,  # Shape: (bs, k, max_subs, 2) from rule unify
    rule_bodies: torch.Tensor,         # Shape: (bs, k, rule_body_len, 3) from rule unify
    valid_mask: torch.Tensor,          # Shape: (bs, k) from rule unify
    vars_idx_tensor: torch.Tensor,     # Tensor of variable indices
    new_max_atoms: Optional[int] = None, # Max atoms in the output state, None to calculate
    PADDING_VALUE: int = -999,
    UNBOUND_VAR: int = -998
) -> torch.Tensor:
    """
    Generates next states based on rule unifications.
    Applies substitutions to the rule body and remaining queries,
    then concatenates them.
    (Function body omitted for brevity as it's unchanged)
    """
    bs, _, n_padding_atoms, _ = states_idx.shape
    k = rule_substitutions.shape[1]
    rule_body_len = rule_bodies.shape[2]
    device = states_idx.device

    # Determine output state length
    num_remaining_atoms_orig = max(0, n_padding_atoms - 1)
    if new_max_atoms is None:
        new_max_atoms = num_remaining_atoms_orig + rule_body_len
    if new_max_atoms < 0: new_max_atoms = 0 # Handle edge case

    # Output tensor initialized with padding
    next_states = torch.full((bs, k, new_max_atoms, 3), PADDING_VALUE, dtype=torch.long, device=device)

    # Get original remaining queries (all atoms except the first one)
    # Shape: (bs, num_remaining_atoms_orig, 3)
    remaining_queries_base = states_idx[:, 0, 1:, :] if n_padding_atoms > 1 else torch.empty((bs, 0, 3), dtype=torch.long, device=device)

    for b in range(bs):
        for k_idx in range(k):
             # Check bounds for valid_mask before accessing
            if b < valid_mask.shape[0] and k_idx < valid_mask.shape[1] and valid_mask[b, k_idx]:
                # Check bounds for other tensors
                if b < rule_substitutions.shape[0] and k_idx < rule_substitutions.shape[1] and \
                   b < rule_bodies.shape[0] and k_idx < rule_bodies.shape[1]:

                    subs = rule_substitutions[b, k_idx] # Shape: (max_subs, 2)
                    body = rule_bodies[b, k_idx]        # Shape: (rule_body_len, 3)
                    rem_q = remaining_queries_base[b] if b < remaining_queries_base.shape[0] else torch.empty((0,3), dtype=torch.long, device=device) # Handle potential empty base

                    subst_body = apply_substitutions_general(body, subs, vars_idx_tensor, PADDING_VALUE=PADDING_VALUE, UNBOUND_VAR=UNBOUND_VAR)
                    subst_rem_q = apply_substitutions_general(rem_q, subs, vars_idx_tensor, PADDING_VALUE=PADDING_VALUE, UNBOUND_VAR=UNBOUND_VAR)

                    concatenated_atoms = torch.cat((subst_body.to(device), subst_rem_q.to(device)), dim=0)

                    current_len = concatenated_atoms.shape[0]
                    final_atoms = torch.full((new_max_atoms, 3), PADDING_VALUE, dtype=torch.long, device=device)

                    len_to_copy = min(current_len, new_max_atoms)
                    if len_to_copy > 0:
                        final_atoms[:len_to_copy, :] = concatenated_atoms[:len_to_copy, :]

                    # Check bounds for next_states before assignment
                    if b < next_states.shape[0] and k_idx < next_states.shape[1]:
                        next_states[b, k_idx] = final_atoms
                    else:
                         print(f"Warning: Index out of bounds during next_states assignment for b={b}, k_idx={k_idx}")
                else:
                     print(f"Warning: Index out of bounds accessing rule_substitutions/rule_bodies for b={b}, k_idx={k_idx}")


    return next_states


# --- Example Usage ---
if __name__ == '__main__':
    device = 'cpu'
    print(f"Using device: {device}")

    K_FACTS = 3 # Max unifications from facts
    K_RULES = 2 # Max unifications from rules
    print(f"Using PADDING_VALUE: {PADDING_VALUE}")
    print(f"Using UNBOUND_VAR: {UNBOUND_VAR}")


    # --- Mappings for Readability ---
    predicate_map = {1: "relatedTo", 2: "typeOf", 3: "hasProperty", 4: "parent", 5: "ancestor"}
    constant_map = {
        10: "ObjA", 11: "ObjB", 12: "ObjC", 13: "ObjD", 14: "ObjE", 15: "ObjF",
        20: "PropX", 21: "PropY", 22: "PropZ",
        30: "Person1", 31: "Person2", 32: "Person3"
    }
    # Variables are negative indices. Assume shared pool for query and rules for simplicity.
    vars_idx_list = [-1, -2, -3, -10, -11, -12] # Added rule variables
    vars_tensor = torch.tensor(vars_idx_list, dtype=torch.long, device=device)
    vars_set = set(vars_idx_list)

    # --- States ---
    states = torch.tensor([
        # State 0: Query (relatedTo, VAR(-1), ObjA), then (typeOf, ObjB, VAR(-2))
        [[[1, -1, 10], [2, 11, -2], [0, 0, 0], [0, 0, 0]]],
        # State 1: Query (ancestor, Person1, VAR(-1)), then (relatedTo, VAR(-1), ObjC)
        [[[5, 30, -1], [1, -1, 12], [0, 0, 0], [0, 0, 0]]],
         # State 2: Query (typeOf, VAR(-1), VAR(-2)), then (relatedTo, VAR(-1), ObjA)
        [[[2, -1, -2], [1, -1, 10], [0, 0, 0], [0, 0, 0]]],
        # State 3: Query (hasProperty, VAR(-1), VAR(-1)), then (typeOf, VAR(-1), ObjC)
        [[[3, -1, -1], [2, -1, 12], [0, 0, 0], [0, 0, 0]]],
        # State 4: Padding query
        [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]]
    ], dtype=torch.long, device=device)
    bs, _, n_atoms_state, _ = states.shape

    # --- Facts ---
    facts_tensor = torch.tensor([
        [1, 11, 10], # F0: (relatedTo, ObjB, ObjA) -> Unifies S0, Sub: {-1: 11}
        [1, 12, 13], # F1: (relatedTo, ObjC, ObjD)
        [1, 14, 10], # F2: (relatedTo, ObjE, ObjA) -> Unifies S0, Sub: {-1: 14}
        [2, 10, 11], # F3: (typeOf, ObjA, ObjB)   -> Unifies S2, Sub: {-1: 10, -2: 11}
        [2, 12, 12], # F4: (typeOf, ObjC, ObjC)   -> Unifies S2, Sub: {-1: 12, -2: 12}
        [1, 15, 10], # F5: (relatedTo, ObjF, ObjA) -> Unifies S0, Sub: {-1: 15}
        [3, 20, 20], # F6: (hasProperty, PropX, PropX) -> Unifies S3, Sub: {-1: 20}
        [3, 21, 22], # F7: (hasProperty, PropY, PropZ) -> Conflict with S3 query, ignored by fact unify
        [4, 30, 31], # F8: (parent, Person1, Person2) -> Used by rule R0/R1
        # [4, 31, 32], # F9: OLD - (parent, Person2, Person3)
        [5, 31, 32], # F9: NEW - (ancestor, Person2, Person3) -> Base case for ancestor rule R1
    ], dtype=torch.long, device=device)

    # --- Rules ---
    # Format: (n_rules, max_rule_atoms, 3). Head is atom 0. Body follows.
    # Using vars -10, -11, -12 for rules
    rules_tensor = torch.tensor([
        # Rule 0: ancestor(X, Y) :- parent(X, Y)
        # Head: (5, -10, -11), Body: (4, -10, -11)
        [[5, -10, -11], [4, -10, -11], [0, 0, 0]],
        # Rule 1: ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z)
        # Head: (5, -10, -12), Body: (4, -10, -11), (5, -11, -12)
        [[5, -10, -12], [4, -10, -11], [5, -11, -12]]
    ], dtype=torch.long, device=device)
    _, n_atoms_rule, _ = rules_tensor.shape
    rule_body_len = n_atoms_rule - 1

    # --- Run Fact Unification ---
    print("\n" + "#" * 20 + " FACT UNIFICATION " + "#" * 20)
    fact_subs, fact_indices, fact_valid_mask = batch_unify_with_facts(
        states, facts_tensor, vars_tensor, k=K_FACTS, device=device,
        verbose=False, # Disable verbose for facts for brevity now
        predicate_idx2str=predicate_map,
        constant_idx2str=constant_map
    )

    # --- Apply Fact Substitutions ---
    print("\n" + "#" * 20 + " APPLYING FACT SUBSTITUTIONS " + "#" * 20)
    next_states_from_facts = apply_fact_substitutions_to_rest(
        states,
        fact_subs,
        fact_valid_mask,
        PADDING_VALUE=PADDING_VALUE
    )
    num_remaining_atoms_fact = next_states_from_facts.shape[2]

    # --- Run Rule Unification ---
    print("\n" + "#" * 20 + " RULE UNIFICATION " + "#" * 20)
    MAX_SUBS_PER_RULE = 5 # Set a limit
    rule_subs, rule_indices, rule_bodies, rule_valid_mask = batch_unify_with_rules(
         states, rules_tensor, vars_tensor, k=K_RULES,
         max_subs_per_rule=MAX_SUBS_PER_RULE,
         device=device,
         verbose=True, # Enable verbose for rules (will focus on state 1 if bs > 1)
         predicate_idx2str=predicate_map,
         constant_idx2str=constant_map
    )

    # --- Apply Rule Substitutions ---
    print("\n" + "#" * 20 + " APPLYING RULE SUBSTITUTIONS " + "#" * 20)
    # Calculate max possible atoms in next state from rules
    output_atoms_rules = (n_atoms_state - 1) + rule_body_len
    next_states_from_rules = generate_next_states_from_rules(
        states,
        rule_subs,
        rule_bodies,
        rule_valid_mask,
        vars_tensor,
        new_max_atoms=output_atoms_rules, # Pad/truncate to this length
        PADDING_VALUE=PADDING_VALUE,
        UNBOUND_VAR=UNBOUND_VAR
    )
    num_remaining_atoms_rule = next_states_from_rules.shape[2]


    # --- Interpreted Results ---
    print("\n" + "=" * 20 + " INTERPRETED RESULTS " + "=" * 20)
    for b in range(bs):
        query_atom = states[b, 0, 0, :]
        query_str = _format_atom(query_atom.cpu(), predicate_map, constant_map, vars_set)
        print(f"\n--- State {b}: Original First Query = {query_str} ---")
        if n_atoms_state > 1:
             print(f"  Original Remaining Queries:")
             for atom_idx in range(1, n_atoms_state):
                 rem_atom = states[b, 0, atom_idx, :]
                 if not torch.all(rem_atom == 0) and not torch.all(rem_atom == PADDING_VALUE):
                     print(f"    {_format_atom(rem_atom.cpu(), predicate_map, constant_map, vars_set)}")
        else: print("  No remaining queries in original state.")

        is_state_valid = not torch.all(query_atom == 0) and not torch.all(query_atom == PADDING_VALUE)
        if not is_state_valid:
            print("  Original query is padding. Skipping results.")
            continue

        # Print Fact Results
        print(f"\n  Fact Unification Results (K_Facts={K_FACTS}):")
        found_fact_unification = False
        for k_idx in range(K_FACTS):
            # Check bounds for fact_valid_mask
            if b < fact_valid_mask.shape[0] and k_idx < fact_valid_mask.shape[1] and fact_valid_mask[b, k_idx]:
                found_fact_unification = True
                fact_idx = fact_indices[b, k_idx].item()
                fact_str = "[Fact not found]"
                if fact_idx != PADDING_VALUE and 0 <= fact_idx < len(facts_tensor):
                     fact_str = _format_atom(facts_tensor[fact_idx].cpu(), predicate_map, constant_map, vars_set)
                else: fact_str = f"[Invalid Fact Idx: {fact_idx}]"

                s_sub = fact_subs[b, k_idx, 0, :]
                o_sub = fact_subs[b, k_idx, 1, :]
                subs_list = []
                if s_sub[0].item() != PADDING_VALUE: subs_list.append(_format_substitution(s_sub.cpu(), constant_map, vars_set))
                if o_sub[0].item() != PADDING_VALUE:
                     is_s_sub_valid = s_sub[0].item() != PADDING_VALUE
                     is_same_var = s_sub[0].item() == o_sub[0].item()
                     if not (is_s_sub_valid and is_same_var): subs_list.append(_format_substitution(o_sub.cpu(), constant_map, vars_set))
                subs_str = ", ".join(subs_list) if subs_list else "{}"

                print(f"  - [Slot {k_idx}]: Unifies with Fact {fact_idx} {fact_str}, Subs: {subs_str}")

                if num_remaining_atoms_fact > 0:
                    print(f"    -> Next State Queries (Fact Slot {k_idx}):")
                    has_next_query = False
                    # Check bounds for next_states_from_facts
                    if b < next_states_from_facts.shape[0] and k_idx < next_states_from_facts.shape[1]:
                        for atom_idx in range(num_remaining_atoms_fact):
                            next_atom = next_states_from_facts[b, k_idx, atom_idx, :]
                            if next_atom[0].item() != PADDING_VALUE and not torch.all(next_atom == 0):
                                print(f"       {_format_atom(next_atom.cpu(), predicate_map, constant_map, vars_set)}")
                                has_next_query = True
                    if not has_next_query: print("       [No remaining non-padding queries or invalid indices]")
                else: print("    -> Next State Queries: [None (no remaining queries)]")

        if not found_fact_unification: print("  No valid fact unifications found.")

        # Print Rule Results
        print(f"\n  Rule Unification Results (K_Rules={K_RULES}):")
        found_rule_unification = False
        for k_idx in range(K_RULES):
             # Check bounds for rule_valid_mask
             if b < rule_valid_mask.shape[0] and k_idx < rule_valid_mask.shape[1] and rule_valid_mask[b, k_idx]:
                 found_rule_unification = True
                 rule_idx = rule_indices[b, k_idx].item()
                 rule_head_str = "[Rule head not found]"
                 if rule_idx != PADDING_VALUE and 0 <= rule_idx < len(rules_tensor):
                      rule_head_str = _format_atom(rules_tensor[rule_idx, 0, :].cpu(), predicate_map, constant_map, vars_set)
                 else: rule_head_str = f"[Invalid Rule Idx: {rule_idx}]"

                 subs_tensor = rule_subs[b, k_idx]
                 subs_strs = []
                 for sub_i in range(subs_tensor.shape[0]):
                     sub_pair = subs_tensor[sub_i]
                     if sub_pair[0].item() != PADDING_VALUE:
                         subs_strs.append(_format_substitution(sub_pair.cpu(), constant_map, vars_set))
                 subs_str = ", ".join(subs_strs) if subs_strs else "{}"

                 print(f"  - [Slot {k_idx}]: Unifies with Rule {rule_idx} (Head: {rule_head_str}), Subs: {subs_str}")

                 if num_remaining_atoms_rule > 0:
                     print(f"    -> Next State Queries (Rule Slot {k_idx}):")
                     has_next_query = False
                     # Check bounds for next_states_from_rules
                     if b < next_states_from_rules.shape[0] and k_idx < next_states_from_rules.shape[1]:
                         for atom_idx in range(num_remaining_atoms_rule):
                             next_atom = next_states_from_rules[b, k_idx, atom_idx, :]
                             is_padding = torch.all(next_atom == 0) or torch.all(next_atom == PADDING_VALUE)
                             if next_atom[0].item() != PADDING_VALUE and not is_padding:
                                 print(f"       {_format_atom(next_atom.cpu(), predicate_map, constant_map, vars_set)}")
                                 has_next_query = True
                     if not has_next_query: print("       [No remaining non-padding queries or invalid indices]")
                 else: print("    -> Next State Queries: [None (no remaining queries)]")


        if not found_rule_unification: print("  No valid rule unifications found.")

