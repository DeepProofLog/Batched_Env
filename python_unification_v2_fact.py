import torch
from typing import Tuple, Optional, Union, List, Set, Dict

# Define a suitable padding value (must not be a valid index, variable, or constant)
# Use -999 instead of -1 to avoid collision with variable index -1
PADDING_VALUE = -999

def _format_atom(
    atom_tensor: torch.Tensor,
    predicate_idx2str: Optional[Dict[int, str]] = None,
    constant_idx2str: Optional[Dict[int, str]] = None,
    vars_idx: Optional[Union[torch.Tensor, List[int], Set[int]]] = None
) -> str:
    """Helper function to format an atom tensor into a readable string."""
    # Check if input is a tensor of the correct shape
    if not isinstance(atom_tensor, torch.Tensor) or atom_tensor.shape != (3,):
        return str(atom_tensor) # Fallback for unexpected input

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
    # Check for the specific padding atom [0, 0, 0]
    elif p_idx == 0 and s_idx == 0 and o_idx == 0:
        return "[padding_atom]" # Changed label for clarity

    # Format subject
    s_str = str(s_idx)
    is_s_var = s_idx in vars_idx_set # Check against the set of variable indices
    if is_s_var:
        s_str = f"VAR({s_idx})"
    elif constant_idx2str and s_idx in constant_idx2str:
        s_str = constant_idx2str[s_idx]
    # Check against the NEW PADDING_VALUE
    elif s_idx == PADDING_VALUE:
         s_str = "[pad_val]" # Changed label for clarity
    # Handle 0 constant if predicate is not 0 (to distinguish from padding atom)
    elif s_idx == 0 and p_idx != 0:
         s_str = str(s_idx)

    # Format object
    o_str = str(o_idx)
    is_o_var = o_idx in vars_idx_set # Check against the set of variable indices
    if is_o_var:
        o_str = f"VAR({o_idx})"
    elif constant_idx2str and o_idx in constant_idx2str:
        o_str = constant_idx2str[o_idx]
    # Check against the NEW PADDING_VALUE
    elif o_idx == PADDING_VALUE:
         o_str = "[pad_val]" # Changed label for clarity
    # Handle 0 constant if predicate is not 0
    elif o_idx == 0 and p_idx != 0:
         o_str = str(o_idx)


    return f"({p_str}, {s_str}, {o_str})"

def _format_substitution(
    sub_pair: torch.Tensor,
    constant_idx2str: Optional[Dict[int, str]] = None
) -> str:
    """Helper function to format a substitution pair. Assumes sub_pair[0] is not PADDING_VALUE."""
    # Check input type and shape
    if not isinstance(sub_pair, torch.Tensor) or sub_pair.shape != (2,):
        return f"[invalid sub pair: {sub_pair}]"

    var_idx, const_idx = sub_pair.tolist()

    # Fallback check against the NEW PADDING_VALUE
    if var_idx == PADDING_VALUE:
        return "[no_sub]"

    # Format the constant part of the substitution
    const_str = str(const_idx)
    if constant_idx2str and const_idx in constant_idx2str:
        const_str = constant_idx2str[const_idx]
    # Check against the NEW PADDING_VALUE
    elif const_idx == PADDING_VALUE:
        const_str = "[pad_val]"


    return f"VAR({var_idx}) -> {const_str}"


def batch_unify_arbitrary_k(
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
    Performs batched unification for the first query (atom at index 0),
    returning substitutions for up to K unifying facts (arbitrarily selected
    if more than K exist) in fixed-size tensors.
    Includes a verbose mode for debugging the first batch element.
    Assumes non-padding atoms always precede padding atoms in states_idx.

    Args:
        states_idx: Tensor containing batch of states. Assumes non-padding atoms
                    are first, followed by padding atoms ([0,0,0]).
                    Shape: (bs, 1, n_padding_atoms, 3).
        facts: Tensor containing ground facts. Shape: (n_facts, 3).
        vars_idx: Tensor, list, or set of variable indices.
        k: The maximum number of unifications to find and return per query.
        device: The torch device ('cuda', 'cpu', etc.).
        verbose: If True, print detailed debugging info for the first batch element.
        predicate_idx2str: Dictionary mapping predicate indices to strings.
        constant_idx2str: Dictionary mapping constant indices to strings.

    Returns:
        A tuple containing:
        - substitutions (torch.Tensor): Shape (bs, k, 2, 2). Stores
          [variable_idx, constant_idx] pairs. Dim k represents the k-th
          unification found (arbitrarily selected). Dim 2 represents
          subject (idx 0) or object (idx 1) substitution. Padded with PADDING_VALUE.
        - fact_indices (torch.Tensor): Shape (bs, k). Stores the index of the
          fact corresponding to the k-th unification slot. Padded with PADDING_VALUE.
        - valid_mask (torch.Tensor): Shape (bs, k). Boolean mask indicating
          which slots in the k dimension contain valid results (successful, non-conflicting unification).
    """
    # --- Setup ---
    if k <= 0:
        raise ValueError("k must be a positive integer")

    if device is None:
        device = states_idx.device
    else:
        # Ensure input tensors are on the specified device
        states_idx = states_idx.to(device)
        facts = facts.to(device)

    # Ensure vars_idx is a tensor on the correct device
    if isinstance(vars_idx, (list, set)):
        vars_idx_tensor = torch.tensor(list(vars_idx), dtype=torch.long, device=device)
    elif isinstance(vars_idx, torch.Tensor):
        vars_idx_tensor = vars_idx.to(device)
    else:
        raise TypeError("vars_idx must be a tensor, list, or set")
    # Keep the original set/list form for the helper function if needed
    vars_idx_set = set(vars_idx_tensor.tolist())


    bs, _, n_padding_atoms, _ = states_idx.shape
    n_facts = facts.shape[0]

    if verbose:
        print("-" * 40)
        print(f"DEBUG: batch_unify_arbitrary_k (Batch Element 0)")
        print("-" * 40)
        print(f"Batch size: {bs}, Num padding atoms: {n_padding_atoms}, Num facts: {n_facts}, k: {k}")
        print(f"Device: {device}")
        print(f"Variable indices: {vars_idx_set}")
        print(f"Using PADDING_VALUE: {PADDING_VALUE}") # Print the padding value being used

    # Handle edge cases where k > n_facts or no facts/states
    actual_k = min(k, n_facts) if n_facts > 0 else 0

    # Pre-allocate output tensors with the NEW PADDING_VALUE
    substitutions_out = torch.full((bs, k, 2, 2), PADDING_VALUE, dtype=torch.long, device=device)
    fact_indices_out = torch.full((bs, k), PADDING_VALUE, dtype=torch.long, device=device)
    valid_mask_out = torch.zeros((bs, k), dtype=torch.bool, device=device)

    if bs == 0 or n_facts == 0 or n_padding_atoms == 0: # Added check for n_padding_atoms
        if verbose: print("DEBUG: Early exit (no batch items, facts, or atoms in state).")
        return substitutions_out, fact_indices_out, valid_mask_out # Return empty/padded tensors

    # --- Simplified First Query Extraction ---
    first_queries = states_idx[:, 0, 0, :] # Shape: (bs, 3)
    # Check if the first query is NOT a padding atom [0, 0, 0]
    has_valid_query = ~torch.all(first_queries == 0, dim=-1) # Shape: (bs,)

    if verbose:
        query0 = first_queries[0]
        query0_str = _format_atom(query0, predicate_idx2str, constant_idx2str, vars_idx_set)
        print(f"\nDEBUG: First Query [0]: {query0_str} (Raw: {query0.tolist()})")
        print(f"DEBUG: Has Valid Query [0]: {has_valid_query[0].item()}")
        if not has_valid_query[0].item():
             print("DEBUG: First query is padding, skipping unification for this element.")


    # --- Calculate Unification Mask ---
    queries_expanded = first_queries.unsqueeze(1) # Shape: (bs, 1, 3)
    facts_expanded = facts.unsqueeze(0)           # Shape: (1, n_facts, 3)

    is_query_var_s = torch.isin(queries_expanded[:, :, 1], vars_idx_tensor) # Shape: (bs, 1)
    is_query_var_o = torch.isin(queries_expanded[:, :, 2], vars_idx_tensor) # Shape: (bs, 1)
    is_query_var_s_expanded = is_query_var_s.expand(bs, n_facts) # Shape: (bs, n_facts)
    is_query_var_o_expanded = is_query_var_o.expand(bs, n_facts) # Shape: (bs, n_facts)

    pred_match = (queries_expanded[:, :, 0] == facts_expanded[:, :, 0]) # Shape: (bs, n_facts)

    # Subject matches if: query subject is a variable OR query subject is not a variable AND matches fact subject
    subj_match = is_query_var_s_expanded | (
        (~is_query_var_s_expanded) & (queries_expanded[:, :, 1] == facts_expanded[:, :, 1])
    )
    # Object matches if: query object is a variable OR query object is not a variable AND matches fact object
    obj_match = is_query_var_o_expanded | (
        (~is_query_var_o_expanded) & (queries_expanded[:, :, 2] == facts_expanded[:, :, 2])
    )

    # Unification requires predicate, subject, and object to match (considering variables)
    unifies_mask = pred_match & subj_match & obj_match

    # Only consider unification if the query itself is valid (not padding)
    unifies_mask = unifies_mask & has_valid_query.unsqueeze(1) # Shape: (bs, n_facts)

    if verbose:
        print("\nDEBUG: Unification Check [Batch Element 0]:")
        unifies_mask_0 = unifies_mask[0]
        print(f"  Raw Mask: {unifies_mask_0.tolist()}")
        unifying_fact_indices_0 = torch.nonzero(unifies_mask_0).squeeze(-1)
        # Handle scalar tensor case if only one match
        if unifying_fact_indices_0.dim() == 0:
            unifying_fact_indices_0 = unifying_fact_indices_0.unsqueeze(0)
        unifying_fact_indices_0 = unifying_fact_indices_0.tolist()

        print(f"  Potentially Unifying Fact Indices: {unifying_fact_indices_0}")
        for fact_idx in unifying_fact_indices_0:
            fact_str = _format_atom(facts[fact_idx], predicate_idx2str, constant_idx2str, vars_idx_set)
            print(f"      - Fact {fact_idx}: {fact_str} (Raw: {facts[fact_idx].tolist()})")
        if not unifying_fact_indices_0:
             print("      No facts potentially unify with the first query.")


    # --- Select Arbitrary K Unifying Facts (using topk with constant score) ---
    # Assign score 1.0 where unification happens, -1.0 otherwise. Shape: (bs, n_facts)
    scores = torch.where(unifies_mask, torch.tensor(1.0, device=device), torch.tensor(-1.0, device=device))

    # If actual_k is 0, topk will fail. Handle this.
    if actual_k == 0:
        if verbose: print("\nDEBUG: actual_k is 0, skipping topk.")
        # Output tensors are already padded, just return them
        return substitutions_out, fact_indices_out, valid_mask_out

    # Get top k scores and their original indices (fact indices)
    # Selection among facts with score 1.0 is arbitrary if more than k exist.
    top_scores, top_fact_indices = torch.topk(scores, k=actual_k, dim=1) # (bs, actual_k)

    # Mask for valid entries among the top k (score > -1 indicates a unification was found)
    initial_valid_k_mask = (top_scores > -1.0) # (bs, actual_k)

    # Restore the NEW PADDING_VALUE in indices where mask is false. Shape: (bs, actual_k)
    top_fact_indices = torch.where(initial_valid_k_mask, top_fact_indices, torch.tensor(PADDING_VALUE, dtype=torch.long, device=device))

    if verbose:
        print(f"\nDEBUG: Top-{actual_k} Selection [Batch Element 0]:")
        print(f"  Scores: {scores[0, :].tolist()}")
        print(f"  Top Scores: {top_scores[0, :].tolist()}")
        print(f"  Top Fact Indices (Raw, Padded with {PADDING_VALUE}): {top_fact_indices[0, :].tolist()}")
        print(f"  Initial Valid K Mask: {initial_valid_k_mask[0, :].tolist()}")
        valid_mask_0 = initial_valid_k_mask[0]
        if valid_mask_0.any():
             valid_indices_in_k_0 = top_fact_indices[0][valid_mask_0].tolist()
             print(f"  Selected Fact Indices (Valid in Top K): {valid_indices_in_k_0}")
        else:
             print(f"  Selected Fact Indices (Valid in Top K): []")


    # --- Gather Data for Top K Valid Unifications ---
    # Get the coordinates of valid unifications in the top k as tuples (bs, k) to facilitate indexing
    valid_k_coords = torch.nonzero(initial_valid_k_mask, as_tuple=True) # (bs_coords, k_coords_in_topk)
    n_valid_topk = len(valid_k_coords[0])

    if verbose: print(f"\nDEBUG: Total valid unifications across batch (before conflict check): {n_valid_topk}")

    # Initialize final mask assuming no conflicts initially
    final_valid_mask_slice = torch.zeros((bs, actual_k), dtype=torch.bool, device=device)

    if n_valid_topk > 0:
        # --- Gather Valid Batch and Fact Indices ---
        valid_bs_indices = valid_k_coords[0]           # Which batch items had valid unifications
        valid_k_indices = valid_k_coords[1]            # Which slot (0 to k-1) the unification is in
        valid_fact_indices = top_fact_indices[valid_k_coords] # The actual indices of the unifying facts

        # --- Gather Top K Queries and Facts (Flattened) ---
        # Select the queries and facts corresponding to the valid unifications
        top_k_queries_flat = first_queries[valid_bs_indices] # Shape: (n_valid_topk, 3)
        top_k_facts_flat = facts[valid_fact_indices]         # Shape: (n_valid_topk, 3)

        # --- Check if Substitution is Needed (Flattened) ---
        top_k_is_s_var_flat = is_query_var_s.squeeze(1)[valid_bs_indices] # Shape: (n_valid_topk,)
        top_k_is_o_var_flat = is_query_var_o.squeeze(1)[valid_bs_indices] # Shape: (n_valid_topk,)

        # --- Check Conflicts within the Top K Valid Unifications ---
        # Conflict: Same variable used for subject and object in query, but fact has different constants
        is_same_var_flat = top_k_is_s_var_flat & top_k_is_o_var_flat & (top_k_queries_flat[:, 1] == top_k_queries_flat[:, 2])
        is_diff_const_flat = top_k_facts_flat[:, 1] != top_k_facts_flat[:, 2]
        is_conflict_flat = is_same_var_flat & is_diff_const_flat # Shape: (n_valid_topk,)

        # --- Create Final Validity Mask ---
        # Start with a mask of False, then set True for initially valid unifications that are NOT conflicts
        # Use the coordinates to place the non-conflicted results into the mask
        final_valid_mask_slice[valid_k_coords] = ~is_conflict_flat

        if verbose:
            print("\nDEBUG: Conflict Check [Batch Element 0]:")
            # Find which of the flattened results correspond to batch element 0
            indices_for_batch_0 = (valid_bs_indices == 0).nonzero().squeeze(-1)
            # Handle scalar tensor case
            if indices_for_batch_0.dim() == 0:
                indices_for_batch_0 = indices_for_batch_0.unsqueeze(0)

            if len(indices_for_batch_0) > 0:
                print(f"  Checking {len(indices_for_batch_0)} potential unifications for conflicts:")
                k_indices_0 = valid_k_indices[indices_for_batch_0] # K-slots for batch 0
                fact_indices_0 = valid_fact_indices[indices_for_batch_0] # Fact indices for batch 0
                conflicts_0 = is_conflict_flat[indices_for_batch_0] # Conflict status for batch 0

                for i in range(len(indices_for_batch_0)):
                    flat_idx = indices_for_batch_0[i].item()
                    k_slot = k_indices_0[i].item()
                    fact_idx = fact_indices_0[i].item()
                    is_conflict = conflicts_0[i].item()
                    # Ensure tensors being passed to _format_atom are on CPU if needed by helper
                    query_str = _format_atom(top_k_queries_flat[flat_idx].cpu(), predicate_idx2str, constant_idx2str, vars_idx_set)
                    fact_str = _format_atom(top_k_facts_flat[flat_idx].cpu(), predicate_idx2str, constant_idx2str, vars_idx_set)
                    print(f"      - K-Slot {k_slot}, Fact {fact_idx}: Query={query_str}, Fact={fact_str} -> Conflict: {is_conflict}")
            else:
                print("  No potential unifications selected in Top K for batch 0 to check for conflicts.")
            print(f"  Final Valid Mask Slice [0]: {final_valid_mask_slice[0].tolist()}")


        # --- Populate Output Tensors ---
        # Place the selected fact indices into the output (padded with NEW PADDING_VALUE where not valid)
        fact_indices_out[:, :actual_k] = torch.where(final_valid_mask_slice, top_fact_indices[:, :actual_k], torch.tensor(PADDING_VALUE, dtype=torch.long, device=device))
        # Store the final validity mask (up to k slots)
        valid_mask_out[:, :actual_k] = final_valid_mask_slice

        # --- Calculate and Populate Substitutions ---
        # Get coordinates of the *final* valid unifications
        final_valid_coords = torch.nonzero(final_valid_mask_slice, as_tuple=True) # (bs_coords, k_coords)
        n_final_valid = len(final_valid_coords[0])

        if verbose: print(f"\nDEBUG: Total final valid unifications across batch (after conflict check): {n_final_valid}")

        if n_final_valid > 0:
            # Find which indices in the original flattened arrays correspond to the final valid ones
            # We need to map from (bs, k) coords back to the flat array indices
            temp_map = torch.full_like(final_valid_mask_slice, -1, dtype=torch.long) # Use -1 temporarily
            temp_map[valid_k_coords] = torch.arange(n_valid_topk, device=device) # Map (bs,k) -> flat_idx (before conflict)
            final_valid_indices_in_flat = temp_map[final_valid_coords] # Select using final coords from map based on initial coords

            # Filter out potential -1 indices if final_valid_coords included padded slots somehow (shouldn't happen with correct logic)
            valid_flat_mask = final_valid_indices_in_flat != -1
            final_valid_coords = (final_valid_coords[0][valid_flat_mask], final_valid_coords[1][valid_flat_mask])
            final_valid_indices_in_flat = final_valid_indices_in_flat[valid_flat_mask]
            n_final_valid = len(final_valid_indices_in_flat) # Update count

            if n_final_valid == 0:
                 if verbose: print("DEBUG: No valid substitutions after filtering -1 indices (unexpected).")
            else:
                # Gather the necessary data only for the final valid substitutions
                s_var_indices = top_k_queries_flat[final_valid_indices_in_flat, 1]
                s_const_indices = top_k_facts_flat[final_valid_indices_in_flat, 1]
                o_var_indices = top_k_queries_flat[final_valid_indices_in_flat, 2]
                o_const_indices = top_k_facts_flat[final_valid_indices_in_flat, 2]

                # Determine if substitution is needed for subject/object for these final valid ones
                s_sub_needed_final = top_k_is_s_var_flat[final_valid_indices_in_flat]
                o_sub_needed_final = top_k_is_o_var_flat[final_valid_indices_in_flat]

                # Populate the substitutions tensor using the final valid coordinates
                # Subject substitutions
                s_sub_final_coords = (final_valid_coords[0][s_sub_needed_final], final_valid_coords[1][s_sub_needed_final])
                if len(s_sub_final_coords[0]) > 0:
                    s_vars_to_assign = s_var_indices[s_sub_needed_final]
                    s_consts_to_assign = s_const_indices[s_sub_needed_final]
                    if verbose: # DEBUG PRINT
                        print(f"\nDEBUG ASSIGN S: Target coords shape: {s_sub_final_coords[0].shape}, Var shape: {s_vars_to_assign.shape}, Const shape: {s_consts_to_assign.shape}")
                        # Print for batch 0 if relevant
                        s_assign_idx_0 = (s_sub_final_coords[0] == 0).nonzero().squeeze(-1)
                        if s_assign_idx_0.dim() == 0: s_assign_idx_0 = s_assign_idx_0.unsqueeze(0) # Handle scalar
                        if len(s_assign_idx_0) > 0:
                            print(f"  Assigning S for Batch 0 at K-slots: {s_sub_final_coords[1][s_assign_idx_0].tolist()}")
                            print(f"    Vars : {s_vars_to_assign[s_assign_idx_0].tolist()}")
                            print(f"    Consts: {s_consts_to_assign[s_assign_idx_0].tolist()}")

                    substitutions_out[s_sub_final_coords[0], s_sub_final_coords[1], 0, 0] = s_vars_to_assign
                    substitutions_out[s_sub_final_coords[0], s_sub_final_coords[1], 0, 1] = s_consts_to_assign

                # Object substitutions
                o_sub_final_coords = (final_valid_coords[0][o_sub_needed_final], final_valid_coords[1][o_sub_needed_final])
                if len(o_sub_final_coords[0]) > 0:
                    o_vars_to_assign = o_var_indices[o_sub_needed_final]
                    o_consts_to_assign = o_const_indices[o_sub_needed_final]
                    if verbose: # DEBUG PRINT
                        print(f"\nDEBUG ASSIGN O: Target coords shape: {o_sub_final_coords[0].shape}, Var shape: {o_vars_to_assign.shape}, Const shape: {o_consts_to_assign.shape}")
                        # Print for batch 0 if relevant
                        o_assign_idx_0 = (o_sub_final_coords[0] == 0).nonzero().squeeze(-1)
                        if o_assign_idx_0.dim() == 0: o_assign_idx_0 = o_assign_idx_0.unsqueeze(0) # Handle scalar
                        if len(o_assign_idx_0) > 0:
                            print(f"  Assigning O for Batch 0 at K-slots: {o_sub_final_coords[1][o_assign_idx_0].tolist()}")
                            print(f"    Vars : {o_vars_to_assign[o_assign_idx_0].tolist()}")
                            print(f"    Consts: {o_consts_to_assign[o_assign_idx_0].tolist()}")

                    substitutions_out[o_sub_final_coords[0], o_sub_final_coords[1], 1, 0] = o_vars_to_assign
                    substitutions_out[o_sub_final_coords[0], o_sub_final_coords[1], 1, 1] = o_consts_to_assign

                # --- VERBOSE PRINT OF FINAL SUBSTITUTIONS (Corrected) ---
                if verbose:
                    print("\nDEBUG: Final Populated Substitutions Check [Batch Element 0]:")
                    # Find which of the final valid substitutions correspond to batch element 0
                    final_indices_for_batch_0 = (final_valid_coords[0] == 0).nonzero().squeeze(-1)
                    if final_indices_for_batch_0.dim() == 0: final_indices_for_batch_0 = final_indices_for_batch_0.unsqueeze(0) # Handle scalar

                    if len(final_indices_for_batch_0) > 0:
                        k_indices_final_0 = final_valid_coords[1][final_indices_for_batch_0] # K-slots for batch 0 final
                        print(f"  Checking {len(final_indices_for_batch_0)} final valid substitution slots for batch 0.")

                        for i in range(len(final_indices_for_batch_0)):
                            k_slot = k_indices_final_0[i].item()
                            s_sub_pair = substitutions_out[0, k_slot, 0, :]
                            o_sub_pair = substitutions_out[0, k_slot, 1, :]
                            # Check against NEW PADDING_VALUE before formatting
                            s_sub_str = _format_substitution(s_sub_pair, constant_idx2str) if s_sub_pair[0].item() != PADDING_VALUE else "[no_sub]"
                            o_sub_str = _format_substitution(o_sub_pair, constant_idx2str) if o_sub_pair[0].item() != PADDING_VALUE else "[no_sub]"
                            fact_idx = fact_indices_out[0, k_slot].item() # Get fact index from final output tensor
                            print(f"      - K-Slot {k_slot} (Fact {fact_idx}): Sub S: {s_sub_str}, Sub O: {o_sub_str}")
                            print(f"        Raw Subs Tensor Slice: {substitutions_out[0, k_slot].tolist()}")
                    else:
                        print("  No final valid substitutions to populate/check for batch 0.")


    if verbose:
        print("-" * 40)
        print("DEBUG: End of Verbose Output for Batch Element 0")
        print("-" * 40)

    # Return substitutions up to k, fact indices up to k, and valid mask up to k
    return substitutions_out, fact_indices_out, valid_mask_out


def apply_substitutions_to_rest(
    states_idx: torch.Tensor,      # Shape: (bs, 1, n_padding_atoms, 3)
    substitutions: torch.Tensor,   # Shape: (bs, k, 2, 2) - Output from batch_unify
    valid_mask: torch.Tensor,      # Shape: (bs, k) - Output from batch_unify
    # vars_idx_tensor: torch.Tensor, # Not strictly needed if PADDING_VALUE is used correctly
    PADDING_VALUE: int = -999
) -> torch.Tensor:
    """
    Applies the substitutions found for the first query to the remaining queries
    in each state for each valid unification.

    Args:
        states_idx: The original batch of states.
                    Shape: (bs, 1, n_padding_atoms, 3).
        substitutions: The substitutions tensor from batch_unify_arbitrary_k.
                       Shape: (bs, k, 2, 2).
        valid_mask: The validity mask from batch_unify_arbitrary_k.
                    Shape: (bs, k).
        PADDING_VALUE: The integer value used for padding.

    Returns:
        next_states (torch.Tensor): A tensor containing the resulting states after
                                   applying substitutions. The first query is removed.
                                   Shape: (bs, k, n_padding_atoms - 1, 3).
                                   Invalid slots (where valid_mask is False) will
                                   be filled with PADDING_VALUE.
    """
    bs, _, n_padding_atoms, _ = states_idx.shape
    k = substitutions.shape[1]
    device = states_idx.device

    # Handle case with no remaining queries
    if n_padding_atoms <= 1:
        # Return an empty tensor with the correct shape dimensions (bs, k, 0, 3)
        return torch.empty((bs, k, 0, 3), dtype=torch.long, device=device)

    # --- Prepare Tensors ---
    # Extract remaining queries (all atoms except the first one)
    # Shape: (bs, n_padding_atoms - 1, 3)
    remaining_queries_base = states_idx[:, 0, 1:, :]

    # Expand remaining queries to match the 'k' dimension
    # Shape: (bs, k, n_padding_atoms - 1, 3)
    remaining_queries = remaining_queries_base.unsqueeze(1).expand(bs, k, n_padding_atoms - 1, 3)

    # --- Extract Substitution Components ---
    # Shape: (bs, k, 1) - Unsqueeze to allow broadcasting
    s_var = substitutions[:, :, 0, 0].unsqueeze(-1)
    s_const = substitutions[:, :, 0, 1].unsqueeze(-1)
    o_var = substitutions[:, :, 1, 0].unsqueeze(-1)
    o_const = substitutions[:, :, 1, 1].unsqueeze(-1)

    # --- Create Masks for Valid Substitutions ---
    # Shape: (bs, k, 1) - Indicates if a substitution exists for S/O
    s_sub_valid = (s_var != PADDING_VALUE)
    o_sub_valid = (o_var != PADDING_VALUE)

    # --- Apply Substitutions Vectorized ---
    # Clone the remaining queries to modify them
    applied_queries = remaining_queries.clone()

    # Get original terms from the remaining queries
    # Shape: (bs, k, n_padding_atoms - 1)
    rq_s = remaining_queries[:, :, :, 1]
    rq_o = remaining_queries[:, :, :, 2]

    # Calculate the new subject term after potential substitutions
    new_s = rq_s # Start with original
    # Apply S substitution if subject term matches S variable AND S substitution is valid
    new_s = torch.where((rq_s == s_var) & s_sub_valid, s_const, new_s)
    # Apply O substitution if subject term matches O variable AND O substitution is valid
    # This correctly handles cases where rq_s matches both s_var and o_var if they are the same variable
    # (the conflict check in unify should ensure s_const == o_const in that case)
    new_s = torch.where((rq_s == o_var) & o_sub_valid, o_const, new_s)

    # Calculate the new object term after potential substitutions
    new_o = rq_o # Start with original
    # Apply S substitution if object term matches S variable AND S substitution is valid
    new_o = torch.where((rq_o == s_var) & s_sub_valid, s_const, new_o)
    # Apply O substitution if object term matches O variable AND O substitution is valid
    new_o = torch.where((rq_o == o_var) & o_sub_valid, o_const, new_o)

    # Update the subject and object columns in the applied_queries tensor
    applied_queries[:, :, :, 1] = new_s
    applied_queries[:, :, :, 2] = new_o

    # --- Mask the Results ---
    # Create the final output tensor, initially filled with PADDING_VALUE
    next_states = torch.full_like(applied_queries, PADDING_VALUE, dtype=torch.long, device=device)

    # Use the valid_mask (expanded) to select where to place the results
    # valid_mask shape: (bs, k)
    # applied_queries shape: (bs, k, n_padding_atoms - 1, 3)
    # We need to expand valid_mask to match applied_queries for torch.where
    valid_mask_expanded = valid_mask.unsqueeze(-1).unsqueeze(-1).expand_as(applied_queries)

    # Place the calculated applied_queries into next_states only where the unification was valid
    next_states = torch.where(valid_mask_expanded, applied_queries, next_states)

    return next_states


# --- Example Usage ---
if __name__ == '__main__':
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Force CPU for easier debugging comparison if needed
    device = 'cpu'
    print(f"Using device: {device}")

    K = 3 # Max unifications to keep
    # PADDING_VALUE = -1 # Old value causing issues
    # PADDING_VALUE = -999 # Defined globally now
    print(f"Using PADDING_VALUE: {PADDING_VALUE}")


    # --- Mappings for Readability ---
    predicate_map = {1: "relatedTo", 2: "typeOf", 3: "hasProperty"}
    constant_map = {
        10: "ObjA", 11: "ObjB", 12: "ObjC", 13: "ObjD", 14: "ObjE", 15: "ObjF",
        20: "PropX", 21: "PropY", 22: "PropZ"
    }
    # Variables are negative indices
    vars_idx_list = [-1, -2]
    vars_tensor = torch.tensor(vars_idx_list, dtype=torch.long, device=device)

    states = torch.tensor([
        # State 0: First query (relatedTo, VAR(-1), ObjA), Second query (typeOf, ObjB, VAR(-2))
        [[[1, -1, 10], [2, 11, -2], [0, 0, 0], [0, 0, 0]]],
        # State 1: First query (relatedTo, ObjC, ObjD), No second query
        [[[1, 12, 13], [0, 0, 0], [0, 0, 0], [0, 0, 0]]],
        # State 2: First query (typeOf, VAR(-1), VAR(-2)), Second query (relatedTo, VAR(-1), ObjA)
        [[[2, -1, -2], [1, -1, 10], [0, 0, 0], [0, 0, 0]]],
        # State 3: First query (hasProperty, VAR(-1), VAR(-1)), Second query (typeOf, VAR(-1), ObjC)
        [[[3, -1, -1], [2, -1, 12], [0, 0, 0], [0, 0, 0]]],
        # State 4: No query (padding)
        [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]]
    ], dtype=torch.long, device=device)

    facts_tensor = torch.tensor([
        [1, 11, 10], # F0: (relatedTo, ObjB, ObjA) -> Unifies S0, Sub: {-1: 11}
        [1, 12, 13], # F1: (relatedTo, ObjC, ObjD) -> Unifies S1, Sub: {}
        [1, 14, 10], # F2: (relatedTo, ObjE, ObjA) -> Unifies S0, Sub: {-1: 14}
        [2, 10, 11], # F3: (typeOf, ObjA, ObjB)   -> Unifies S2, Sub: {-1: 10, -2: 11}
        [2, 12, 12], # F4: (typeOf, ObjC, ObjC)   -> Unifies S2, Sub: {-1: 12, -2: 12}
        [1, 15, 10], # F5: (relatedTo, ObjF, ObjA) -> Unifies S0, Sub: {-1: 15}
        [3, 20, 20], # F6: (hasProperty, PropX, PropX) -> Unifies S3, Sub: {-1: 20}
        [3, 21, 22]  # F7: (hasProperty, PropY, PropZ) -> Conflict with S3 query, ignored
    ], dtype=torch.long, device=device)


    print(f"\n--- Running Arbitrary-K Version (K={K}) WITH Verbose ---")
    # Pass the mappings and set verbose=True
    subs_tensor_v, facts_idx_tensor_v, valid_mask_tensor_v = batch_unify_arbitrary_k(
        states, facts_tensor, vars_tensor, k=K, device=device,
        verbose=True,
        predicate_idx2str=predicate_map,
        constant_idx2str=constant_map
    )

    print("\n--- Applying Substitutions to Remaining Queries ---")
    next_states_tensor = apply_substitutions_to_rest(
        states,
        subs_tensor_v,
        valid_mask_tensor_v,
        PADDING_VALUE=PADDING_VALUE
    )

    print("\n--- Interpreted Results: Unifications and Next States ---")
    print("(Note: Order of results within K slots might vary between runs or devices)")
    bs, _, n_padding_atoms, _ = states.shape
    num_remaining_atoms = next_states_tensor.shape[2] # n_padding_atoms - 1

    for b in range(bs):
        query_atom = states[b, 0, 0, :]
        query_str = _format_atom(query_atom, predicate_map, constant_map, vars_idx_list)
        print(f"\nState {b}: Original First Query = {query_str}")
        if n_padding_atoms > 1:
             print(f"  Original Remaining Queries:")
             for atom_idx in range(1, n_padding_atoms):
                 rem_atom = states[b, 0, atom_idx, :]
                 # Only print non-padding atoms
                 if not torch.all(rem_atom == 0):
                     print(f"    {_format_atom(rem_atom, predicate_map, constant_map, vars_idx_list)}")
        else:
            print("  No remaining queries in original state.")


        found_unification = False
        for k_idx in range(K):
            if valid_mask_tensor_v[b, k_idx]: # Use the results from verbose run
                found_unification = True
                fact_idx = facts_idx_tensor_v[b, k_idx].item()
                fact_str = "[Fact not found]"
                # Check fact_idx is valid before accessing facts_tensor
                if fact_idx != PADDING_VALUE and fact_idx >= 0 and fact_idx < len(facts_tensor):
                     fact_str = _format_atom(facts_tensor[fact_idx], predicate_map, constant_map, vars_idx_list)
                else:
                     fact_str = f"[Invalid Fact Idx: {fact_idx}]"

                s_sub = subs_tensor_v[b, k_idx, 0, :]
                o_sub = subs_tensor_v[b, k_idx, 1, :]
                subs_list = []
                # Check against NEW PADDING_VALUE
                if s_sub[0].item() != PADDING_VALUE:
                    subs_list.append(_format_substitution(s_sub, constant_map))
                # Check against NEW PADDING_VALUE
                if o_sub[0].item() != PADDING_VALUE:
                    is_s_sub_valid = s_sub[0].item() != PADDING_VALUE
                    is_same_var = s_sub[0].item() == o_sub[0].item()
                    # Avoid duplicating substitution if S and O use the same variable
                    # and S substitution was already added
                    if not (is_s_sub_valid and is_same_var):
                         subs_list.append(_format_substitution(o_sub, constant_map))

                subs_str = ", ".join(subs_list) if subs_list else "{}"
                print(f"  - [Slot {k_idx}]: Unifies with Fact {fact_idx} {fact_str}, Subs: {subs_str}")

                # Print the corresponding next state (remaining queries after substitution)
                if num_remaining_atoms > 0:
                    print(f"    -> Next State Queries (Slot {k_idx}):")
                    has_next_query = False
                    for atom_idx in range(num_remaining_atoms):
                        next_atom = next_states_tensor[b, k_idx, atom_idx, :]
                        # Check it's not a padding atom or fully padded due to invalid slot
                        if next_atom[0].item() != PADDING_VALUE and not torch.all(next_atom == 0):
                            print(f"       {_format_atom(next_atom, predicate_map, constant_map, vars_idx_list)}")
                            has_next_query = True
                    if not has_next_query:
                         print("       [No remaining non-padding queries]")
                else:
                     print("    -> Next State Queries: [None (no remaining queries)]")


        is_state_valid = ~torch.all(query_atom == 0) # Check if not padding
        if not found_unification and is_state_valid:
             print("  No valid unifications found (or kept within K).")
        elif not is_state_valid:
             print("  No valid first query (padding atom).")

