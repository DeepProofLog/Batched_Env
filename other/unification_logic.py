
import torch
from typing import List, Tuple, Dict, Optional, Set


def get_next_unification_pt(
    current_state: torch.Tensor,
    facts_tensor: torch.Tensor,
    rules: torch.Tensor,
    rule_lengths: torch.Tensor,
    index_manager: "IndexManager",
    next_var_index: int,
    verbose: int = 0,
) -> Tuple[List[torch.Tensor], int]:
    """
    One-step prover: unify the leftmost atom with either rules or facts
    (both are explored), prune trivial ground facts, canonicalize variables,
    deduplicate states, and pad back to the original row count.

    Returns:
        (list_of_next_states_padded, updated_next_var_index)
        If any branch yields an empty goal set, [True] is returned immediately.
    """
    pad = index_manager.padding_idx
    device = current_state.device

    # Validate/compact current_state
    valid_mask = current_state[:, 0] != pad
    if not valid_mask.any():
        return [index_manager.true_tensor], next_var_index

    state = current_state[valid_mask]
    if (state[:, 0] == index_manager.false_pred_idx).any():
        return [index_manager.false_tensor], next_var_index
    if state.shape[0] == 0:
        return [index_manager.true_tensor], next_var_index

    query, remaining = state[0:1], state[1:]

    # 1) Rules
    rule_states = _unify_with_rules(query, remaining, rules, rule_lengths, index_manager)

    # 2) Facts
    fact_states = _unify_with_facts_fully_indexed(query, remaining, index_manager)
    if fact_states and torch.equal(fact_states[0], index_manager.true_tensor):
        return [index_manager.true_tensor], next_var_index

    # Combine
    candidates = rule_states + fact_states
    if not candidates:
        return [index_manager.false_tensor], next_var_index

    # 3) Prune ground facts; detect immediate proof.
    pruned: List[torch.Tensor] = []
    for s in candidates:
        pruned_s = _prune_eager_true_ground_atoms(s, facts_tensor, index_manager)
        if pruned_s is None:
            return [index_manager.true_tensor], next_var_index
        pruned.append(pruned_s)

    # 4) Canonicalize + dedup
    final_states: List[torch.Tensor] = []
    seen: Set[int] = set()
    for s in pruned:
        if s.numel() == 0:
            return [index_manager.true_tensor], next_var_index
        canon, next_var_index = canonicalize_variables_in_state_idx(s, index_manager, next_var_index)
        h = _tensor_hash(canon)
        if h not in seen:
            final_states.append(canon)
            seen.add(h)

    if not final_states:
        return [index_manager.false_tensor], next_var_index

    # 5) Pad back to original number of rows (for drop-in compatibility).
    original_rows = current_state.shape[0]
    padded: List[torch.Tensor] = []
    for s in final_states:
        r, c = s.shape
        if r < original_rows:
            pad_rows = torch.full(
                (original_rows - r, c), pad, dtype=s.dtype, device=s.device
            )
            padded.append(torch.cat([s, pad_rows], dim=0))
        elif r > original_rows:
            # Keep behavior-compatible with previous engine: truncate if longer.
            padded.append(s[:original_rows])
        else:
            padded.append(s)

    return padded, next_var_index
