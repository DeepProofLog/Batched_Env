
import torch
from typing import List, Tuple, Dict, Optional, Set

# Notes:
# - Arity is assumed to be 2 for atoms: shape [*, 3] = [predicate, arg1, arg2].
# - This module is a drop-in replacement for the previous unification_engine.py.
# - It integrates the fixes/optimizations discussed:
#   * Correct handling of substitutions coming back from fact unification
#   * Faster, vectorized canonicalization of variables
#   * Collision-resistant rolling hash for state deduplication
#   * Arity-2 conflict check without global sort/unique
#   * Eager pruning of ground facts from derived states
#   * Minor memory hygiene in substitutions application

# ---------------------------------------------------------------------------
# Rolling hash for tensors (order-aware, low-collision).
# ---------------------------------------------------------------------------

def _tensor_hash(tensor: torch.Tensor) -> int:
    """
    Order-aware rolling hash over the flattened contents of a 2D long tensor.
    Uses a large Mersenne prime as modulus for robustness.

    This function is CPU-friendly for small tensors (typical for LP states).
    If you batch thousands of states, consider vectorizing or porting to CUDA.
    """
    if tensor.numel() == 0:
        return 0
    # Ensure tensor is on CPU and long for stable hashing in Python.
    flat = tensor.contiguous().view(-1).to("cpu", non_blocking=True).long()
    mod = (1 << 61) - 1
    base = 131
    h = 0
    for x in flat:
        h = (h * base + int(x)) % mod
    return int(h)


# ---------------------------------------------------------------------------
# Substitution utilities
# ---------------------------------------------------------------------------

def apply_substitutions_simple(
    goals: torch.Tensor,
    substitutions: torch.Tensor,
    im: "IndexManager",
) -> torch.Tensor:
    """
    Apply a set of (from, to) substitutions to a single goals tensor.

    Args:
        goals: Tensor[G, A+1]
        substitutions: Tensor[S, 2] where each row is [from_idx, to_idx].
        im: IndexManager (for pad index)

    Returns:
        goals': Tensor with substitutions applied.
    """
    pad = im.padding_idx
    if goals.numel() == 0 or substitutions.numel() == 0:
        return goals

    valid_mask = substitutions[:, 0] != pad
    if not valid_mask.any():
        return goals

    subs = substitutions[valid_mask]
    # Keep mapping domain tight to reduce memory.
    max_idx = int(max(goals.max(), subs.max())) + 1
    device = goals.device
    mapping = torch.arange(max_idx, device=device)
    mapping[subs[:, 0].long()] = subs[:, 1].long()
    original_shape = goals.shape
    return mapping[goals.reshape(-1).long()].reshape(original_shape)


def apply_substitutions_batched(
    goals_batch: torch.Tensor,
    substitutions_batch: torch.Tensor,
    im: "IndexManager",
) -> torch.Tensor:
    """
    Batched substitutions: apply S pairs per batch item to its goal rows.

    Args:
        goals_batch: Tensor[B, G, A+1]
        substitutions_batch: Tensor[B, S, 2]
        im: IndexManager

    Returns:
        Tensor[B, G, A+1]
    """
    pad = im.padding_idx
    device = goals_batch.device
    B = goals_batch.shape[0]
    if B == 0:
        return goals_batch

    # Tight mapping table sized to the maximum id present across both tensors.
    max_idx = int(max(goals_batch.max(), substitutions_batch.max())) + 1
    mapping = torch.arange(max_idx, device=device).unsqueeze(0).expand(B, -1)

    # Filter padding substitutions and scatter into mapping per batch.
    valid = substitutions_batch[..., 0] != pad
    if valid.any():
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(substitutions_batch[..., 0])[valid]
        src = substitutions_batch[valid][:, 0].long()
        dst = substitutions_batch[valid][:, 1].long()
        mapping[batch_idx, src] = dst

    # Apply per-batch gather
    b_idx = torch.arange(B, device=device).view(B, 1, 1)
    return mapping[b_idx, goals_batch.long()]


# ---------------------------------------------------------------------------
# Variable canonicalization (vectorized)
# ---------------------------------------------------------------------------

def canonicalize_variables_in_state_idx(
    state_tensor: torch.Tensor, index_manager: "IndexManager", next_var_index: int
) -> Tuple[torch.Tensor, int]:
    """
    Renumber variables in a state so they become dense starting from next_var_index.
    This is vectorized using unique(..., return_inverse=True), avoiding Python loops.

    Returns:
        (canonical_state, new_next_var_index)
    """
    if state_tensor.numel() == 0:
        return state_tensor, next_var_index

    device = state_tensor.device
    var_start = index_manager.constant_no + 1

    flat = state_tensor.view(-1)
    is_var = flat >= var_start
    if not is_var.any():
        return state_tensor, next_var_index

    vars_only = flat[is_var]
    uniq, inv = torch.unique(vars_only, return_inverse=True)
    new_vals = torch.arange(next_var_index, next_var_index + uniq.numel(), device=device)
    flat[is_var] = new_vals[inv]
    return flat.view_as(state_tensor), next_var_index + uniq.numel()


# ---------------------------------------------------------------------------
# Core unification (arity 2, optimized conflict check)
# ---------------------------------------------------------------------------

def _unify_one_to_one_optimized(
    queries: torch.Tensor, terms: torch.Tensor, im: "IndexManager"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Unify each row in `queries` with the corresponding row in `terms`.
    Both are [B, 3]: [pred, arg1, arg2].
    Returns:
        mask: [B] bool - success per row
        subs: [B, 2, 2] long - for arity=2, pairs of [from, to] (pad where not bound)
    """
    device = queries.device
    pad = im.padding_idx
    B = queries.shape[0]
    if B == 0:
        return (torch.empty(0, dtype=torch.bool, device=device),
                torch.empty(0, 2, 2, dtype=torch.long, device=device))

    var_start = im.constant_no + 1

    pred_match = queries[:, 0] == terms[:, 0]

    q1, q2 = queries[:, 1], queries[:, 2]
    t1, t2 = terms[:, 1], terms[:, 2]

    q1v, q2v = q1 >= var_start, q2 >= var_start
    t1v, t2v = t1 >= var_start, t2 >= var_start

    # Constant mismatches kill unification immediately.
    const_mismatch = ((~q1v & ~t1v) & (q1 != t1)) | ((~q2v & ~t2v) & (q2 != t2))
    ok = pred_match & ~const_mismatch

    # Arity-2 special conflict checks (no global sort needed):
    # Case A: same query var used twice, both term args are different constants.
    same_qvar = (q1 == q2) & q1v & q2v
    t_both_const = (~t1v & ~t2v)
    conflict_A = same_qvar & t_both_const & (t1 != t2)

    # Case B: same term var used twice, both query args are different constants.
    same_tvar = (t1 == t2) & t1v & t2v
    q_both_const = (~q1v & ~q2v)
    conflict_B = same_tvar & q_both_const & (q1 != q2)

    ok = ok & ~conflict_A & ~conflict_B

    # Build substitutions only for successful rows.
    subs = torch.full((B, 2, 2), pad, dtype=torch.long, device=device)

    success_idx = ok.nonzero(as_tuple=True)[0]
    if success_idx.numel() == 0:
        return ok, subs

    # For success rows, bind var->const (or const->var) where appropriate.
    qq1, qq2 = q1[success_idx], q2[success_idx]
    tt1, tt2 = t1[success_idx], t2[success_idx]
    qq1v, qq2v = qq1 >= var_start, qq2 >= var_start
    tt1v, tt2v = tt1 >= var_start, tt2 >= var_start

    # Slot 0 bindings
    #  - q var to t const
    mask_q1_to_t1 = qq1v & (~(tt1v))
    #  - t var to q const
    mask_t1_to_q1 = (~(qq1v)) & tt1v

    # Slot 1 bindings
    mask_q2_to_t2 = qq2v & (~(tt2v))
    mask_t2_to_q2 = (~(qq2v)) & tt2v

    subs_succ = torch.full((success_idx.numel(), 2, 2), pad, dtype=torch.long, device=device)

    if mask_q1_to_t1.any():
        subs_succ[mask_q1_to_t1, 0] = torch.stack([qq1[mask_q1_to_t1], tt1[mask_q1_to_t1]], dim=-1)
    if mask_t1_to_q1.any():
        subs_succ[mask_t1_to_q1, 0] = torch.stack([tt1[mask_t1_to_q1], qq1[mask_t1_to_q1]], dim=-1)
    if mask_q2_to_t2.any():
        subs_succ[mask_q2_to_t2, 1] = torch.stack([qq2[mask_q2_to_t2], tt2[mask_q2_to_t2]], dim=-1)
    if mask_t2_to_q2.any():
        subs_succ[mask_t2_to_q2, 1] = torch.stack([tt2[mask_t2_to_q2], qq2[mask_t2_to_q2]], dim=-1)

    subs[success_idx] = subs_succ
    return ok, subs


# ---------------------------------------------------------------------------
# Rule and Fact unification entry points
# ---------------------------------------------------------------------------

def _unify_with_rules(
    query: torch.Tensor,
    remaining_goals: torch.Tensor,
    rules: torch.Tensor,
    rule_lengths: torch.Tensor,
    index_manager: "IndexManager",
) -> List[torch.Tensor]:
    """
    Unify query with rule heads; each success yields the rule body (instantiated)
    concatenated with the instantiated remainder of goals.
    """
    device = query.device
    pad = index_manager.padding_idx

    qpred = query[0, 0].item()
    rel = index_manager.rule_index.get(qpred, [])
    if not rel:
        return []

    rel_t = torch.tensor(rel, device=device, dtype=torch.long)
    rule_heads = rules[rel_t, 0, :]

    mask, subs = _unify_one_to_one_optimized(query.expand(rule_heads.shape[0], -1), rule_heads, index_manager)
    succ = mask.nonzero(as_tuple=True)[0]
    if succ.numel() == 0:
        return []

    next_states: List[torch.Tensor] = []
    for i in succ.tolist():
        ridx = rel_t[i].item()
        this_subs = subs[i]
        body = rules[ridx, 1:rule_lengths[ridx].item()]

        inst_body = apply_substitutions_simple(body, this_subs, index_manager)
        if remaining_goals.numel() > 0:
            inst_rest = apply_substitutions_simple(remaining_goals, this_subs, index_manager)
            next_states.append(torch.cat([inst_body, inst_rest], dim=0))
        else:
            next_states.append(inst_body)

    return next_states


def _unify_with_facts_fully_indexed(
    query: torch.Tensor,
    remaining_goals: torch.Tensor,
    index_manager: "IndexManager",
) -> List[torch.Tensor]:
    """
    Use composite key index for precise candidate facts:
      key = (predicate, (pos, const) ...)

    Returns:
        - [] on failure
        - [index_manager.true_tensor] if success AND there were no remaining_goals
        - list of instantiated remaining_goals for each matching fact otherwise
    """
    pad = index_manager.padding_idx
    device = query.device

    qpred = query[0, 0].item()
    qargs = query[0, 1:]

    # Build key of constant arguments with positions.
    constant_args_with_pos = []
    for pos in range(index_manager.max_arity):
        val = int(qargs[pos].item())
        if not index_manager.is_var_idx(val) and val != pad:
            constant_args_with_pos.append((pos, val))

    lookup_key = (qpred,) + tuple(constant_args_with_pos)
    candidate_ids = index_manager.fact_index.get(lookup_key, None)
    if candidate_ids is None or candidate_ids.numel() == 0:
        return []

    facts = index_manager.facts_tensor[candidate_ids]
    mask, subs = _unify_one_to_one_optimized(query.expand(facts.shape[0], -1), facts, index_manager)
    succ = mask.nonzero(as_tuple=True)[0]
    if succ.numel() == 0:
        return []

    if remaining_goals.numel() == 0:
        # A fact matched; with no remaining goals we have a proof.
        return [index_manager.true_tensor]

    succ_subs = subs[succ]
    # Broadcast and apply substitutions to the remainder.
    expanded = remaining_goals.unsqueeze(0).expand(succ_subs.shape[0], -1, -1)
    instantiated = apply_substitutions_batched(expanded, succ_subs, index_manager)
    return list(torch.unbind(instantiated, dim=0))


# ---------------------------------------------------------------------------
# Pruning helpers
# ---------------------------------------------------------------------------

def _prune_eager_true_ground_atoms(
    state: torch.Tensor, facts_tensor: torch.Tensor, im: "IndexManager"
) -> Optional[torch.Tensor]:
    """
    Remove atoms that appear verbatim in the fact base.
    If everything is removed, return None to signal a proven state.
    """
    pad = im.padding_idx
    if state.numel() == 0:
        return None

    # Filter padding rows first.
    valid = state[state[:, 0] != pad]
    if valid.numel() == 0:
        return None

    # Broadcast-equality to detect facts.
    is_fact = (valid.unsqueeze(1) == facts_tensor.unsqueeze(0)).all(dim=2).any(dim=1)
    kept = valid[~is_fact]
    if kept.numel() == 0:
        return None
    return kept


# ---------------------------------------------------------------------------
# Public stepping APIs
# ---------------------------------------------------------------------------

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


def get_next_unification_pt_with_correct_logic(
    current_state: torch.Tensor,
    facts_tensor: torch.Tensor,
    rules: torch.Tensor,
    rule_lengths: torch.Tensor,
    index_manager: "IndexManager",
    next_var_index: int,
    verbose: int = 0,
) -> Tuple[List[torch.Tensor], int]:
    """
    Variant that mirrors the original "rules-first then facts on the new first goal" logic,
    with the **fix** that we do NOT re-apply substitutions returned by fact unification.
    Also prunes ground facts eagerly.
    """
    pad = index_manager.padding_idx

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

    # Step 1: unify with rules to get intermediate states
    inter = _unify_with_rules(query, remaining, rules, rule_lengths, index_manager)
    if not inter:
        return [index_manager.false_tensor], next_var_index

    after_facts: List[torch.Tensor] = []

    # Step 2: for each intermediate, unify its new first goal with facts
    for s in inter:
        s_valid = s[s[:, 0] != pad]
        if s_valid.numel() == 0:
            return [index_manager.true_tensor], next_var_index

        first, rest = s_valid[0:1], s_valid[1:]

        fact_states = _unify_with_facts_fully_indexed(first, rest, index_manager)

        # If unification with facts succeeded and there were no remaining goals,
        # we have a proof.
        if rest.numel() == 0 and fact_states:
            if not torch.equal(fact_states[0], index_manager.false_tensor):
                return [index_manager.true_tensor], next_var_index

        # Process results of fact unification:
        for new_state in fact_states:
            # FIX: `new_state` is already INSTantiated remainder (or true_tensor).
            if torch.equal(new_state, index_manager.true_tensor):
                candidate = rest  # everything proven, keep rest as proof (empty OK)
            else:
                candidate = new_state

            # Eager prune ground facts
            pruned = _prune_eager_true_ground_atoms(candidate, facts_tensor, index_manager)
            if pruned is None:
                return [index_manager.true_tensor], next_var_index
            after_facts.append(pruned)

    # Step 3: Canonicalize + dedup + pad
    if not after_facts:
        return [index_manager.false_tensor], next_var_index

    final_states: List[torch.Tensor] = []
    seen: Set[int] = set()
    for s in after_facts:
        if s.numel() == 0:
            return [index_manager.true_tensor], next_var_index
        canon, next_var_index = canonicalize_variables_in_state_idx(s, index_manager, next_var_index)
        h = _tensor_hash(canon)
        if h not in seen:
            final_states.append(canon)
            seen.add(h)

    if not final_states:
        return [index_manager.false_tensor], next_var_index

    original_rows = current_state.shape[0]
    padded: List[torch.Tensor] = []
    for s in final_states:
        r, c = s.shape
        if r < original_rows:
            pad_rows = torch.full((original_rows - r, c), pad, dtype=s.dtype, device=s.device)
            padded.append(torch.cat([s, pad_rows], dim=0))
        elif r > original_rows:
            padded.append(s[:original_rows])
        else:
            padded.append(s)

    return padded, next_var_index
