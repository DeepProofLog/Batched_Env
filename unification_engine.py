"""
Unification Engine - GPU-friendly, batched, arity-2, indices-only

This module implements a vectorised, single-step unification engine compatible
with the provided DataHandler / IndexManager / BatchedVecEnv stack.

Design choices (aligned with user's constraints):
- Arity-2 atoms only, shape [k, 3] for a state (predicate, arg1, arg2)
- Only constants and variables (no function symbols)
- Rule atoms use only variables (no constants in rule heads/bodies)
- Occurs-check omitted
- Single step returns *all* successors (no heuristic ordering)
- Eager fact closure: drop any ground goal that is already a fact
- Index-only: constants in [1..n], variables >= n+1, padding=0

Compatibility notes:
- Works directly with IndexManager tensors:
  * facts_idx            : [F, 3] (sorted by predicate)
  * predicate_range_map  : [num_pred+1, 2] ranges on CPU
  * rules_idx            : [R, M, 3] bodies (padded)
  * rules_heads_idx      : [R, 3] heads
  * rule_lens            : [R]
- Provides UnificationEngine.from_index_manager(im)
- API used by BatchedVecEnv:
  * get_derived_states(current_states, next_var_indices, excluded_queries, labels)
  * is_true_state / is_false_state / get_true_state / get_false_state

Implementation highlights:
- Fast batched unification via sort-and-compare conflict detection
- GPU-parallel hashing for deduplication (no CPU roundtrips)
- Vectorised fact candidate retrieval using predicate ranges and broadcasting
- Optional exclusion of the original query (positive label) during fact unification
- Eager fact closure on derived states
"""
from __future__ import annotations
from typing import List, Tuple, Optional

import torch
from torch import Tensor

# ------------------------------------------------------------
# Utilities: hashing & deduplication (GPU-friendly)
# ------------------------------------------------------------

def _gpu_parallel_hash(states: Tensor, padding_idx: int) -> Tensor:
    """Compute a simple polynomial rolling hash for a batch of padded states.
    Args:
        states: [B, max_atoms, 3]
        padding_idx: index used for padding
    Returns:
        hashes: [B]
    """
    if states.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=states.device)

    B, max_atoms, arity_p1 = states.shape
    device = states.device

    valid_mask = states[:, :, 0] != padding_idx  # [B, max_atoms]

    prime = 31
    mod_val = 2**31 - 1

    flat_states = states.reshape(B, -1).long()

    max_len = max_atoms * arity_p1
    powers = torch.arange(max_len, device=device, dtype=torch.long)
    prime_powers = torch.pow(torch.tensor(prime, device=device, dtype=torch.long), powers) % mod_val

    hashes = (flat_states * prime_powers.unsqueeze(0)).sum(dim=1) % mod_val

    has_atoms = valid_mask.any(dim=1)
    hashes = hashes * has_atoms.long()
    return hashes


def _gpu_batch_unique(states: Tensor, hashes: Tensor) -> Tensor:
    """Deduplicate a batch of states using precomputed hashes. Returns unique rows.
    Args:
        states: [B, max_atoms, 3]
        hashes: [B]
    Returns:
        unique_states: [U, max_atoms, 3]
    """
    if states.numel() == 0:
        return states[:0]

    device = states.device
    B = states.shape[0]

    sorted_hashes, sort_indices = torch.sort(hashes)
    unique_mask = torch.ones(B, dtype=torch.bool, device=device)
    if B > 1:
        unique_mask[1:] = sorted_hashes[1:] != sorted_hashes[:-1]
    unique_indices = sort_indices[unique_mask]
    return states[unique_indices]


def _deduplicate_states_gpu(states: List[Tensor], padding_idx: int) -> List[Tensor]:
    """Stack, hash, unique, then unpad back into a Python list."""
    if not states:
        return []
    if len(states) <= 2:
        return states

    device = states[0].device
    max_atoms = max(s.shape[0] for s in states)

    padded = []
    for s in states:
        if s.shape[0] < max_atoms:
            pad = torch.full((max_atoms - s.shape[0], s.shape[1]), padding_idx, dtype=s.dtype, device=device)
            s = torch.cat([s, pad], dim=0)
        else:
            s = s[:max_atoms]
        padded.append(s)

    batch = torch.stack(padded, dim=0)
    hashes = _gpu_parallel_hash(batch, padding_idx)
    unique_batch = _gpu_batch_unique(batch, hashes)

    out: List[Tensor] = []
    for i in range(unique_batch.shape[0]):
        st = unique_batch[i]
        mask = st[:, 0] != padding_idx
        if mask.any():
            out.append(st[mask])
    return out

# ------------------------------------------------------------
# Core unification helpers
# ------------------------------------------------------------

@torch.no_grad()
def _apply_substitutions_simple(goals: Tensor, substitutions: Tensor, padding_idx: int) -> Tensor:
    """Apply substitutions [S,2] to goals [G,3] in a single shot.
    substitutions may contain padding rows with padding_idx in the first column.
    Substitutions only apply to argument positions (columns 1,2), not predicates (column 0).
    """
    if goals.numel() == 0 or substitutions.numel() == 0:
        return goals

    valid_mask = substitutions[:, 0] != padding_idx
    if not valid_mask.any():
        return goals

    subs = substitutions[valid_mask]
    # Only apply substitutions to arguments, not predicates
    args = goals[:, 1:]  # [G, 2]
    preds = goals[:, 0:1]  # [G, 1]
    
    flat = args.flatten()
    max_idx = int(torch.max(torch.cat([flat.long(), subs.flatten().long()])).item()) + 1
    mapping = torch.arange(max_idx, device=goals.device, dtype=torch.long)
    mapping[subs[:, 0].long()] = subs[:, 1].long()
    
    # Apply substitution only to arguments
    subst_args = mapping[flat.long()].reshape(args.shape)
    
    # Concatenate predicates back with substituted arguments
    return torch.cat([preds, subst_args], dim=1)


@torch.no_grad()
def _apply_substitutions_batched(goals_batch: Tensor, substitutions_batch: Tensor, padding_idx: int) -> Tensor:
    """Batched variant.
    goals_batch: [B, G, 3]
    substitutions_batch: [B, S, 2]
    Substitutions only apply to argument positions (columns 1,2), not predicates (column 0).
    """
    B, G, A = goals_batch.shape
    if B == 0:
        return goals_batch

    # Separate predicates from arguments
    preds = goals_batch[:, :, 0:1]  # [B, G, 1]
    args = goals_batch[:, :, 1:]  # [B, G, 2]

    max_idx = int(torch.max(torch.cat([args.flatten(), substitutions_batch.flatten()])).item()) + 1
    mapping = torch.arange(max_idx, device=goals_batch.device).unsqueeze(0).expand(B, -1).clone()

    pad = padding_idx
    valid_mask = substitutions_batch[..., 0] != pad
    batch_idx = torch.arange(B, device=goals_batch.device).unsqueeze(1).expand_as(substitutions_batch[..., 0])[valid_mask]
    from_vars = substitutions_batch[valid_mask][:, 0].long()
    to_vals = substitutions_batch[valid_mask][:, 1].long()
    mapping[batch_idx, from_vars] = to_vals

    batch_indices = torch.arange(B, device=goals_batch.device).unsqueeze(1).unsqueeze(1)
    subst_args = mapping[batch_indices, args.long()]
    
    # Concatenate predicates back with substituted arguments
    return torch.cat([preds, subst_args], dim=2)


@torch.no_grad()
def _unify_one_to_one_optimized(queries: Tensor, terms: Tensor, constant_no: int, padding_idx: int) -> Tuple[Tensor, Tensor]:
    """Unify N query atoms with N term atoms (pairwise rows).
    Returns (mask: [N], substitutions: [N, 2, 2]) for arity=2 (we store 2 pairs per row, padded with padding_idx).
    """
    device = queries.device
    N = queries.shape[0]
    if N == 0:
        pad_subs = torch.full((0, 2, 2), padding_idx, dtype=torch.long, device=device)
        return torch.empty(0, dtype=torch.bool, device=device), pad_subs

    var_start = constant_no + 1

    pred_match = queries[:, 0] == terms[:, 0]
    q_args, t_args = queries[:, 1:], terms[:, 1:]

    q_var = q_args >= var_start
    t_var = t_args >= var_start

    const_mismatch = (~q_var & ~t_var) & (q_args != t_args)
    initial_unifiable = pred_match & ~const_mismatch.any(dim=1)

    mask = initial_unifiable

    # Build substitutions only where mask
    subs = torch.full((N, 2, 2), padding_idx, dtype=torch.long, device=device)
    idx = mask.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return mask, subs

    q = q_args[idx]
    t = t_args[idx]

    qv = q >= var_start
    tv = t >= var_start

    # var <- const
    qv_tc = qv & ~tv
    if qv_tc.any():
        qv_vars = q[qv_tc].to(dtype=torch.long)
        qv_vals = t[qv_tc].to(dtype=torch.long)
        # Each row has at most 2 substitutions; pack into 2 slots deterministically by arg order
        rows = qv_tc.nonzero(as_tuple=True)[0]
        cols = qv_tc.nonzero(as_tuple=True)[1]
        # slot is 0 for arg1, 1 for arg2
        slots = cols
        subs_rows = idx[rows]
        subs[subs_rows, slots, 0] = qv_vars
        subs[subs_rows, slots, 1] = qv_vals

    # const <- var  (we rewrite it as var <- const for symmetry by swapping)
    tv_qc = ~qv & tv
    if tv_qc.any():
        tv_vars = t[tv_qc].to(dtype=torch.long)
        tv_vals = q[tv_qc].to(dtype=torch.long)
        rows = tv_qc.nonzero(as_tuple=True)[0]
        cols = tv_qc.nonzero(as_tuple=True)[1]
        slots = cols
        subs_rows = idx[rows]
        subs[subs_rows, slots, 0] = tv_vars
        subs[subs_rows, slots, 1] = tv_vals

    # When both are vars, no immediate binding needed (we skip var<->var since rule vars are local)
    return mask, subs

# ------------------------------------------------------------
# Engine
# ------------------------------------------------------------

class UnificationEngine:
    """Encapsulates facts/rules and batched single-step unification."""

    def __init__(
        self,
        facts_idx: Tensor,
        rules_idx: Tensor,
        rule_lens: Tensor,
        padding_idx: int,
        constant_no: int,
        runtime_var_end_index: int,
        true_pred_idx: Optional[int],
        false_pred_idx: Optional[int],
        max_arity: int,
        predicate_range_map: Optional[Tensor] = None,
        rules_heads_idx: Optional[Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device if device is not None else torch.device('cpu')

        # Core tensors (on device where they will be used)
        self.facts_idx = facts_idx.to(self.device, dtype=torch.int32) if facts_idx is not None else torch.empty((0, 3), dtype=torch.int32, device=self.device)
        self.rules_idx = rules_idx.to(self.device, dtype=torch.int32) if rules_idx is not None else torch.empty((0, 0, 3), dtype=torch.int32, device=self.device)
        self.rule_lens = rule_lens.to(self.device, dtype=torch.int32) if rule_lens is not None else torch.empty((0,), dtype=torch.int32, device=self.device)
        self.rules_heads_idx = rules_heads_idx.to(self.device, dtype=torch.int32) if rules_heads_idx is not None else torch.empty((0, 3), dtype=torch.int32, device=self.device)

        # Indices & ranges
        self.padding_idx = int(padding_idx)
        self.constant_no = int(constant_no)
        self.runtime_var_end_index = int(runtime_var_end_index)
        self.max_arity = int(max_arity)

        self.true_pred_idx = int(true_pred_idx) if true_pred_idx is not None else None
        self.false_pred_idx = int(false_pred_idx) if false_pred_idx is not None else None

        # True/False tensors
        if self.true_pred_idx is not None:
            self.true_tensor = torch.tensor([[self.true_pred_idx, self.padding_idx, self.padding_idx]], dtype=torch.int32, device=self.device)
        else:
            self.true_tensor = None
        if self.false_pred_idx is not None:
            self.false_tensor = torch.tensor([[self.false_pred_idx, self.padding_idx, self.padding_idx]], dtype=torch.int32, device=self.device)
        else:
            self.false_tensor = None

        # Predicate range map is CPU in IndexManager; keep a copy on CPU
        # shape [num_pred+1, 2], rows are (start, end) in facts_idx
        self.predicate_range_map = predicate_range_map if predicate_range_map is not None else torch.zeros((1, 2), dtype=torch.int32)

        # Build predicate -> rule indices (CPU list of LongTensor indices)
        self._pred_to_rule_idx: List[Tensor] = []
        self._build_pred_to_rule_indices()

    # -------------------------
    # Construction helpers
    # -------------------------
    @classmethod
    def from_index_manager(cls, index_manager, take_ownership: bool = False) -> "UnificationEngine":
        rules_heads = getattr(index_manager, 'rules_heads_idx', None)
        engine = cls(
            facts_idx=index_manager.facts_idx,
            rules_idx=index_manager.rules_idx,
            rule_lens=index_manager.rule_lens,
            padding_idx=index_manager.padding_idx,
            constant_no=index_manager.constant_no,
            runtime_var_end_index=index_manager.runtime_var_end_index,
            true_pred_idx=index_manager.true_pred_idx,
            false_pred_idx=index_manager.false_pred_idx,
            max_arity=index_manager.max_arity,
            predicate_range_map=getattr(index_manager, 'predicate_range_map', None),
            rules_heads_idx=rules_heads,
            device=index_manager.device,
        )
        # Optionally allow the caller to clear large tensors on the IndexManager
        if take_ownership:
            index_manager.facts_idx = None
            index_manager.rules_idx = None
            index_manager.rule_lens = None
            index_manager.rules_heads_idx = None
        return engine

    def _build_pred_to_rule_indices(self) -> None:
        """Precompute rule index lists per predicate from rules_heads_idx[:,0]."""
        if self.rules_heads_idx is None or self.rules_heads_idx.numel() == 0:
            self._pred_to_rule_idx = []
            return
        pcol = self.rules_heads_idx[:, 0].to('cpu')  # [R]
        if pcol.numel() == 0:
            self._pred_to_rule_idx = []
            return
        max_pred = int(pcol.max().item())
        buckets: List[List[int]] = [[] for _ in range(max_pred + 1)]
        for r_idx, p in enumerate(pcol.tolist()):
            buckets[p].append(r_idx)
        self._pred_to_rule_idx = [torch.tensor(b, dtype=torch.long) if b else torch.empty((0,), dtype=torch.long) for b in buckets]

    # -------------------------
    # Public API used by env
    # -------------------------
    def get_derived_states(
        self,
        current_states: Tensor,
        next_var_indices: Tensor,
        excluded_queries: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        verbose: int = 0,
    ) -> Tuple[List[List[Tensor]], Tensor]:
        """Vectorised single-step expansion for a batch of states.
        Returns (list of derived state lists, updated next_var_indices [unchanged]).
        """
        B, padding_atoms, _ = current_states.shape
        pad = self.padding_idx
        device = current_states.device

        # --- Extract leftmost queries and remaining goals
        valid_mask = current_states[:, :, 0] != pad
        if not valid_mask.any():
            # All states are empty/terminal - return empty derived states for all
            all_derived: List[List[Tensor]] = [[] for _ in range(B)]
            return all_derived, next_var_indices

        first_idx = valid_mask.long().argmax(dim=1)
        batch_idx = torch.arange(B, device=device)
        queries = current_states[batch_idx, first_idx]  # [B, 3]
        
        # Handle environments where there are no valid atoms (all padding)
        # These get set to False
        empty_states = ~valid_mask.any(dim=1)  # [B] - True if all atoms are padding

        # Terminal flags (True/False predicates)
        if self.false_pred_idx is not None:
            false_states = (current_states[:, :, 0] == self.false_pred_idx).any(dim=1)
        else:
            false_states = torch.zeros(B, dtype=torch.bool, device=device)
        if self.true_pred_idx is not None:
            true_queries = queries[:, 0] == self.true_pred_idx
        else:
            true_queries = torch.zeros(B, dtype=torch.bool, device=device)
        terminal = false_states | true_queries

        # Remaining goals (excluding the leftmost)
        remaining_mask = valid_mask.clone()
        remaining_mask[batch_idx, first_idx] = False
        remaining_counts = remaining_mask.sum(dim=1)
        max_remaining = int(remaining_counts.max().item()) if remaining_counts.numel() > 0 else 0
        if max_remaining > 0:
            remaining_goals = torch.full((B, max_remaining, 3), pad, dtype=torch.long, device=device)
            for b in range(B):
                if remaining_counts[b] > 0:
                    remaining_goals[b, :remaining_counts[b]] = current_states[b, remaining_mask[b]]
        else:
            remaining_goals = torch.full((B, 1, 3), pad, dtype=torch.long, device=device)

        all_derived: List[List[Tensor]] = [[] for _ in range(B)]
        updated_next = next_var_indices.clone()

        # Handle empty states (all padding) - these should return False
        for b in range(B):
            if empty_states[b]:
                all_derived[b] = [self.false_tensor] if self.false_tensor is not None else []

        # Handle terminal
        for b in range(B):
            if terminal[b]:
                if true_queries[b] and self.true_tensor is not None:
                    all_derived[b] = [self.true_tensor]
                elif false_states[b] and self.false_tensor is not None:
                    all_derived[b] = [self.false_tensor]
                else:
                    all_derived[b] = []

        # Active (not terminal and not empty)
        active = (~terminal & ~empty_states).nonzero(as_tuple=True)[0]
        if active.numel() == 0:
            return all_derived, updated_next

        a_queries = queries[active]
        a_remaining = remaining_goals[active]
        a_counts = remaining_counts[active]

        preds = a_queries[:, 0]
        unique_preds = torch.unique(preds)

        for p in unique_preds:
            pmask = preds == p
            loc_idx = pmask.nonzero(as_tuple=True)[0]
            glob_idx = active[loc_idx]

            q = a_queries[loc_idx]                 # [N, 3]
            rem = a_remaining[loc_idx]            # [N, max_rem, 3]
            rem_counts = a_counts[loc_idx]        # [N]

            # --- Rule unification
            rule_res = self._unify_with_rules_batched(q, rem, rem_counts, int(p.item()))

            # --- Fact unification
            exq = excluded_queries[glob_idx] if excluded_queries is not None else None
            lbs = labels[glob_idx] if labels is not None else None
            fact_res = self._unify_with_facts_batched(q, rem, rem_counts, int(p.item()), excluded_queries=exq, labels=lbs)

            # --- Combine, prune eager facts, deduplicate, cap length
            for i, bidx in enumerate(glob_idx.tolist()):
                states_i = (rule_res[i] if i < len(rule_res) else []) + (fact_res[i] if i < len(fact_res) else [])
                if not states_i:
                    all_derived[bidx] = [self.false_tensor] if self.false_tensor is not None else []
                    continue

                # Eager fact closure on each derived state
                pruned: List[Tensor] = []
                for st in states_i:
                    pst = self._prune_eager_true_ground_atoms(st)
                    if pst is None:
                        # All goals proven true -> return [True]
                        pruned.append(self.true_tensor)
                    else:
                        pruned.append(pst)

                # Deduplicate
                uniq = _deduplicate_states_gpu(pruned, self.padding_idx)

                # Truncate extremely large branching (safety)
                if len(uniq) > 100:
                    uniq = uniq[:100]

                all_derived[bidx] = uniq if uniq else ([self.false_tensor] if self.false_tensor is not None else [])

        return all_derived, updated_next

    # -------------------------
    # Rule unification
    # -------------------------
    @torch.no_grad()
    def _unify_with_rules_batched(
        self,
        queries: Tensor,                  # [N,3]
        remaining_goals: Tensor,          # [N,max_rem,3]
        remaining_counts: Tensor,         # [N]
        pred_idx: int,
    ) -> List[List[Tensor]]:
        device = self.device
        pad = self.padding_idx

        # Fetch rule indices for this predicate
        rule_indices = self._pred_to_rule_idx[pred_idx] if pred_idx < len(self._pred_to_rule_idx) else torch.empty((0,), dtype=torch.long, device=device)
        if rule_indices.numel() == 0:
            return [[] for _ in range(queries.shape[0])]
        
        # Move rule_indices to device if needed
        if rule_indices.device != device:
            rule_indices = rule_indices.to(device)

        heads = self.rules_heads_idx.index_select(0, rule_indices)  # [R,3]
        R = heads.shape[0]
        N = queries.shape[0]

        # Pairwise unify all N x R combinations
        q_exp = queries.unsqueeze(1).expand(-1, R, -1).reshape(-1, 3)
        h_exp = heads.unsqueeze(0).expand(N, -1, -1).reshape(-1, 3)

        mask, subs = _unify_one_to_one_optimized(q_exp, h_exp, self.constant_no, pad)  # [N*R], [N*R,2,2]
        mask = mask.reshape(N, R)
        subs = subs.reshape(N, R, 2, 2)

        # For each query, build next states for successful rules
        results: List[List[Tensor]] = []
        rule_lens_cpu = self.rule_lens.index_select(0, rule_indices).to('cpu').numpy()  # body lens aligned to local rules

        for qi in range(N):
            succ = mask[qi].nonzero(as_tuple=True)[0]
            if succ.numel() == 0:
                results.append([])
                continue

            out_i: List[Tensor] = []
            rem_i = remaining_goals[qi, :remaining_counts[qi]] if remaining_counts[qi] > 0 else None

            for j_local in succ.tolist():
                r_global = int(rule_indices[j_local].item())
                body_len = int(rule_lens_cpu[j_local])
                if body_len == 0:
                    # Fact-like rule: head :- .
                    if rem_i is None or rem_i.numel() == 0:
                        out_i.append(self.true_tensor)
                    else:
                        out_i.append(rem_i)
                    continue

                body = self.rules_idx[r_global, :body_len]  # [k,3]
                s = subs[qi, j_local].reshape(-1, 2)
                body_inst = _apply_substitutions_simple(body, s, pad)

                if rem_i is not None and rem_i.numel() > 0:
                    rem_inst = _apply_substitutions_simple(rem_i, s, pad)
                    next_state = torch.cat([body_inst, rem_inst], dim=0)
                else:
                    next_state = body_inst
                out_i.append(next_state)

            results.append(out_i)
        return results

    # -------------------------
    # Fact unification (vectorised candidate retrieval)
    # -------------------------
    @torch.no_grad()
    def _unify_with_facts_batched(
        self,
        queries: Tensor,                  # [N,3]
        remaining_goals: Tensor,          # [N,max_rem,3]
        remaining_counts: Tensor,         # [N]
        pred_idx: int,
        excluded_queries: Optional[Tensor] = None,  # [N,3]
        labels: Optional[Tensor] = None,           # [N]
    ) -> List[List[Tensor]]:
        device = self.device
        pad = self.padding_idx
        N = queries.shape[0]

        # Slice facts for this predicate using predicate_range_map (CPU ints OK)
        start, end = 0, 0
        if self.predicate_range_map is not None and pred_idx < self.predicate_range_map.shape[0]:
            start = int(self.predicate_range_map[pred_idx, 0].item())
            end = int(self.predicate_range_map[pred_idx, 1].item())
        if end <= start:
            return [[] for _ in range(N)]

        facts_p = self.facts_idx[start:end]  # [F,3]
        F = facts_p.shape[0]

        # Build [N,F] candidate mask by matching constants in query positions
        q_args = queries[:, 1:]  # [N,2]
        f_args = facts_p[:, 1:]  # [F,2]

        const_mask = (q_args <= self.constant_no) & (q_args != pad)  # [N,2]
        # Broadcast compare: [N,1,2] vs [1,F,2]
        eq = (q_args.unsqueeze(1) == f_args.unsqueeze(0))  # [N,F,2]
        ok = (~const_mask).unsqueeze(1) | eq               # [N,F,2]
        cand_mask = ok.all(dim=2)                          # [N,F]

        # Exclude original query (for positive labels)
        if excluded_queries is not None and labels is not None:
            # Match equality with full atom
            eq_full = (facts_p.unsqueeze(0) == excluded_queries.unsqueeze(1)).all(dim=2)  # [N,F]
            pos = labels.view(-1, 1) == 1
            cand_mask = cand_mask & (~(pos & eq_full))

        # If no candidates for all, return
        if not cand_mask.any():
            return [[] for _ in range(N)]

        # Expand successful pairs into a flat list for unification
        q_idx, f_idx = cand_mask.nonzero(as_tuple=True)
        q_flat = queries.index_select(0, q_idx)
        f_flat = facts_p.index_select(0, f_idx)

        mask, subs = _unify_one_to_one_optimized(q_flat, f_flat, self.constant_no, pad)
        # Subset to successes (should be all True, but keep robust)
        succ_idx = mask.nonzero(as_tuple=True)[0]
        if succ_idx.numel() == 0:
            return [[] for _ in range(N)]

        q_succ = q_idx.index_select(0, succ_idx)
        subs_succ = subs.index_select(0, succ_idx)  # [S,2,2]

        # Group by original query
        results: List[List[Tensor]] = [[] for _ in range(N)]
        for qi in range(N):
            sel = (q_succ == qi).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue

            # If no remaining goals, success -> [True]
            if remaining_counts[qi] == 0:
                results[qi] = [self.true_tensor]
                continue

            rem_i = remaining_goals[qi:qi+1, :remaining_counts[qi]]   # [1,G,3]
            subs_i = subs_succ.index_select(0, sel)                   # [K,2,2]

            # Expand & apply substitutions in batch
            K = subs_i.shape[0]
            rem_exp = rem_i.expand(K, -1, -1).clone()                 # [K,G,3]
            inst = _apply_substitutions_batched(rem_exp, subs_i, pad) # [K,G,3]
            results[qi] = [inst[k] for k in range(K)]
        return results

    # -------------------------
    # Eager fact closure
    # -------------------------
    @torch.no_grad()
    def _prune_eager_true_ground_atoms(self, state: Tensor) -> Optional[Tensor]:
        """Remove any ground atoms that are already in the fact base.
        Returns:
            - None if *all* atoms were pruned (i.e., state proved True)
            - Tensor with remaining atoms otherwise
        """
        pad = self.padding_idx
        if state.numel() == 0:
            return None

        mask_valid = state[:, 0] != pad
        atoms = state[mask_valid]
        if atoms.numel() == 0:
            return None

        keep_mask = torch.ones(atoms.shape[0], dtype=torch.bool, device=atoms.device)

        for i in range(atoms.shape[0]):
            p, a, b = atoms[i]
            p = int(p.item())
            a = int(a.item())
            b = int(b.item())

            # Only ground atoms (both constants)
            if 1 <= a <= self.constant_no and 1 <= b <= self.constant_no:
                # Check membership in facts via predicate slice
                start = int(self.predicate_range_map[p, 0].item()) if self.predicate_range_map is not None and p < self.predicate_range_map.shape[0] else 0
                end = int(self.predicate_range_map[p, 1].item()) if self.predicate_range_map is not None and p < self.predicate_range_map.shape[0] else 0
                if end > start:
                    facts_p = self.facts_idx[start:end]
                    # Compare second & third columns
                    hit = ((facts_p[:, 1] == a) & (facts_p[:, 2] == b)).any()
                    if hit:
                        keep_mask[i] = False

        remaining = atoms[keep_mask]
        if remaining.numel() == 0:
            return None
        return remaining

    # -------------------------
    # Small helpers used by env
    # -------------------------
    def is_true_state(self, state: Tensor) -> bool:
        if self.true_tensor is None:
            return False
        if state.dim() == 2:
            state = state.squeeze(0)
        return torch.equal(state, self.true_tensor.squeeze(0))

    def is_false_state(self, state: Tensor) -> bool:
        if self.false_tensor is None:
            return False
        if state.dim() == 2:
            state = state.squeeze(0)
        return torch.equal(state, self.false_tensor.squeeze(0))

    def get_false_state(self) -> Tensor:
        if self.false_tensor is None:
            raise ValueError("False predicate not defined")
        return self.false_tensor.clone()

    def get_true_state(self) -> Tensor:
        if self.true_tensor is None:
            raise ValueError("True predicate not defined")
        return self.true_tensor.clone()
