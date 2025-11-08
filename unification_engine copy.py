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
from unification_ops import deduplicate_states_packed

@torch.no_grad()
def _batch_prune_all_candidates(engine, candidates: Tensor, atom_counts: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Remove ground atoms that are known facts (pure GPU) and detect proofs.
    Args:
        candidates:  [N, M, 3]
        atom_counts: [N]
    Returns:
        pruned_states: [N, M', 3]
        pruned_counts: [N]
        is_proof:     [N]  (True if all atoms removed)
    """
    device = candidates.device
    pad = engine.padding_idx
    if candidates.numel() == 0:
        return (candidates, atom_counts, atom_counts == 0)
    N, M = candidates.shape[:2]
    valid = candidates[:, :, 0] != pad
    args = candidates[:, :, 1:3]
    const_no = engine.constant_no
    is_const = (args <= const_no).all(dim=2)
    ground = valid & is_const
    if ground.any():
        atoms = candidates.reshape(-1, 3)
        ground_flat = ground.flatten()
        ground_atoms = atoms[ground_flat]
        h_atoms = ground_atoms[:, 0].long() * 1000000 + ground_atoms[:, 1].long() * 1000 + ground_atoms[:, 2].long()
        is_in = torch.isin(h_atoms, engine.fact_index.fact_hashes)
        drop_mask_flat = torch.zeros_like(ground_flat, dtype=torch.bool, device=device)
        drop_mask_flat[ground_flat] = is_in
        drop_mask = drop_mask_flat.view(N, M)
    else:
        drop_mask = torch.zeros_like(valid)
    keep = valid & ~drop_mask
    pruned_counts = keep.sum(dim=1)
    is_proof = pruned_counts == 0
    pos = torch.cumsum(keep.long(), dim=1) - 1
    pos = torch.clamp(pos, min=0)
    row = torch.arange(N, device=device).unsqueeze(1).expand_as(pos)
    max_len = int(pruned_counts.max().item()) if pruned_counts.numel() > 0 else 1
    out = torch.full((N, max_len, 3), pad, dtype=torch.int32, device=device)
    if keep.any():
        idx_src = torch.arange(M, device=device).unsqueeze(0).expand_as(keep)
        out[row[keep], pos[keep]] = candidates[row[keep], idx_src[keep]]
    return (out, pruned_counts, is_proof)

@torch.no_grad()
def _unify_with_facts_all_preds(engine, queries: Tensor, remaining_goals: Tensor, remaining_counts: Tensor, pred_indices: Tensor, excluded_queries: Optional[Tensor]=None, labels: Optional[Tensor]=None, max_output_per_query: int=100) -> Tuple[Tensor, Tensor]:
    """Loop-free, all-predicate fact unification.
    Args:
        queries:           [A, 3]
        remaining_goals:   [A, G, 3]
        remaining_counts:  [A]
        pred_indices:      [A]
    Returns:
        states: [A, K, M, 3]
        counts: [A]
    """
    device = queries.device
    pad = engine.padding_idx
    A, G = remaining_goals.shape[:2]
    facts = engine.facts_idx
    F = facts.shape[0]
    if A == 0 or F == 0:
        empty = torch.full((A, 0, 1, 3), pad, dtype=torch.int32, device=device)
        return (empty, torch.zeros(A, dtype=torch.long, device=device))
    match = pred_indices.unsqueeze(1) == facts[:, 0].unsqueeze(0)
    qi, fi = match.nonzero(as_tuple=True)
    if qi.numel() == 0:
        empty = torch.full((A, 0, 1, 3), pad, dtype=torch.int32, device=device)
        return (empty, torch.zeros(A, dtype=torch.long, device=device))
    q_pairs = queries.index_select(0, qi)
    f_pairs = facts.index_select(0, fi)
    ok, subs = _unify_one_to_one_optimized(q_pairs, f_pairs, engine.constant_no, pad)
    if not ok.any():
        empty = torch.full((A, 0, 1, 3), pad, dtype=torch.int32, device=device)
        return (empty, torch.zeros(A, dtype=torch.long, device=device))
    qi = qi[ok]
    subs = subs[ok]
    remaining_sel = remaining_goals.index_select(0, qi)
    subs_b = subs.view(subs.shape[0], -1, 2)
    remain_inst = _apply_substitutions_batched(remaining_sel, subs_b, pad)
    pos_g = torch.arange(G, device=device).unsqueeze(0).expand_as(remain_inst[:, :, 0])
    take_g = pos_g < remaining_counts.index_select(0, qi).unsqueeze(1)
    counts = take_g.sum(1)
    M = int(counts.max().item()) if counts.numel() > 0 else 1
    cat_compact = torch.full((counts.shape[0], M, 3), pad, dtype=torch.int32, device=device)
    if take_g.any():
        row = torch.arange(counts.shape[0], device=device).unsqueeze(1).expand_as(take_g)
        pos = torch.cumsum(take_g.long(), dim=1) - 1
        valid = take_g
        cat_compact[row[valid], pos[valid]] = remain_inst[row[valid], torch.arange(G, device=device).unsqueeze(0).expand_as(take_g)[valid]]
    K_per_owner = torch.bincount(qi, minlength=A)
    K = int(min(max_output_per_query, K_per_owner.max().item())) if K_per_owner.numel() > 0 else 0
    if K == 0:
        empty = torch.full((A, 0, 1, 3), pad, dtype=torch.int32, device=device)
        return (empty, torch.zeros(A, dtype=torch.long, device=device))
    out = torch.full((A, K, M, 3), pad, dtype=torch.int32, device=device)
    counts_per_q = torch.zeros(A, dtype=torch.long, device=device)
    sort_vals, sort_idx = torch.sort(qi)
    owner_sorted = sort_vals
    cat_sorted = cat_compact.index_select(0, sort_idx)
    seg_start = torch.ones_like(owner_sorted, dtype=torch.bool, device=device)
    seg_start[1:] = owner_sorted[1:] != owner_sorted[:-1]
    pos_in_owner = torch.cumsum(seg_start, dim=0) - 1
    keep_assign = pos_in_owner < K
    if keep_assign.any():
        db = owner_sorted[keep_assign]
        ds = pos_in_owner[keep_assign]
        out[db, ds] = cat_sorted[keep_assign]
        maxpos = torch.full((A,), -1, dtype=torch.long, device=device)
        maxpos.index_put_((db,), ds, accumulate=True)
        counts_per_q = torch.clamp(maxpos + 1, min=0, max=K)
    return (out, counts_per_q)

@torch.no_grad()
def _unify_with_rules_all_preds(engine, queries: Tensor, remaining_goals: Tensor, remaining_counts: Tensor, pred_indices: Tensor, max_output_per_query: int=100) -> Tuple[Tensor, Tensor]:
    """Loop-free, all-predicate rule unification.
    Args:
        queries:           [A, 3]
        remaining_goals:   [A, G, 3]
        remaining_counts:  [A]
        pred_indices:      [A] (queries[:,0])
    Returns:
        states: [A, K, M, 3]  (K <= max_output_per_query)
        counts: [A]           (number of produced states per query)
    """
    device = queries.device
    pad = engine.padding_idx
    A, G = remaining_goals.shape[:2]
    if queries.numel() == 0:
        empty = torch.full((0, 0, 1, 3), pad, dtype=torch.int32, device=device)
        return (empty, torch.zeros(0, dtype=torch.long, device=device))
    rule_heads = engine.rules_heads_idx
    R = rule_heads.shape[0]
    if R == 0:
        empty = torch.full((A, 0, 1, 3), pad, dtype=torch.int32, device=device)
        return (empty, torch.zeros(A, dtype=torch.long, device=device))
    match = pred_indices.unsqueeze(1) == rule_heads[:, 0].unsqueeze(0)
    qi, ri = match.nonzero(as_tuple=True)
    if qi.numel() == 0:
        empty = torch.full((A, 0, 1, 3), pad, dtype=torch.int32, device=device)
        return (empty, torch.zeros(A, dtype=torch.long, device=device))
    q_pairs = queries.index_select(0, qi)
    h_pairs = rule_heads.index_select(0, ri)
    ok, subs = _unify_one_to_one_optimized(q_pairs, h_pairs, engine.constant_no, pad)
    if not ok.any():
        empty = torch.full((A, 0, 1, 3), pad, dtype=torch.int32, device=device)
        return (empty, torch.zeros(A, dtype=torch.long, device=device))
    qi = qi[ok]
    ri = ri[ok]
    subs = subs[ok]
    bodies_all = engine.rules_idx.index_select(0, ri)
    body_lens = engine.rule_lens.index_select(0, ri)
    remaining_sel = remaining_goals.index_select(0, qi)
    subs_b = subs.view(subs.shape[0], -1, 2)
    bodies_inst = _apply_substitutions_batched(bodies_all, subs_b, pad)
    remain_inst = _apply_substitutions_batched(remaining_sel, subs_b, pad)
    Bmax = bodies_all.shape[1]
    pos_b = torch.arange(Bmax, device=device).unsqueeze(0).expand_as(bodies_inst[:, :, 0])
    take_b = pos_b < body_lens.unsqueeze(1)
    pos_g = torch.arange(G, device=device).unsqueeze(0).expand_as(remain_inst[:, :, 0])
    take_g = pos_g < remaining_counts.index_select(0, qi).unsqueeze(1)
    counts = take_b.sum(1) + take_g.sum(1)
    M = int(counts.max().item()) if counts.numel() > 0 else 1
    cand = torch.full((counts.shape[0], M, 3), pad, dtype=torch.int32, device=device)
    if take_b.any():
        rb = bodies_inst[:, :, :]
        rb = rb[take_b]
        pos_body = pos_b[take_b]
        row_ids_b = torch.arange(counts.shape[0], device=device).unsqueeze(1).expand(-1, Bmax)[take_b]
        ones_b = torch.ones_like(pos_b, dtype=torch.long)[take_b]
    keep_b = take_b
    rb = bodies_inst
    rg = remain_inst
    cat = torch.full((counts.shape[0], Bmax + G, 3), pad, dtype=torch.int32, device=device)
    cat[:, :Bmax] = rb
    cat[:, Bmax:] = rg
    keep = torch.cat([keep_b, take_g], dim=1)
    pos = torch.cumsum(keep.long(), dim=1) - 1
    pos = torch.clamp(pos, min=0)
    row = torch.arange(counts.shape[0], device=device).unsqueeze(1).expand_as(pos)
    valid = keep
    if valid.any():
        cat_compact = torch.full((counts.shape[0], int(keep.sum(1).max().item()), 3), pad, dtype=torch.int32, device=device)
        cat_compact[row[valid], pos[valid]] = cat[row[valid], torch.arange(keep.shape[1], device=device).unsqueeze(0).expand_as(keep)[valid]]
    else:
        cat_compact = torch.full((counts.shape[0], 1, 3), pad, dtype=torch.int32, device=device)
    K_per_owner = torch.bincount(qi, minlength=A)
    K = int(min(max_output_per_query, K_per_owner.max().item())) if K_per_owner.numel() > 0 else 0
    if K == 0:
        empty = torch.full((A, 0, 1, 3), pad, dtype=torch.int32, device=device)
        return (empty, torch.zeros(A, dtype=torch.long, device=device))
    M_final = cat_compact.shape[1]
    out = torch.full((A, K, M_final, 3), pad, dtype=torch.int32, device=device)
    counts_per_q = torch.zeros(A, dtype=torch.long, device=device)
    sort_vals, sort_idx = torch.sort(qi)
    owner_sorted = sort_vals
    cat_sorted = cat_compact.index_select(0, sort_idx)
    seg_start = torch.ones_like(owner_sorted, dtype=torch.bool, device=device)
    seg_start[1:] = owner_sorted[1:] != owner_sorted[:-1]
    pos_in_owner = torch.cumsum(seg_start, dim=0) - 1
    keep_assign = pos_in_owner < K
    if keep_assign.any():
        db = owner_sorted[keep_assign]
        ds = pos_in_owner[keep_assign]
        out[db, ds] = cat_sorted[keep_assign]
        maxpos = torch.full((A,), -1, dtype=torch.long, device=device)
        maxpos.index_put_((db,), ds, accumulate=True)
        counts_per_q = torch.clamp(maxpos + 1, min=0, max=K)
    return (out, counts_per_q)

class GPUHashCache:
    """Pre-computed prime powers for fast hashing"""

    def __init__(self, device: torch.device, max_len: int=1024):
        self.device = device
        self.prime = 31
        self.mod_val = 2 ** 31 - 1
        self._build_cache(max_len)

    def _build_cache(self, max_len: int):
        """Build cached prime powers"""
        powers = torch.arange(max_len, device=self.device, dtype=torch.int64)
        self.prime_powers = torch.pow(torch.tensor(self.prime, device=self.device, dtype=torch.int64), powers) % self.mod_val
        self.max_len = max_len

    def get_powers(self, length: int) -> Tensor:
        """Get cached powers, extending if necessary"""
        if length > self.max_len:
            self._build_cache(length)
        return self.prime_powers[:length]

class GPUFactIndex:
    """
    GPU-accelerated fact lookup using sorted tensors + searchsorted.
    Replaces Python hash set with pure GPU operations.
    """

    def __init__(self, facts: Tensor):
        """
        Args:
            facts: [F, 3] int32 tensor of ground facts
        """
        self.device = facts.device
        if facts.numel() == 0:
            self.sorted_facts = torch.empty((0, 3), dtype=torch.int32, device=self.device)
            self.fact_hashes = torch.empty((0,), dtype=torch.int64, device=self.device)
            return
        self.sorted_facts = facts.int()
        self.fact_hashes = self.sorted_facts[:, 0].long() * 1000000 + self.sorted_facts[:, 1].long() * 1000 + self.sorted_facts[:, 2].long()
        sort_idx = torch.argsort(self.fact_hashes)
        self.fact_hashes = self.fact_hashes[sort_idx]
        self.sorted_facts = self.sorted_facts[sort_idx]

    @torch.no_grad()
    def contains(self, atoms: Tensor) -> Tensor:
        """
        Check if atoms exist in facts using GPU searchsorted.
        Args:
            atoms: [N, 3] int32 tensor of atoms to check
        Returns:
            mask: [N] bool tensor, True if atom is in facts
        """
        if atoms.numel() == 0 or self.fact_hashes.numel() == 0:
            return torch.zeros(atoms.shape[0], dtype=torch.bool, device=self.device)
        atom_hashes = atoms[:, 0].long() * 1000000 + atoms[:, 1].long() * 1000 + atoms[:, 2].long()
        indices = torch.searchsorted(self.fact_hashes, atom_hashes)
        mask = torch.zeros(atoms.shape[0], dtype=torch.bool, device=self.device)
        valid = indices < self.fact_hashes.shape[0]
        if valid.any():
            mask[valid] = self.fact_hashes[indices[valid]] == atom_hashes[valid]
        return mask

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
    preds = goals_batch[:, :, 0:1]
    args = goals_batch[:, :, 1:]
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
        return (torch.empty(0, dtype=torch.bool, device=device), pad_subs)
    var_start = constant_no + 1
    pred_match = queries[:, 0] == terms[:, 0]
    q_args, t_args = (queries[:, 1:], terms[:, 1:])
    q_var = q_args >= var_start
    t_var = t_args >= var_start
    const_mismatch = ~q_var & ~t_var & (q_args != t_args)
    initial_unifiable = pred_match & ~const_mismatch.any(dim=1)
    mask = initial_unifiable
    subs = torch.full((N, 2, 2), padding_idx, dtype=torch.long, device=device)
    idx = mask.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return (mask, subs)
    q = q_args[idx]
    t = t_args[idx]
    qv = q >= var_start
    tv = t >= var_start
    qv_tc = qv & ~tv
    if qv_tc.any():
        qv_vars = q[qv_tc].to(dtype=torch.long)
        qv_vals = t[qv_tc].to(dtype=torch.long)
        rows = qv_tc.nonzero(as_tuple=True)[0]
        cols = qv_tc.nonzero(as_tuple=True)[1]
        slots = cols
        subs_rows = idx[rows]
        subs[subs_rows, slots, 0] = qv_vars
        subs[subs_rows, slots, 1] = qv_vals
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
    return (mask, subs)

class UnificationEngine:
    """Encapsulates facts/rules and batched single-step unification."""

    @classmethod
    def from_index_manager(cls, index_manager, take_ownership: bool=False) -> 'UnificationEngine':
        rules_heads = getattr(index_manager, 'rules_heads_idx', None)
        engine = cls(facts_idx=index_manager.facts_idx, rules_idx=index_manager.rules_idx, rule_lens=index_manager.rule_lens, padding_idx=index_manager.padding_idx, constant_no=index_manager.constant_no, runtime_var_end_index=index_manager.runtime_var_end_index, true_pred_idx=index_manager.true_pred_idx, false_pred_idx=index_manager.false_pred_idx, max_arity=index_manager.max_arity, predicate_range_map=getattr(index_manager, 'predicate_range_map', None), rules_heads_idx=rules_heads, device=index_manager.device)
        if take_ownership:
            index_manager.facts_idx = None
            index_manager.rules_idx = None
            index_manager.rule_lens = None
            index_manager.rules_heads_idx = None
        return engine

    def get_derived_states(self, current_states: Tensor, next_var_indices: Tensor, excluded_queries: Optional[Tensor]=None, labels: Optional[Tensor]=None, verbose: int=0) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Fully vectorised single-step expansion for a batch of states.
        Follows the requested flow strictly:

        1) Preprocessing
        2) Rule unification
        3) Fact unification
        4) Prune ground facts & detect immediate proof
        5) Combine into next states
        6) Canonicalise + deduplicate
        7) Pad back to original batch shape
        """
        device = current_states.device
        pad = self.padding_idx
        B, max_atoms, _ = current_states.shape
        valid_mask = current_states[:, :, 0] != pad
        has_any = valid_mask.any(dim=1)
        empty_states = ~has_any
        if self.false_pred_idx is not None:
            has_false = (current_states[:, :, 0] == self.false_pred_idx).any(dim=1)
        else:
            has_false = torch.zeros(B, dtype=torch.bool, device=device)
        if self.true_pred_idx is not None:
            only_true = has_any & ((current_states[:, :, 0] == self.true_pred_idx) | ~valid_mask).all(dim=1)
        else:
            only_true = torch.zeros(B, dtype=torch.bool, device=device)
        terminal = empty_states | has_false | only_true
        max_derived_per_state = 128
        all_derived = torch.full((B, max_derived_per_state, max_atoms, 3), pad, dtype=torch.int32, device=device)
        derived_counts = torch.zeros(B, dtype=torch.long, device=device)
        updated_next_var_indices = next_var_indices.clone()
        if only_true.any():
            all_derived[only_true, 0, 0] = self.true_tensor.squeeze(0)
            derived_counts[only_true] = 1
        if (has_false | empty_states).any():
            mf = (has_false | empty_states) & ~only_true
            all_derived[mf, 0, 0] = self.false_tensor.squeeze(0)
            derived_counts[mf] = 1
        active = ~terminal
        if not active.any():
            return (all_derived, derived_counts, updated_next_var_indices)
        active_idx = active.nonzero(as_tuple=True)[0]
        A = active_idx.shape[0]
        active_states = current_states[active_idx]
        active_valid = valid_mask[active_idx]
        first_pos = active_valid.long().argmax(dim=1)
        arangeA = torch.arange(A, device=device)
        queries = active_states[arangeA, first_pos]
        remaining_goals = torch.full_like(active_states, pad)
        remaining_counts = (active_valid.sum(dim=1) - 1).clamp(min=0)
        if A > 0:
            pos_grid = torch.arange(max_atoms, device=device).unsqueeze(0).expand(A, -1)
            before_mask = active_valid & (pos_grid < first_pos.unsqueeze(1))
            after_mask = active_valid & (pos_grid > first_pos.unsqueeze(1))
            dst_pos = torch.zeros_like(pos_grid)
            dst_pos = torch.where(before_mask, pos_grid, dst_pos)
            dst_pos = torch.where(after_mask, pos_grid - 1, dst_pos)
            scatter_mask = before_mask | after_mask
            if scatter_mask.any():
                src_batch = arangeA.unsqueeze(1).expand(-1, max_atoms)[scatter_mask]
                src_atom = pos_grid[scatter_mask]
                dst_batch = src_batch
                dst_atom = dst_pos[scatter_mask]
                remaining_goals[dst_batch, dst_atom] = active_states[src_batch, src_atom]
        preds = queries[:, 0]
        rule_states, rule_counts = _unify_with_rules_all_preds(self, queries, remaining_goals, remaining_counts, preds)
        fact_states, fact_counts = _unify_with_facts_all_preds(self, queries, remaining_goals, remaining_counts, preds, excluded_queries=excluded_queries[active_idx] if excluded_queries is not None else None, labels=labels[active_idx] if labels is not None else None)
        immediate = (remaining_counts == 0) & (fact_counts > 0)
        Rmax, Fmax = (rule_states.shape[1], fact_states.shape[1])
        max_atoms_rules = rule_states.shape[2] if rule_states.numel() > 0 else max_atoms
        max_atoms_facts = fact_states.shape[2] if fact_states.numel() > 0 else max_atoms
        max_atoms_comb = max(max_atoms_rules, max_atoms_facts, max_atoms)
        total_slots = Rmax + Fmax
        packed = torch.full((A, total_slots, max_atoms_comb, 3), pad, dtype=torch.int32, device=device)
        if Rmax > 0:
            packed[:, :Rmax, :max_atoms_rules] = rule_states
        if Fmax > 0:
            packed[:, Rmax:Rmax + Fmax, :max_atoms_facts] = fact_states
        owner_rule_mask = torch.arange(Rmax, device=device).unsqueeze(0).expand(A, -1) < rule_counts.unsqueeze(1)
        owner_fact_mask = torch.arange(Fmax, device=device).unsqueeze(0).expand(A, -1) < fact_counts.unsqueeze(1)
        owner_mask = torch.cat([owner_rule_mask, owner_fact_mask], dim=1) & (~immediate).unsqueeze(1)
        if not owner_mask.any():
            if immediate.any():
                dst = active_idx[immediate]
                all_derived[dst, 0, 0] = self.true_tensor.squeeze(0)
                derived_counts[dst] = 1
            return (all_derived, derived_counts, updated_next_var_indices)
        owner_ids, slot_ids = owner_mask.nonzero(as_tuple=True)
        candidates = packed[owner_ids, slot_ids]
        cand_counts = (candidates[:, :, 0] != pad).sum(dim=1)
        pruned_states, pruned_counts, is_proof = _batch_prune_all_candidates(self, candidates, cand_counts)
        owners_for_cands = active_idx[owner_ids]
        if immediate.any():
            dst = active_idx[immediate]
            all_derived[dst, 0, 0] = self.true_tensor.squeeze(0)
            derived_counts[dst] = 1
        if is_proof.any():
            proof_owners = torch.unique(owners_for_cands[is_proof])
            all_derived[proof_owners, 0, 0] = self.true_tensor.squeeze(0)
            derived_counts[proof_owners] = 1
        keep_mask = ~is_proof
        if not keep_mask.any():
            return (all_derived, derived_counts, updated_next_var_indices)
        owners_kept = owners_for_cands[keep_mask]
        states_kept = pruned_states[keep_mask]
        counts_kept = pruned_counts[keep_mask]
        canon_states, updated_next_vars_per_state = self._canonicalize_variables_batched(states_kept, counts_kept, self.constant_no, next_var_indices[owners_kept], pad)
        if canon_states.shape[0] > 0:
            owner_max = torch.full((B,), 0, dtype=next_var_indices.dtype, device=device)
            owner_max.index_put_((owners_kept,), updated_next_vars_per_state, accumulate=True)
            updated_next_var_indices = torch.maximum(updated_next_var_indices, owner_max)
        K = min(total_slots, max_derived_per_state)
        packed_states = torch.full((B, K, max_atoms_comb, 3), pad, dtype=torch.int32, device=device)
        packed_counts = torch.zeros(B, dtype=torch.long, device=device)
        if owners_kept.numel() > 0:
            sort_vals, sort_idx = torch.sort(owners_kept)
            states_sorted = canon_states[sort_idx]
            counts_sorted = counts_kept[sort_idx]
            seg_start = torch.ones_like(sort_vals, dtype=torch.bool, device=device)
            seg_start[1:] = sort_vals[1:] != sort_vals[:-1]
            pos = torch.cumsum(seg_start, dim=0) - 1
            pos = torch.minimum(pos, torch.full_like(pos, K - 1))
            dst_batch = sort_vals
            dst_slot = pos
            packed_states[dst_batch, dst_slot] = states_sorted
            owner_maxpos = torch.full((B,), -1, dtype=torch.long, device=device)
            owner_maxpos.index_put_((dst_batch,), dst_slot, accumulate=True)
            packed_counts = torch.clamp(owner_maxpos + 1, min=0, max=K)
        unique_states, unique_counts = deduplicate_states_packed(packed_states, packed_counts, pad, getattr(self, 'hash_cache', None))
        write_mask = unique_counts > 0
        write_idx = write_mask.nonzero(as_tuple=True)[0]
        if write_idx.numel() > 0:
            max_atoms_write = min(max_atoms_comb, max_atoms)
            all_derived[write_idx, :unique_states.shape[1], :max_atoms_write] = unique_states[write_idx, :, :max_atoms_write]
            derived_counts[write_idx] = unique_counts[write_idx]
        return (all_derived, derived_counts, updated_next_var_indices)

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
            raise ValueError('False predicate not defined')
        return self.false_tensor.clone()

    def get_true_state(self) -> Tensor:
        if self.true_tensor is None:
            raise ValueError('True predicate not defined')
        return self.true_tensor.clone()