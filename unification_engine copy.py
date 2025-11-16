
# unification_engine_modular.py
# -----------------------------------------------------------------------------
# A clean, modular rewrite of the batched single-step unification pipeline.
# The focus is on a *readable* and *maintainable* orchestration of:
#
#   1) preprocessing
#   2) rule unification
#   3) fact unification
#   4) combine (rules ⊕ facts)
#   5) prune & collapse proofs (this is the ONLY place with pruning)
#   6) standardize/canonicalize runtime variables
#   7) deduplicate (optional)
#   8) pad back to original batch size (drop-in compatibility)
#
# All functions carry shape comments and short explanations.
#
# Conventions:
#   - "B" is the batch size (#input states)
#   - "A" is the number of *active* states (after filtering terminals)
#   - "K_r" / "K_f" = rule/fact successors per active state
#   - "K"   = total successors per active state (K_r + K_f)
#   - "N"   = total number of candidates across the active set
#   - "M"   = atoms per candidate state
#   - Triples are shaped [*, 3] with layout [pred, arg0, arg1]
#   - padding_idx is used in all dimensions where variable-length packing occurs
#
# Notes:
#   - All pruning (replacing ground facts by True / dropping them) happens in
#     `prune_and_collapse`. Neither `_unify_with_facts` nor `_unify_with_rules`
#     is allowed to do proof detection or pruning.
#   - Standardization ensures that *all* derived states from the *same* owner
#     start renumbering new runtime variables at the *same* base (next_var_index).
# -----------------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch
from torch import Tensor


# ============================================================================
# Small helpers
# ============================================================================

@torch.no_grad()
def _pack_triples_64(atoms: Tensor, base: int) -> Tensor:
    """
    Pack triples [pred, a, b] into 64-bit integers for fast set operations.
    atoms: [N, 3]  (long), base >= max index + 1
    return: [N]  (int64)
    hash = ((pred * base) + a) * base + b
    """
    if atoms.numel() == 0:
        return torch.empty(0, dtype=torch.int64, device=atoms.device)
    p, a, b = atoms[:, 0].long(), atoms[:, 1].long(), atoms[:, 2].long()
    base = torch.as_tensor(base, dtype=torch.int64, device=atoms.device)
    return ((p * base) + a) * base + b


class GPUHashCache:
    """
    Tiny cache for polynomial hashing used in deduplication.
    The larger mod reduces collisions vs small 32-bit primes.
    """
    def __init__(self, device: torch.device, max_len: int = 4096):
        self.device = device
        self.prime = 31
        self.mod_val = 2**61 - 1
        self._build_cache(max_len)

    def _build_cache(self, max_len: int):
        powers = torch.arange(max_len, device=self.device, dtype=torch.int64)
        self.prime_powers = torch.pow(torch.tensor(self.prime, device=self.device, dtype=torch.int64), powers) % self.mod_val
        self.max_len = max_len

    def get_powers(self, length: int) -> Tensor:
        if length > self.max_len:
            self._build_cache(length)
        return self.prime_powers[:length]


class GPUFactIndex:
    """
    GPU-accelerated fact membership using searchsorted over packed 64-bit keys.
    facts: [F, 3] (long)
    """
    def __init__(self, facts: Tensor, pack_base: int):
        self.device = facts.device
        self.pack_base = int(pack_base)
        if facts.numel() == 0:
            self.fact_hashes = torch.empty(0, dtype=torch.int64, device=self.device)
        else:
            self.fact_hashes = _pack_triples_64(facts.long(), self.pack_base).sort()[0]

    @torch.no_grad()
    def contains(self, atoms: Tensor) -> Tensor:
        """
        atoms: [N, 3]  -> return [N] boolean
        """
        if atoms.numel() == 0 or self.fact_hashes.numel() == 0:
            return torch.zeros((atoms.shape[0],), dtype=torch.bool, device=atoms.device)
        keys = _pack_triples_64(atoms.long(), self.pack_base)
        idx  = torch.searchsorted(self.fact_hashes, keys)
        mask = torch.zeros_like(keys, dtype=torch.bool)
        valid = idx < self.fact_hashes.shape[0]
        mask[valid] = (self.fact_hashes[idx[valid]] == keys[valid])
        return mask


# ============================================================================
# Deduplication (per-owner, on packed [B, K, M, 3])
# ============================================================================

@torch.no_grad()
def deduplicate_states_packed(
    states: Tensor,       # [B, K, M, 3]
    counts: Tensor,       # [B]
    padding_idx: int,
    hash_cache: Optional[GPUHashCache] = None
) -> Tuple[Tensor, Tensor]:
    """
    Stateless GPU dedup (no Python loops). Dedup compares whole candidate states.
    Returns compact [B, K', M, 3] with corresponding counts [B].
    """
    if states.numel() == 0:
        return states, counts

    B, K, M, D = states.shape
    device = states.device

    # Flatten each state -> [B, K, M*3]
    flat = states.reshape(B, K, -1).long()

    # Valid states per owner mask
    valid_mask = (torch.arange(K, device=device).unsqueeze(0) < counts.unsqueeze(1))  # [B, K]

    # Polynomial hash per state [B, K]
    prime = 31
    mod_val = 2**61 - 1
    L = M * D
    if hash_cache is None:
        powers = torch.arange(L, device=device, dtype=torch.int64)
        prime_powers = torch.pow(torch.tensor(prime, device=device, dtype=torch.int64), powers) % mod_val
    else:
        prime_powers = hash_cache.get_powers(L)

    hashes = (flat * prime_powers.view(1, 1, -1)).sum(dim=2) % mod_val
    hashes = torch.where(valid_mask, hashes, torch.full_like(hashes, mod_val))

    # Sort by hash (batched) and compare neighbors
    sorted_hashes, sort_idx = torch.sort(hashes, dim=1)
    sorted_states = torch.gather(states, 1, sort_idx.view(B, K, 1, 1).expand(B, K, M, D))

    # First occurrence or different content -> unique
    unique_mask = torch.ones((B, K), dtype=torch.bool, device=device)
    if K > 1:
        hash_diff = sorted_hashes[:, 1:] != sorted_hashes[:, :-1]  # [B, K-1]
        same_hash = ~hash_diff
        if same_hash.any():
            eq = (sorted_states[:, 1:] == sorted_states[:, :-1]).all(dim=(2, 3))  # [B, K-1]
            unique_mask[:, 1:] = hash_diff | ~eq
        else:
            unique_mask[:, 1:] = hash_diff

    unique_mask &= (sort_idx < counts.unsqueeze(1))

    # Count uniques and scatter to compact output
    uniq_counts = unique_mask.sum(dim=1)  # [B]
    if uniq_counts.max() == 0:
        return torch.full((B, 0, M, D), padding_idx, dtype=states.dtype, device=device), uniq_counts

    pos = torch.cumsum(unique_mask.long(), dim=1) - 1  # [B, K]
    outK = int(K)  # upper bound
    out = torch.full((B, outK, M, D), padding_idx, dtype=states.dtype, device=device)

    b_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(pos)[unique_mask]
    u_idx = pos[unique_mask]
    out[b_idx, u_idx] = sorted_states[unique_mask.unsqueeze(-1).unsqueeze(-1).expand_as(sorted_states)].reshape(-1, M, D)
    return out, uniq_counts


# ============================================================================
# Substitution / pairwise unify helpers
# ============================================================================

@torch.no_grad()
def apply_substitutions(goals: Tensor, subs_pairs: Tensor, padding_idx: int) -> Tensor:
    """
    Apply substitutions selectively on args only (columns 1,2).
    goals:      [N, M, 3]
    subs_pairs: [N, S, 2] where each pair is [from, to], padded with padding_idx.
    return:     [N, M, 3]
    """
    if goals.numel() == 0:
        return goals
    N, M = goals.shape[:2]
    preds = goals[:, :, 0:1]
    args = goals[:, :, 1:].clone()

    valid = subs_pairs[..., 0] != padding_idx
    if not valid.any():
        return goals

    S = subs_pairs.shape[1]
    for s in range(S):
        mask = valid[:, s]
        if not mask.any():
            continue
        rows = torch.nonzero(mask, as_tuple=False).squeeze(1)
        frm = subs_pairs[rows, s, 0].view(-1, 1, 1)
        to  = subs_pairs[rows, s, 1].view(-1, 1, 1)
        args_sel = args.index_select(0, rows)
        args_sel = torch.where(args_sel == frm, to, args_sel)
        args.index_copy_(0, rows, args_sel)

    return torch.cat([preds, args], dim=2)


@torch.no_grad()
def unify_one_to_one(queries: Tensor, terms: Tensor, constant_no: int, padding_idx: int) -> Tuple[Tensor, Tensor]:
    """
    Pairwise unify rows of queries vs rows of terms (same length).
    queries: [L, 3]
    terms:   [L, 3]
    return:
      mask: [L]                  -> rows that unify
      subs: [L, 2, 2] (padding)  -> up to 2 pairs [from, to] for the two args
    """
    device = queries.device
    L = queries.shape[0]
    if L == 0:
        return (torch.empty(0, dtype=torch.bool, device=device),
                torch.full((0, 2, 2), padding_idx, dtype=torch.long, device=device))

    var_start = constant_no + 1
    pred_ok = (queries[:, 0] == terms[:, 0])

    q_args, t_args = queries[:, 1:], terms[:, 1:]
    q_const, t_const = (q_args <= constant_no), (t_args <= constant_no)
    const_conflict = (q_const & t_const & (q_args != t_args)).any(dim=1)
    


    mask = pred_ok & ~const_conflict
    subs = torch.full((L, 2, 2), padding_idx, dtype=torch.long, device=device)
    if not mask.any():
        return mask, subs

    idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
    q = q_args.index_select(0, idx)
    t = t_args.index_select(0, idx)
    qv = (q >= var_start) & (q != padding_idx)
    tv = (t >= var_start) & (t != padding_idx)

    subs_sel = torch.full((idx.numel(), 2, 2), padding_idx, dtype=torch.long, device=device)
    from_slot = subs_sel[:, :, 0]
    to_slot   = subs_sel[:, :, 1]

    # qVar <- tConst
    case1 = qv & (~tv) & (t != 0)
    subs_sel[:, :, 0] = torch.where(case1, q.long(), from_slot)
    subs_sel[:, :, 1] = torch.where(case1, t.long(), to_slot)

    # tVar <- qConst
    case2 = (~qv) & (q != 0) & tv
    subs_sel[:, :, 0] = torch.where(case2, t.long(), subs_sel[:, :, 0])
    subs_sel[:, :, 1] = torch.where(case2, q.long(), subs_sel[:, :, 1])

    # both variables
    case3 = qv & tv
    subs_sel[:, :, 0] = torch.where(case3, t.long(), subs_sel[:, :, 0])
    subs_sel[:, :, 1] = torch.where(case3, q.long(), subs_sel[:, :, 1])

    # consistency: same variable cannot map to different targets in one row
    same_var  = (subs_sel[:, :, 0][:, 0] == subs_sel[:, :, 0][:, 1]) & (subs_sel[:, :, 0][:, 0] != padding_idx)
    diff_tgt  = subs_sel[:, :, 1][:, 0] != subs_sel[:, :, 1][:, 1]
    conflict  = same_var & diff_tgt
    if conflict.any():
        bad = idx[conflict]
        mask[bad] = False
        subs_sel[conflict] = padding_idx

    subs.index_copy_(0, idx, subs_sel)
    return mask, subs


# ============================================================================
# Pairing by predicate range
# ============================================================================

@torch.no_grad()
def pairs_via_predicate_ranges(query_preds: Tensor, seg_starts: Tensor, seg_lens: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Build all (query_idx, item_idx) pairs for queries grouped by predicate ranges.
    query_preds: [A]
    seg_starts:  [P]  start index per predicate in the items (facts or rule-heads)
    seg_lens:    [P]  length per predicate in the items
    return:
      qi: [L] indices into queries
      ii: [L] indices into items
    """
    device = query_preds.device
    if query_preds.numel() == 0:
        z = torch.zeros(0, dtype=torch.long, device=device)
        return z, z

    lens   = seg_lens[query_preds.long()]        # [A]
    starts = seg_starts[query_preds.long()]      # [A]
    keep   = lens > 0
    kept_q = torch.arange(query_preds.shape[0], device=device)[keep]
    if kept_q.numel() == 0:
        z = torch.zeros(0, dtype=torch.long, device=device)
        return z, z

    lens_kept   = lens[kept_q]
    starts_kept = starts[kept_q]

    row_ids = torch.repeat_interleave(torch.arange(lens_kept.numel(), device=device), lens_kept)
    if row_ids.numel() == 0:
        z = torch.zeros(0, dtype=torch.long, device=device)
        return z, z
    prefix = torch.cumsum(lens_kept, dim=0) - lens_kept
    pos_in = torch.arange(row_ids.numel(), device=device) - prefix[row_ids]
    item_idx = starts_kept[row_ids] + pos_in
    query_idx = kept_q[row_ids]
    return query_idx, item_idx


# ============================================================================
# Modular pipeline pieces
# ============================================================================

@dataclass
class PreprocessResult:
    active_idx: Tensor           # [A]
    queries: Tensor              # [A, 3]
    remaining: Tensor            # [A, G, 3]
    remaining_counts: Tensor     # [A]
    preds: Tensor                # [A]
    terminal_true: Tensor        # [T1] owners that are trivially TRUE
    terminal_false: Tensor       # [T2] owners that are trivially FALSE


@torch.no_grad()
def preprocess_states(states: Tensor,
                      true_pred_idx: Optional[int],
                      false_pred_idx: Optional[int],
                      padding_idx: int) -> PreprocessResult:
    """
    Split batch into terminal vs. active and extract (query, rest).
    states: [B, max_atoms, 3]
    return: PreprocessResult
    """
    device = states.device
    B, max_atoms = states.shape[:2]
    pad = padding_idx

    valid = (states[:, :, 0] != pad)                    # [B, max_atoms]
    has_any = valid.any(dim=1)                          # [B]
    empty   = ~has_any

    has_false = torch.zeros(B, dtype=torch.bool, device=device)
    only_true = torch.zeros(B, dtype=torch.bool, device=device)

    has_false = (states[:, :, 0] == false_pred_idx).any(dim=1)
    # "only_true": all non-pad preds are TRUE
    only_true = (valid & (states[:, :, 0] == true_pred_idx) | ~valid).all(dim=1) & has_any

    terminal_true  = only_true
    terminal_false = empty | has_false

    active = ~(terminal_true | terminal_false)
    active_idx = torch.nonzero(active, as_tuple=False).view(-1)
    A = active_idx.numel()

    if A == 0:
        # Return empty placeholders for active pieces
        z3  = torch.empty((0, 3), dtype=states.dtype, device=device)
        z33 = torch.empty((0, max_atoms, 3), dtype=states.dtype, device=device)
        zA  = torch.empty((0,), dtype=torch.long, device=device)
        return PreprocessResult(
            active_idx=zA, queries=z3, remaining=z33, remaining_counts=zA,
            preds=zA, terminal_true=torch.nonzero(terminal_true, as_tuple=False).view(-1),
            terminal_false=torch.nonzero(terminal_false, as_tuple=False).view(-1)
        )

    sA = states.index_select(0, active_idx)            # [A, max_atoms, 3]
    validA = valid.index_select(0, active_idx)         # [A, max_atoms]

    # First goal (leftmost) per active row
    first_pos = validA.long().argmax(dim=1)            # [A]
    arangeA = torch.arange(A, device=device)
    queries = sA[arangeA, first_pos]                   # [A, 3]

    # Remaining goals (drop the first one, keep order)
    remaining = torch.full_like(sA, pad)
    rem_counts = (validA.sum(dim=1) - 1).clamp(min=0)  # [A]
    if A > 0:
        pos = torch.arange(max_atoms, device=device).unsqueeze(0).expand(A, -1)
        before = validA & (pos < first_pos.unsqueeze(1))
        after  = validA & (pos > first_pos.unsqueeze(1))
        scatter = before | after
        if scatter.any():
            rows = arangeA.view(-1, 1).expand_as(pos)[scatter]
            src  = pos[scatter]
            dst  = torch.where(before, pos, pos - 1)[scatter]
            remaining[rows, dst] = sA[rows, src]

    preds = queries[:, 0]                               # [A]
    return PreprocessResult(
        active_idx=active_idx, queries=queries, remaining=remaining,
        remaining_counts=rem_counts, preds=preds,
        terminal_true=torch.nonzero(terminal_true, as_tuple=False).view(-1),
        terminal_false=torch.nonzero(terminal_false, as_tuple=False).view(-1)
    )


@torch.no_grad()
def unify_with_facts(  # PURE: no pruning/proof detection here
    facts_idx: Tensor,                    # [F, 3]
    predicate_range_map: Optional[Tensor],# [P, 2] (start,end) per predicate over facts_idx
    queries: Tensor,                      # [A, 3]
    remaining: Tensor,                    # [A, G, 3]
    remaining_counts: Tensor,             # [A]
    preds: Tensor,                        # [A]
    constant_no: int,
    padding_idx: int,
    fact_index: GPUFactIndex,
    excluded_queries: Optional[Tensor] = None,         # [A, max_atoms, 3] -> drop trivial self-match
    verbose: int = 0
) -> Tuple[Tensor, Tensor]:
    """
    One-step fact unification via predicate ranges.
    Returns:
      states: [A, K_f, M_f, 3]  (padded)
      counts: [A]
    """
    device = queries.device
    pad = padding_idx
    A, G = remaining.shape[:2]

    if A == 0 or facts_idx.numel() == 0:
        return torch.full((A, 0, 1, 3), pad, dtype=torch.long, device=device), torch.zeros(A, dtype=torch.long, device=device)

    q_args = queries[:, 1:]
    ground = (q_args <= constant_no).all(dim=1)   # [A]

    qi_accum: List[Tensor] = []
    rem_accum: List[Tensor] = []
    cnt_accum: List[Tensor] = []

    # Case 1: ground queries -> O(1) membership test
    g_idx = torch.nonzero(ground, as_tuple=False).view(-1)
    if g_idx.numel() > 0:
        gq = queries.index_select(0, g_idx)
        hits = fact_index.contains(gq)            # [|g_idx|]
        if excluded_queries is not None:
            excl_first = excluded_queries.index_select(0, g_idx)[:, 0, :]
            hits = hits & ~( (excl_first == gq).all(dim=1) )
        if hits.any():
            keep = g_idx[hits]
            qi_accum.append(keep)
            rem_accum.append(remaining.index_select(0, keep))
            cnt_accum.append(remaining_counts.index_select(0, keep))

    # Case 2: non-ground -> pair by predicate range then unify
    ng_mask = ~ground
    if ng_mask.any():
        ng_idx  = torch.nonzero(ng_mask, as_tuple=False).view(-1)
        q_ng    = queries.index_select(0, ng_idx)
        p_ng    = preds.index_select(0, ng_idx)

        if predicate_range_map is None or predicate_range_map.numel() == 0:
            # Fallback: pair with all facts (less efficient, but simple)
            seg_starts = torch.zeros((int(facts_idx[:,0].max().item())+2,), dtype=torch.long, device=device)
            seg_lens   = torch.zeros_like(seg_starts)
            # compute ranges on-the-fly
            order = torch.argsort(facts_idx[:,0])
            facts_sorted = facts_idx.index_select(0, order)
            preds_sorted = facts_sorted[:,0]
            uniq, counts = torch.unique_consecutive(preds_sorted, return_counts=True)
            starts = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.long, device=device), counts[:-1]]), dim=0)
            seg_starts[uniq] = starts
            seg_lens[uniq]   = counts
            qi_local, fi = pairs_via_predicate_ranges(p_ng, seg_starts, seg_lens)
            facts_for_pred = facts_sorted
        else:
            seg_starts = predicate_range_map[:,0].long()
            seg_lens   = (predicate_range_map[:,1] - predicate_range_map[:,0]).long()
            qi_local, fi = pairs_via_predicate_ranges(p_ng, seg_starts, seg_lens)
            facts_for_pred = facts_idx

        if qi_local.numel() > 0:
            q_pairs = q_ng.index_select(0, qi_local)
            f_pairs = facts_for_pred.index_select(0, fi)

            # Filter: where query has a constant, fact must match that constant
            q_const = q_pairs[:, 1:] <= constant_no
            matches = (~q_const | (q_pairs[:, 1:] == f_pairs[:, 1:])).all(dim=1)
            if matches.any():
                qi_local = qi_local[matches]
                f_pairs  = f_pairs[matches]
                q_pairs  = q_pairs[matches]
            else:
                qi_local = torch.empty(0, dtype=torch.long, device=device)

        if qi_local.numel() > 0:
            ok, subs = unify_one_to_one(q_pairs, f_pairs, constant_no, pad)
            if ok.any():
                qi_ok = ng_idx.index_select(0, qi_local[ok])   # map back to [A]
                subs_ok = subs[ok].view(-1, 2, 2)

                # Optional drop of trivial self-match
                if excluded_queries is not None and qi_ok.numel() > 0:
                    f_ok = f_pairs[ok]
                    excl_atoms = excluded_queries.index_select(0, qi_ok)[:, 0, :]
                    not_excluded = (f_ok != excl_atoms).any(dim=1)
                    if not_excluded.any():
                        qi_ok   = qi_ok[not_excluded]
                        subs_ok = subs_ok[not_excluded]
                    else:
                        qi_ok = torch.empty(0, dtype=torch.long, device=device)

                if qi_ok.numel() > 0:
                    rem_sel = remaining.index_select(0, qi_ok)
                    rem_inst = apply_substitutions(rem_sel, subs_ok.view(subs_ok.shape[0], -1, 2), pad)
                    qi_accum.append(qi_ok)
                    rem_accum.append(rem_inst)
                    cnt_accum.append(remaining_counts.index_select(0, qi_ok))

    if not qi_accum:
        return torch.full((A, 0, 1, 3), pad, dtype=torch.long, device=device), torch.zeros(A, dtype=torch.long, device=device)

    qi = torch.cat(qi_accum, dim=0)                # [L]
    remain_inst = torch.cat(rem_accum, dim=0)      # [L, G, 3]
    rem_counts  = torch.cat(cnt_accum, dim=0)      # [L]

    # Pack states per owner (no proof/True here)
    G = remain_inst.shape[1] if remain_inst.ndim == 3 else 1
    states_per_match = remain_inst[:, :G] if remain_inst.numel() > 0 else torch.empty((0, G, 3), dtype=torch.long, device=device)
    states_per_owner = torch.bincount(qi, minlength=A) if qi.numel() > 0 else torch.zeros(A, dtype=torch.long, device=device)

    if states_per_owner.max() == 0:
        return torch.full((A, 0, 1, 3), pad, dtype=torch.long, device=device), torch.zeros(A, dtype=torch.long, device=device)

    Kf = int(states_per_owner.max().item())
    out = torch.full((A, Kf, G, 3), pad, dtype=torch.long, device=device)
    if qi.numel() > 0:
        owner_sorted, perm = torch.sort(qi, stable=True)
        cat_sorted = states_per_match.index_select(0, perm)
        seg_start = torch.ones_like(owner_sorted, dtype=torch.bool, device=device)
        seg_start[1:] = owner_sorted[1:] != owner_sorted[:-1]
        seg_indices = torch.arange(owner_sorted.shape[0], device=device)
        seg_first   = torch.zeros_like(owner_sorted, dtype=torch.long)
        seg_first[seg_start] = seg_indices[seg_start]
        seg_first = torch.cummax(seg_first, dim=0)[0]
        pos_in_owner = seg_indices - seg_first
        out[owner_sorted, pos_in_owner] = cat_sorted
    return out, states_per_owner


@torch.no_grad()
def unify_with_rules(
    rules_heads_sorted: Tensor,     # [R, 3] rule heads sorted by predicate
    rules_bodies_sorted: Tensor,    # [R, Bmax, 3] rule bodies sorted in the same order
    rule_lens_sorted: Tensor,       # [R] body lengths
    rule_seg_starts: Tensor,        # [P] start idx per predicate (heads)
    rule_seg_lens: Tensor,          # [P] len per predicate (heads)
    queries: Tensor,                # [A, 3]
    remaining: Tensor,              # [A, G, 3]
    remaining_counts: Tensor,       # [A]
    preds: Tensor,                  # [A]
    constant_no: int,
    padding_idx: int,
    next_var_indices: Tensor        # [A]
) -> Tuple[Tensor, Tensor]:
    """
    One-step rule unification using range pairing against sorted rule heads.
    Returns:
      states: [A, K_r, M_r, 3]
      counts: [A]
    """
    device = queries.device
    pad = padding_idx
    A, G = remaining.shape[:2]

    if A == 0 or rules_heads_sorted.numel() == 0:
        return torch.full((A, 0, 1, 3), pad, dtype=torch.long, device=device), torch.zeros(A, dtype=torch.long, device=device)

    # Pair by predicate
    qi, ri = pairs_via_predicate_ranges(preds, rule_seg_starts, rule_seg_lens)  # [L], [L]
    if qi.numel() == 0:
        return torch.full((A, 0, 1, 3), pad, dtype=torch.long, device=device), torch.zeros(A, dtype=torch.long, device=device)

    q_pairs = queries.index_select(0, qi)                    # [L, 3]
    h_templ = rules_heads_sorted.index_select(0, ri)         # [L, 3]
    b_templ = rules_bodies_sorted.index_select(0, ri)        # [L, Bmax, 3]
    Bmax    = b_templ.shape[1]

    # Freshen template vars: rename rule vars to runtime vars starting at next_var_indices[qi]
    template_start = constant_no + 1
    next_for_match = next_var_indices.index_select(0, qi)    # [L]

    # Head args rename
    h_pairs = h_templ.clone()
    for arg in (1, 2):
        args = h_pairs[:, arg]
        is_t = (args >= template_start) & (args != pad)
        if is_t.any():
            h_pairs[:, arg] = torch.where(is_t, next_for_match + (args - template_start), args)

    # Body args rename
    b_pairs = b_templ.clone()
    for arg in (1, 2):
        args = b_pairs[:, :, arg]                            # [L, Bmax]
        is_t = (args >= template_start) & (args != pad)
        if is_t.any():
            b_pairs[:, :, arg] = torch.where(is_t, next_for_match.unsqueeze(1) + (args - template_start), args)

    ok, subs = unify_one_to_one(q_pairs, h_pairs, constant_no, pad)
    if not ok.any():
        return torch.full((A, 0, 1, 3), pad, dtype=torch.long, device=device), torch.zeros(A, dtype=torch.long, device=device)

    qi = qi[ok]
    b_pairs = b_pairs[ok]                                    # [M', Bmax, 3]
    subs   = subs[ok].view(-1, 2, 2)
    rem_sel = remaining.index_select(0, qi)                  # [M', G, 3]

    subs_b = subs.view(subs.shape[0], -1, 2)
    bodies_inst = apply_substitutions(b_pairs,  subs_b, pad)
    remain_inst = apply_substitutions(rem_sel, subs_b, pad)

    # Keep only active atoms
    lens_b = rule_lens_sorted.index_select(0, ri[ok])        # [M']
    take_b = (torch.arange(Bmax, device=device).unsqueeze(0) < lens_b.unsqueeze(1))
    take_g = (torch.arange(G,    device=device).unsqueeze(0) < remaining_counts.index_select(0, qi).unsqueeze(1))

    counts = take_b.sum(1) + take_g.sum(1)                   # [M']
    M = Bmax + G
    cat = torch.full((counts.shape[0], M, 3), pad, dtype=torch.long, device=device)
    # Move active body first, then active remaining
    if take_b.any():
        pos_b = torch.cumsum(take_b.long(), dim=1) - 1
        rows_b = torch.arange(take_b.shape[0], device=device).unsqueeze(1).expand_as(take_b)[take_b]
        cols_b = torch.arange(Bmax, device=device).unsqueeze(0).expand_as(take_b)[take_b]
        cat[rows_b, pos_b[take_b]] = bodies_inst[rows_b, cols_b]
    if take_g.any():
        pos_g = take_b.sum(1, keepdim=True) + (torch.cumsum(take_g.long(), dim=1) - 1)
        rows_g = torch.arange(take_g.shape[0], device=device).unsqueeze(1).expand_as(take_g)[take_g]
        cols_g = torch.arange(G, device=device).unsqueeze(0).expand_as(take_g)[take_g]
        cat[rows_g, pos_g[take_g]] = remain_inst[rows_g, cols_g]

    # Pack per owner
    K_per_owner = torch.bincount(qi, minlength=A)            # [A]
    if K_per_owner.max() == 0:
        return torch.full((A, 0, 1, 3), pad, dtype=torch.long, device=device), torch.zeros(A, dtype=torch.long, device=device)
    Kr = int(K_per_owner.max().item())
    out = torch.full((A, Kr, M, 3), pad, dtype=torch.long, device=device)

    owner_sorted, perm = torch.sort(qi, stable=True)
    cat_sorted = cat.index_select(0, perm)
    seg_start = torch.ones_like(owner_sorted, dtype=torch.bool, device=device)
    seg_start[1:] = owner_sorted[1:] != owner_sorted[:-1]
    seg_indices = torch.arange(owner_sorted.shape[0], device=device)
    seg_first   = torch.zeros_like(owner_sorted, dtype=torch.long)
    seg_first[seg_start] = seg_indices[seg_start]
    seg_first = torch.cummax(seg_first, dim=0)[0]
    pos_in_owner = seg_indices - seg_first
    out[owner_sorted, pos_in_owner] = cat_sorted
    return out, K_per_owner


@torch.no_grad()
def combine_candidates(
    rule_states: Tensor, rule_counts: Tensor,      # [A, Kr, Mr, 3], [A]
    fact_states: Tensor, fact_counts: Tensor,      # [A, Kf, Mf, 3], [A]
    active_idx: Tensor,                            # [A] -> owners in [0..B)
    padding_idx: int
) -> Tuple[Tensor, Tensor, Tensor, int]:
    """
    Merge rule and fact successors into a single flat candidate list.
    Returns:
      candidates: [N, M, 3]
      cand_counts: [N]
      owners: [N]  (values in 0..B-1)
      max_atoms_comb: int
    """
    device = rule_states.device
    pad = padding_idx
    A = active_idx.shape[0]
    if A == 0:
        z3 = torch.empty((0, 1, 3), dtype=rule_states.dtype, device=device)
        z1 = torch.empty((0,), dtype=torch.long, device=device)
        return z3, z1, z1, 1

    Kr = rule_states.shape[1] if rule_states.ndim == 4 else 0
    Kf = fact_states.shape[1] if fact_states.ndim == 4 else 0
    Mr = rule_states.shape[2] if rule_states.ndim == 4 else 1
    Mf = fact_states.shape[2] if fact_states.ndim == 4 else 1
    M  = max(Mr, Mf, 1)

    total_slots = Kr + Kf
    if total_slots == 0:
        z3 = torch.empty((0, M, 3), dtype=rule_states.dtype, device=device)
        z1 = torch.empty((0,), dtype=torch.long, device=device)
        return z3, z1, z1, M

    # Build [A, total_slots, M, 3] then pick valid via owner masks
    packed = torch.full((A, total_slots, M, 3), pad, dtype=rule_states.dtype, device=device)
    if Kr > 0:
        packed[:, :Kr, :Mr] = rule_states
    if Kf > 0:
        packed[:, Kr:Kr+Kf, :Mf] = fact_states

    owner_rule_mask = (torch.arange(Kr, device=device).unsqueeze(0) < rule_counts.unsqueeze(1)) if Kr > 0 else torch.zeros((A, 0), dtype=torch.bool, device=device)
    owner_fact_mask = (torch.arange(Kf, device=device).unsqueeze(0) < fact_counts.unsqueeze(1)) if Kf > 0 else torch.zeros((A, 0), dtype=torch.bool, device=device)
    owner_mask = torch.cat([owner_rule_mask, owner_fact_mask], dim=1)  # [A, total_slots]

    if not owner_mask.any():
        z3 = torch.empty((0, M, 3), dtype=rule_states.dtype, device=device)
        z1 = torch.empty((0,), dtype=torch.long, device=device)
        return z3, z1, z1, M

    rows = torch.arange(A, device=device).unsqueeze(1).expand_as(owner_mask)[owner_mask]
    cols = torch.arange(total_slots, device=device).unsqueeze(0).expand_as(owner_mask)[owner_mask]
    candidates = packed[rows, cols]                 # [N, M, 3]
    cand_counts = (candidates[:, :, 0] != pad).sum(dim=1)  # [N]
    owners = active_idx.index_select(0, rows)       # [N] -> 0..B-1
    return candidates, cand_counts, owners, M


@torch.no_grad()
def prune_and_collapse(
    candidates: Tensor,             # [N, M, 3]
    cand_counts: Tensor,            # [N]
    owners: Tensor,                 # [N] values in 0..B-1
    fact_index: GPUFactIndex,
    constant_no: int,
    padding_idx: int,
    B: int,
    excluded_first_atoms: Optional[Tensor] = None   # [N, 3] first atom of excluded query per candidate (owner-aligned)
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Drop ground atoms known to be facts, then collapse owners with proof.
    Proof criterion: after dropping, a candidate has zero atoms (all-True) ->
    that owner collapses to a single TRUE state and all its other candidates are discarded.

    Returns:
      survivors:    [N', M, 3]   (pruned candidates of non-proof owners)
      surv_counts:  [N']
      proof_mask_B: [B] bool mask of owners that are proofs
    """
    device = candidates.device
    pad = padding_idx

    if candidates.numel() == 0:
        return candidates, cand_counts, torch.zeros(B, dtype=torch.bool, device=device)

    N, M = candidates.shape[:2]
    valid = (candidates[:, :, 0] != pad)           # [N, M]
    args  = candidates[:, :, 1:3]
    is_ground = (args <= constant_no).all(dim=2)   # [N, M]
    ground = valid & is_ground

    # Ground fact membership
    drop = torch.zeros_like(valid)
    if ground.any():
        flat_mask = ground.flatten()
        atoms = candidates.reshape(-1, 3)[flat_mask]        # [G0, 3]
        is_fact = fact_index.contains(atoms)                # [G0]
        place = torch.zeros_like(flat_mask, dtype=torch.bool, device=device)
        place[flat_mask] = is_fact
        drop = place.view(N, M)

    # Do NOT drop atoms equal to excluded first atom (to avoid circular proofs)
    if excluded_first_atoms is not None and ground.any():
        excl = excluded_first_atoms.view(N, 1, 3).expand(-1, M, -1)  # [N, M, 3]
        keep_if_excl = ground & (candidates == excl).all(dim=2)
        drop &= ~keep_if_excl

    keep = valid & ~drop
    pruned_counts = keep.sum(dim=1)                           # [N]
    is_proof_cand = pruned_counts == 0                        # [N]

    # Collapse by owner
    proof_mask_B = torch.zeros(B, dtype=torch.bool, device=device)
    if is_proof_cand.any():
        proof_owners = torch.unique(owners[is_proof_cand])    # [<=B]
        proof_mask_B[proof_owners] = True

    # Keep only survivors from non-proof owners
    keep_mask = ~is_proof_cand & ~proof_mask_B.index_select(0, owners)
    if not keep_mask.any():
        return (torch.empty((0, M, 3), dtype=candidates.dtype, device=device),
                torch.empty((0,), dtype=torch.long, device=device),
                proof_mask_B)

    survivors = candidates[keep_mask]
    surv_keep = keep[keep_mask]
    pos = torch.cumsum(surv_keep.long(), dim=1) - 1           # [N', M]
    out = torch.full_like(survivors, pad)
    if surv_keep.any():
        ridx = torch.arange(survivors.shape[0], device=device).unsqueeze(1).expand_as(surv_keep)[surv_keep]
        cidx = torch.arange(M, device=device).unsqueeze(0).expand_as(surv_keep)[surv_keep]
        out[ridx, pos[surv_keep]] = survivors[ridx, cidx]
    surv_counts = surv_keep.sum(dim=1)
    return out, surv_counts, proof_mask_B


@torch.no_grad()
def standardize_derived_states(
    states: Tensor,                 # [N, M, 3]
    counts: Tensor,                 # [N]
    owners: Tensor,                 # [N] -> 0..B-1
    next_var_start_B: Tensor,       # [B]
    constant_no: int,
    runtime_var_end_index: int,
    padding_idx: int
) -> Tuple[Tensor, Tensor]:
    """
    Rename *new* runtime vars per state so that, for a given owner, *all* its
    derived states start numbering at the *same* base = next_var_start_B[owner].
    Existing vars (< base) are preserved.

    Returns:
      canon_states:     [N, M, 3]
      next_var_end_B:   [B] updated per-owner next var index (amax across survivors)
    """
    if states.numel() == 0:
        return states, next_var_start_B

    device = states.device
    pad = padding_idx
    N, M = states.shape[:2]
    args = states[:, :, 1:2+1]  # [:, :, 1:3]

    base_per_state = next_var_start_B.index_select(0, owners).view(N, 1, 1)  # [N,1,1]
    is_var = (args > constant_no) & (args != pad)
    is_new = is_var & (args >= base_per_state)
    if not is_new.any():
        return states, next_var_start_B

    # First-appearance order per state (visit atoms left->right, then arg0->arg1)
    lin = torch.arange(M * 2, device=device).view(1, M, 2).expand(N, -1, -1)
    # Unique key per occurrence (state id * big + var id), big must exceed any var id span
    SHIFT = runtime_var_end_index + 2
    st_id = torch.arange(N, device=device).view(N, 1, 1).expand_as(args)
    keys  = args + st_id * SHIFT  # [N, M, 2]
    occ_keys = keys[is_new]
    occ_lin  = lin[is_new]
    uniq, inv = torch.unique(occ_keys, return_inverse=True, sorted=True)  # per-state groups are intermixed but share state group via SHIFT

    # Rank groups per state by first occurrence
    BIG = M * 2 + 7
    key2 = inv.long() * BIG + occ_lin.long()
    order = torch.argsort(key2)
    inv_sorted = inv[order]
    lin_sorted = occ_lin[order]

    seg_start = torch.ones_like(inv_sorted, dtype=torch.bool)
    seg_start[1:] = inv_sorted[1:] != inv_sorted[:-1]
    first_lin = lin_sorted[seg_start]                  # first pos per group

    # Reconstruct state ids per group from uniq
    group_state = (uniq // SHIFT).long()               # [#groups]
    vars_per_state = torch.bincount(group_state, minlength=N)  # [N]

    # New id = base_per_state + rank-within-state
    # Compute ranks in the original (unsorted) order
    key3 = group_state * BIG + first_lin
    ord3 = torch.argsort(key3)
    sorted_state = group_state[ord3]
    bnd = torch.ones_like(sorted_state, dtype=torch.bool)
    bnd[1:] = sorted_state[1:] != sorted_state[:-1]
    seg_id = torch.cumsum(bnd.long(), dim=0) - 1
    if seg_id.numel() > 0:
        num_groups = int(seg_id.max().item()) + 1
        starts = torch.full((num_groups,), len(seg_id), dtype=torch.long, device=device)
        idxpos = torch.arange(len(seg_id), device=device)
        starts.scatter_reduce_(0, seg_id, idxpos, reduce='amin', include_self=False)
        rank_sorted = idxpos - starts[seg_id]
    else:
        rank_sorted = seg_id
    rank = torch.empty_like(rank_sorted)
    rank[ord3] = rank_sorted

    base_groups = base_per_state.view(-1).index_select(0, group_state)
    new_id_per_group = base_groups + rank
    if new_id_per_group.numel() > 0:
        overflow = new_id_per_group > runtime_var_end_index
        if overflow.any():
            raise RuntimeError("Variable renaming exceeded runtime budget; increase max_total_vars.")

    new_id_per_occ = new_id_per_group[inv]
    canon = states.clone()
    canon[:, :, 1:3][is_new] = new_id_per_occ.long()

    # Aggregate max next var end per owner
    next_end_per_state = base_per_state.view(-1) + vars_per_state
    next_end_B = next_var_start_B.clone()
    if next_end_per_state.numel() > 0:
        next_end_B.scatter_reduce_(0, owners, next_end_per_state, reduce='amax', include_self=False)
    return canon, next_end_B


@torch.no_grad()
def pack_by_owner(
    states: Tensor, counts: Tensor, owners: Tensor, B: int, M: int, padding_idx: int
) -> Tuple[Tensor, Tensor]:
    """
    Pack a flat candidate list into [B, K, M, 3] by owner.
    Returns (packed_states, packed_counts).
    """
    device = states.device
    pad = padding_idx
    if states.numel() == 0:
        return torch.full((B, 0, M, 3), pad, dtype=states.dtype, device=device), torch.zeros(B, dtype=torch.long, device=device)

    per_owner = torch.bincount(owners, minlength=B)   # [B]
    if per_owner.max() == 0:
        return torch.full((B, 0, M, 3), pad, dtype=states.dtype, device=device), torch.zeros(B, dtype=torch.long, device=device)

    K = int(per_owner.max().item())
    out = torch.full((B, K, M, 3), pad, dtype=states.dtype, device=device)
    counts_out = per_owner.clone()

    owner_sorted, perm = torch.sort(owners, stable=True)
    st_sorted  = states.index_select(0, perm)

    seg_start = torch.ones_like(owner_sorted, dtype=torch.bool, device=device)
    seg_start[1:] = owner_sorted[1:] != owner_sorted[:-1]
    seg_indices = torch.arange(owner_sorted.shape[0], device=device)
    seg_first   = torch.zeros_like(owner_sorted, dtype=torch.long)
    seg_first[seg_start] = seg_indices[seg_start]
    seg_first = torch.cummax(seg_first, dim=0)[0]
    pos_in_owner = seg_indices - seg_first

    out[owner_sorted, pos_in_owner] = st_sorted
    return out, counts_out


@torch.no_grad()
def cap_states_per_owner(
    states: Tensor, counts: Tensor, K_cap: int
) -> Tuple[Tensor, Tensor]:
    """
    Truncate the number of states per owner to at most K_cap, preserving order.
    states: [B, K, M, 3]
    counts: [B]
    """
    if states.numel() == 0 or K_cap <= 0:
        B, _, M, D = states.shape if states.numel() else (counts.shape[0], 0, 1, 3)
        return torch.empty((B, 0, M, D), dtype=states.dtype, device=states.device if states.numel() else counts.device), torch.zeros_like(counts)
    B, K, M, D = states.shape
    K_new = min(K_cap, K)
    states_cap = states[:, :K_new]
    counts_cap = torch.clamp(counts, max=K_new)
    return states_cap, counts_cap


# ============================================================================
# Engine
# ============================================================================

class UnificationEngine:
    """
    Encapsulates facts/rules and runs a single-step *modular* expansion.

    get_derived_states orchestrates the eight clean steps listed on top.
    """

    def __init__(self,
                 facts_idx: Tensor,              # [F, 3]
                 rules_idx: Tensor,              # [R, Bmax, 3]
                 rule_lens: Tensor,             # [R]
                 rules_heads_idx: Tensor,       # [R, 3]
                 padding_idx: int,
                 constant_no: int,
                 runtime_var_end_index: int,
                 true_pred_idx: Optional[int],
                 false_pred_idx: Optional[int],
                 max_arity: int,
                 predicate_range_map: Optional[Tensor],
                 device: torch.device,
                 pack_base: Optional[int] = None,
                 stringifier_params: Optional[dict] = None,
                 canonical_action_order: bool = False,
                 end_pred_idx: Optional[int] = None,
                 end_proof_action: bool = False):
        self.device = device
        self.padding_idx = int(padding_idx)
        self.constant_no = int(constant_no)
        self.runtime_var_end_index = int(runtime_var_end_index)
        self.max_arity = int(max_arity)
        self.true_pred_idx = true_pred_idx
        self.false_pred_idx = false_pred_idx
        self.end_pred_idx = end_pred_idx
        self.end_proof_action = bool(end_proof_action)
        self.canonical_action_order = bool(canonical_action_order)

        # Tensors
        self.facts_idx       = facts_idx.to(device=device, dtype=torch.long)
        self.rules_idx       = rules_idx.to(device=device, dtype=torch.long)
        self.rule_lens       = rule_lens.to(device=device, dtype=torch.long)
        self.rules_heads_idx = rules_heads_idx.to(device=device, dtype=torch.long)

        # Derived sizes
        self.max_rule_body_size = int(rule_lens.max().item()) if rule_lens.numel() > 0 else 1

        # Pack base (>= any index + 1)
        max_idx = 1
        for t in (self.facts_idx, self.rules_idx, self.rules_heads_idx):
            if t.numel() > 0:
                mx = int(t.max().item())
                if mx > max_idx:
                    max_idx = mx
        self.pack_base = int(pack_base if pack_base is not None else (max(max_idx, self.runtime_var_end_index) + 2))

        # Facts index
        self.fact_index = GPUFactIndex(self.facts_idx, self.pack_base)
        
        # Store packed facts for compatibility
        if self.facts_idx.numel() > 0:
            self.facts_packed = _pack_triples_64(self.facts_idx.long(), self.pack_base)
        else:
            self.facts_packed = torch.empty((0,), dtype=torch.long, device=device)

        # Pre-sort rules by predicate and build heads predicate ranges
        if self.rules_heads_idx.numel() > 0:
            order = torch.argsort(self.rules_heads_idx[:, 0])
            self.rules_heads_sorted  = self.rules_heads_idx.index_select(0, order)
            self.rules_idx_sorted    = self.rules_idx.index_select(0, order)
            self.rule_lens_sorted    = self.rule_lens.index_select(0, order)

            preds = self.rules_heads_sorted[:, 0]
            uniq, counts = torch.unique_consecutive(preds, return_counts=True)
            starts = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.long, device=device), counts[:-1]]), dim=0)
            num_pred = int(preds.max().item()) + 2
            self.rule_seg_starts = torch.zeros((num_pred,), dtype=torch.long, device=device)
            self.rule_seg_lens   = torch.zeros((num_pred,), dtype=torch.long, device=device)
            self.rule_seg_starts[uniq] = starts
            self.rule_seg_lens[uniq]   = counts
        else:
            self.rules_heads_sorted = self.rules_heads_idx
            self.rules_idx_sorted   = self.rules_idx
            self.rule_lens_sorted   = self.rule_lens
            self.rule_seg_starts = torch.zeros((1,), dtype=torch.long, device=device)
            self.rule_seg_lens   = torch.zeros((1,), dtype=torch.long, device=device)

        # Optional fact predicate map for fast fact pairing
        self.predicate_range_map = predicate_range_map.to(device=device, dtype=torch.long) if (predicate_range_map is not None and predicate_range_map.numel() > 0) else None

        # Hash cache for dedup
        self.hash_cache = GPUHashCache(device)

        # Canonical TRUE/FALSE atoms (1x3) and tensors for compatibility
        pad = self.padding_idx
        self.true_atom  = torch.tensor([self.true_pred_idx,  pad, pad], dtype=torch.long, device=device) if self.true_pred_idx  is not None else None
        self.false_atom = torch.tensor([self.false_pred_idx, pad, pad], dtype=torch.long, device=device) if self.false_pred_idx is not None else None
        
        # Store as tensors with shape [1, 3] for compatibility
        self.true_tensor = torch.tensor([[self.true_pred_idx, pad, pad]], dtype=torch.long, device=device) if self.true_pred_idx is not None else None
        self.false_tensor = torch.tensor([[self.false_pred_idx, pad, pad]], dtype=torch.long, device=device) if self.false_pred_idx is not None else None
        if self.end_pred_idx is not None:
            self.end_tensor = torch.tensor([[self.end_pred_idx, pad, pad]], dtype=torch.long, device=device)
        else:
            self.end_tensor = None
        
        # Initialize DebugHelper for verbose output
        from debug_helper import DebugHelper as DH
        self.debug_helper = DH(**stringifier_params) if stringifier_params else None

    # ---- factory ----
    @classmethod
    def from_index_manager(cls, im, take_ownership: bool = False, stringifier_params: Optional[dict] = None,
                           canonical_action_order: bool = False, end_pred_idx: Optional[int] = None,
                           end_proof_action: bool = False) -> 'UnificationEngine':
        engine = cls(
            facts_idx=getattr(im, 'facts_idx', None),
            rules_idx=getattr(im, 'rules_idx', None),
            rule_lens=getattr(im, 'rule_lens', None),
            rules_heads_idx=getattr(im, 'rules_heads_idx', None),
            padding_idx=im.padding_idx,
            constant_no=im.constant_no,
            runtime_var_end_index=im.runtime_var_end_index,
            true_pred_idx=im.true_pred_idx,
            false_pred_idx=im.false_pred_idx,
            max_arity=im.max_arity,
            predicate_range_map=getattr(im, 'predicate_range_map', None),
            device=im.device,
            pack_base=getattr(im, 'total_vocab_size', None),
            stringifier_params=stringifier_params,
            canonical_action_order=canonical_action_order,
            end_pred_idx=end_pred_idx,
            end_proof_action=end_proof_action
        )
        if take_ownership:
            im.facts_idx = None
            im.rules_idx = None
            im.rule_lens = None
            im.rules_heads_idx = None
        return engine

    # ---------------------------------------------------------------------
    #  Orchestration (as clean as possible)
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def get_derived_states(self,
                           current_states: Tensor,         # [B, max_atoms, 3]
                           next_var_indices: Tensor,       # [B]
                           excluded_queries: Optional[Tensor] = None,  # [B, max_atoms, 3]
                           max_derived_per_state: int = 500,
                           deduplicate: bool = True,
                           verbose: int = 0
                           ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
          derived_states: [B, K, M, 3]  (padded to K=max_derived_per_state if needed)
          derived_counts: [B]
          updated_next_var_indices: [B]
        """
        device = current_states.device
        B, max_atoms = current_states.shape[:2]
        pad = self.padding_idx
        
        if verbose > 0 and self.debug_helper:
            print(f"\n[ENGINE DEBUG] get_derived_states called with next_var_indices={next_var_indices.tolist()}, constant_no={self.constant_no}")

        # Preallocate final output (padded). We'll fill progressively.
        max_atoms_out = max(max_atoms + self.max_rule_body_size, 1)
        final_states  = torch.full((B, max_derived_per_state, max_atoms_out, 3), pad, dtype=torch.long, device=device)
        final_counts  = torch.zeros(B, dtype=torch.long, device=device)
        updated_next  = next_var_indices.clone()

        # ---- 1) preprocessing
        pre = preprocess_states(current_states, self.true_pred_idx, self.false_pred_idx, pad)
        if verbose > 0 and self.debug_helper:
            self.debug_helper.print_states("[ENGINE] 1. CURRENT STATES", current_states)

        # Handle terminal TRUE
        if self.true_atom is not None and pre.terminal_true.numel() > 0:
            dst = pre.terminal_true
            final_states[dst, 0, 0] = self.true_atom
            final_counts[dst] = 1

        # Handle terminal FALSE or empty
        if self.false_atom is not None and pre.terminal_false.numel() > 0:
            dst = pre.terminal_false
            final_states[dst, 0, 0] = self.false_atom
            final_counts[dst] = 1

        # If no active states remain, we are done.
        if pre.active_idx.numel() == 0:
            return final_states, final_counts, updated_next

        # Convenient aliases
        A = pre.active_idx.numel()
        next_var_active = updated_next.index_select(0, pre.active_idx)

        # ---- 2) rule unification (pure; no pruning)
        rule_states, rule_counts = unify_with_rules(
            self.rules_heads_sorted, self.rules_idx_sorted, self.rule_lens_sorted,
            self.rule_seg_starts, self.rule_seg_lens,
            pre.queries, pre.remaining, pre.remaining_counts, pre.preds,
            self.constant_no, pad, next_var_active
        )
        
        if verbose > 0 and self.debug_helper:
            self.debug_helper.print_states("[ENGINE] 2. RULE UNIFICATIONS", rule_states, rule_counts)

        # ---- 3) fact unification (pure; no pruning)
        facts_excl = excluded_queries.index_select(0, pre.active_idx) if (excluded_queries is not None) else None
        fact_states, fact_counts = unify_with_facts(
            self.facts_idx, self.predicate_range_map,
            pre.queries, pre.remaining, pre.remaining_counts, pre.preds,
            self.constant_no, pad, self.fact_index,
            excluded_queries=facts_excl
        )
        
        if verbose > 0 and self.debug_helper:
            self.debug_helper.print_states("[ENGINE] 3. FACT UNIFICATIONS", fact_states, fact_counts)

        # ---- 4) combine rule+fact into flat candidates
        cand_states, cand_counts, owners, M_comb = combine_candidates(
            rule_states, rule_counts, fact_states, fact_counts, pre.active_idx, pad
        )

        # If none, mark these actives as FALSE (no outgoing edges)
        if cand_states.numel() == 0:
            if self.false_atom is not None:
                final_states[pre.active_idx, 0, 0] = self.false_atom
                final_counts[pre.active_idx] = 1
            return final_states, final_counts, updated_next

        # Prepare excluded first atoms aligned to candidates (for circular-proof guard)
        excl_first = None
        if facts_excl is not None and facts_excl.numel() > 0:
            # Map owner -> its first excluded atom; then broadcast to candidates by owners
            first_atom_per_owner = facts_excl[:, 0, :]                      # [A, 3]
            # owners is in 0..B-1; we need local mapping active->B for A-sized table:
            # Build a B-sized table with padding; then index by owners
            table = torch.full((B, 3), pad, dtype=torch.long, device=device)
            table[pre.active_idx] = first_atom_per_owner
            excl_first = table.index_select(0, owners)                      # [N, 3]

        # ---- 5) prune ground facts and collapse proof owners to TRUE (only here!)
        surv_states, surv_counts, proof_mask_B = prune_and_collapse(
            cand_states, cand_counts, owners, self.fact_index, self.constant_no, pad, B, excl_first
        )

        # Write TRUE for proof owners
        if self.true_atom is not None and proof_mask_B.any():
            dst = torch.nonzero(proof_mask_B, as_tuple=False).view(-1)
            final_states[dst, 0, 0] = self.true_atom
            final_counts[dst] = 1

        # If all active owners are proofs, we are done.
        nonproof_mask_B = torch.zeros(B, dtype=torch.bool, device=device)
        nonproof_mask_B[owners] = True
        nonproof_mask_B &= ~proof_mask_B
        if not nonproof_mask_B.any():
            return final_states, final_counts, updated_next

        # Filter survivors to only non-proof owners
        if surv_states.numel() == 0:
            # No survivors for non-proof owners -> FALSE
            if self.false_atom is not None:
                nonproof_idx = torch.nonzero(nonproof_mask_B, as_tuple=False).view(-1)
                final_states[nonproof_idx, 0, 0] = self.false_atom
                final_counts[nonproof_idx] = 1
            return final_states, final_counts, updated_next

        surv_owners = owners[ (surv_counts >= 0) & (~proof_mask_B.index_select(0, owners)) ]  # match survivors' owners

        # ---- 6) standardize variables (shared base per owner) and update next_var per owner
        if verbose > 0 and self.debug_helper:
            print(f"[ENGINE] Before standardize: next_var indices = {updated_next.tolist() if updated_next.numel() <= 10 else updated_next[:10].tolist()}")
            print(f"  constant_no = {self.constant_no}, runtime_var_end = {self.runtime_var_end_index}")
            self.debug_helper.print_states("[ENGINE] 5. BEFORE STANDARDIZE", surv_states, surv_counts)
        
        std_states, next_end_B = standardize_derived_states(
            surv_states, surv_counts, surv_owners, updated_next, self.constant_no,
            self.runtime_var_end_index, pad
        )
        updated_next = torch.maximum(updated_next, next_end_B)
        
        if verbose > 0 and self.debug_helper:
            self.debug_helper.print_states("[ENGINE] 6. AFTER STANDARDIZE", std_states, surv_counts)

        # ---- 7) pack by owner -> (optional) dedup -> cap K
        packed, packed_counts = pack_by_owner(std_states, surv_counts, surv_owners, B, M_comb, pad)

        if deduplicate:
            packed, packed_counts = deduplicate_states_packed(packed, packed_counts, pad, self.hash_cache)

        packed, packed_counts = cap_states_per_owner(packed, packed_counts, max_derived_per_state)

        # Write packed into final buffers (keeping any terminal/proof already written)
        write_mask = packed_counts > 0
        if write_mask.any():
            dst = torch.nonzero(write_mask, as_tuple=False).view(-1)
            # Avoid overwriting TRUE already set for proof owners
            dst = dst[~proof_mask_B.index_select(0, dst)]
            if dst.numel() > 0:
                final_states[dst, :packed.shape[1], :packed.shape[2]] = packed[dst]
                final_counts[dst] = torch.maximum(final_counts[dst], packed_counts[dst])

        # ---- Fallback: any active owner with no derived state yet -> FALSE
        need_false = (final_counts == 0) & (torch.isin(torch.arange(B, device=device), pre.active_idx))
        need_false &= ~proof_mask_B
        if need_false.any() and self.false_atom is not None:
            idx = torch.nonzero(need_false, as_tuple=False).view(-1)
            final_states[idx, 0, 0] = self.false_atom
            final_counts[idx] = 1

        return final_states, final_counts, updated_next

    # ---- Utility methods for compatibility ----
    def canonicalize_state(self, state: Tensor) -> Tensor:
        """Canonicalize a single tensor state (drop batch dim if present)."""
        if self.debug_helper is None:
            raise RuntimeError("debug_helper not initialized. Pass stringifier_params to __init__.")
        return self.debug_helper.canonicalize_state(state, self.constant_no, self.padding_idx)

    def canonical_state_to_str(self, state: Tensor) -> str:
        """Canonicalize tensor state and convert to string for comparisons."""
        if self.debug_helper is None:
            raise RuntimeError("debug_helper not initialized. Pass stringifier_params to __init__.")
        return self.debug_helper.canonical_state_to_str(state, self.constant_no, self.padding_idx)

    def is_true_state(self, state: Tensor) -> bool:
        """
        Check if state contains only the True predicate (with padding).
        
        Args:
            state: [max_atoms, 3] tensor
            
        Returns:
            True if state has exactly 1 non-padding atom and it equals True predicate
        """
        if self.true_tensor is None:
            return False
        # Count non-padding atoms
        non_padding = (state[:, 0] != self.padding_idx).sum()
        if non_padding != 1:
            return False
        # Compare first atom only
        return torch.equal(state[0], self.true_tensor.squeeze(0))

    def is_false_state(self, state: Tensor) -> bool:
        """
        Check if state contains only the False predicate (with padding).
        
        Args:
            state: [max_atoms, 3] tensor
            
        Returns:
            True if state has exactly 1 non-padding atom and it equals False predicate
        """
        if self.false_tensor is None:
            return False
        # Count non-padding atoms
        non_padding = (state[:, 0] != self.padding_idx).sum()
        if non_padding != 1:
            return False
        # Compare first atom only
        return torch.equal(state[0], self.false_tensor.squeeze(0))

    def get_false_state(self) -> Tensor:
        """Return a clone of the false tensor."""
        if self.false_tensor is None:
            raise ValueError("False predicate not defined.")
        return self.false_tensor.clone()

    def get_true_state(self) -> Tensor:
        """Return a clone of the true tensor."""
        if self.true_tensor is None:
            raise ValueError("True predicate not defined.")
        return self.true_tensor.clone()

    def get_end_state(self) -> Tensor:
        """Get a state containing only the END predicate."""
        if self.end_tensor is None:
            raise ValueError("End predicate not defined.")
        return self.end_tensor.clone()

    def is_terminal_pred(self, pred_indices: Tensor) -> Tensor:
        """
        Check if predicate indices correspond to terminal predicates (TRUE/FALSE/END).
        
        Args:
            pred_indices: Tensor of predicate indices, shape [N]
            
        Returns:
            Boolean tensor of shape [N] indicating which predicates are terminal
        """
        is_terminal = torch.zeros_like(pred_indices, dtype=torch.bool)
        
        if self.true_pred_idx is not None:
            is_terminal |= (pred_indices == self.true_pred_idx)
        
        if self.false_pred_idx is not None:
            is_terminal |= (pred_indices == self.false_pred_idx)
        
        if self.end_proof_action and self.end_pred_idx is not None:
            is_terminal |= (pred_indices == self.end_pred_idx)
        
        return is_terminal
