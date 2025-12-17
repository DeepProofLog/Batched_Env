"""
Modular Tensor-Based Unification Engine for Logical Reasoning.

This module implements a fully batched, GPU-accelerated symbolic unification engine
designed for high-throughput reasoning in large knowledge graphs. It orchestrates
the backward chaining process, generating derived states (subgoals) from current
states and a set of logic rules.

Algorithm Overview:
-------------------
The unification process transforms a batch of "current states" (conjunctions of atoms)
into a new batch of "derived states" (successors) through the following pipeline:

1.  **Preprocessing**:
    - Separates active non-terminal states from terminal states (True/False).
    - Extracts the first atom (leftmost subgoal) from each active state as the "query".

2.  **Unification (Fact & Rule)**:
    - **Fact Unification**: Matches queries against the Knowledge Graph (facts).
      - Ground queries: fast O(1) hash lookup.
      - Non-ground queries: predicate-range bounded search.
    - **Rule Unification**: Matches queries against rule heads.
      - Renames rule variables to unique runtime variables.
      - Unifies query args with rule head args to generate substitutions.
    - Both steps produce "intermediate states" (ground facts or rule bodies with substitutions applied).

3.  **Combination**:
    - Merges successful matches from rules and facts into a single list of candidates.

4.  **Pruning & Proof Detection**:
    - **Grounding**: Identifies atoms in candidates that are already known facts.
    - **Pruning**: Removes these known facts from the conjunction.
    - **Proof Check**: If a candidate becomes empty (all atoms proven), the owner state is proven TRUE.
    - **Collapse**: If an owner is proven, all other candidates for that owner are discarded.

5.  **Standardization**:
    - Renormalizes variable indices in the surviving states to ensure a canonical representation.
    - Maps variable IDs to a contiguous range starting from `next_var_indices`.

6.  **Packing**:
    - Groups derived states by their original owner index.
    - Pads the output to a fixed maximum branching factor `K` per owner.

Tensor Shape Conventions:
-------------------------
Dimensions:
    B:       Batch size (number of parallel environments/proofs)
    A:       Number of *active* states (subset of B not yet terminal)
    N:       Total number of candidate derived states (across all active owners)
    F:       Number of facts in the Knowledge Graph
    R:       Number of rules
    P:       Number of predicates
    Kr / Kf: Number of rule/fact successors per active state
    Mr / Mf: Number of atoms per rule/fact derived state
    M:       Maximum atoms per derived state
    G:       Number of atoms in the *remaining* part of the parent state
    D:       Atom dimension (always 3: [predicate, arg0, arg1])

Variables:
    padding_idx: Value used to pad variable-length sequences.
    constant_no: Threshold for constants vs variables (<= constant_no are constants).

Performance Notes:
    - All operations are vectorized on GPU.
    - No Python loops over batch elements.
    - Uses 64-bit packed hashes for fast set membership.
    - Relies on sorted indices and `searchsorted` for efficient joins.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch
from torch import Tensor
import tensor.tensor_utils as utils_funcs

# Global flag for parity mode - when True, uses deterministic behavior for tests
# When False, uses optimized behavior without synchronization points
PARITY_MODE = False

# Global flag for torch.compile - when True, compiles hot path functions
COMPILE_MODE = False

def set_parity_mode(enabled: bool) -> None:
    """Enable or disable parity mode globally."""
    global PARITY_MODE
    PARITY_MODE = enabled

def get_parity_mode() -> bool:
    """Check if parity mode is enabled."""
    return PARITY_MODE

def set_compile_mode(enabled: bool) -> None:
    """Enable or disable torch.compile for hot paths."""
    global COMPILE_MODE
    COMPILE_MODE = enabled

def get_compile_mode() -> bool:
    """Check if compile mode is enabled."""
    return COMPILE_MODE

# Cache for compiled functions - avoids recompiling on each call
_compiled_cache = {}

# Cache for arange tensors - reduces repeated allocations
class ArangeCache:
    """
    Cache for pre-computed torch.arange tensors.
    
    Many unification functions call torch.arange repeatedly with the same size.
    This cache avoids repeated GPU memory allocations.
    """
    def __init__(self, max_cached_size: int = 4096):
        self._cache: Dict[Tuple[int, torch.device], Tensor] = {}
        self._max_size = max_cached_size
    
    def get(self, size: int, device: torch.device) -> Tensor:
        """Get an arange tensor of the given size, using cache if available."""
        if size > self._max_size or size <= 0:
            return torch.arange(size, device=device, dtype=torch.long)
        
        key = (size, device)
        if key not in self._cache:
            self._cache[key] = torch.arange(size, device=device, dtype=torch.long)
        return self._cache[key]
    
    def clear(self):
        """Clear the cache."""
        self._cache.clear()

# Global arange cache instance
_arange_cache = ArangeCache()

def maybe_compile(fn, name: str, dynamic: bool = True):
    """
    Conditionally compile a function with torch.compile when COMPILE_MODE is True.
    Uses caching to avoid recompilation.
    
    Args:
        fn: Function to compile
        name: Cache key for the compiled function
        dynamic: If True, allows dynamic shapes (for varying batch sizes)
    """
    if not COMPILE_MODE:
        return fn
    cache_key = f"{name}_dyn{dynamic}"
    if cache_key not in _compiled_cache:
        # Use 'default' mode instead of 'reduce-overhead' to avoid CUDA graph 
        # memory issues with dynamic tensor shapes in unification functions
        _compiled_cache[cache_key] = torch.compile(
            fn, 
            mode='default',  # 'reduce-overhead' uses CUDA graphs which conflict with dynamic tensors
            fullgraph=False,
            dynamic=dynamic
        )
    return _compiled_cache[cache_key]


def compile_if_enabled(fn):
    """
    Decorator to conditionally compile a function when COMPILE_MODE is True.
    Uses function name as cache key.
    """
    def wrapper(*args, **kwargs):
        compiled_fn = maybe_compile(fn, fn.__name__, dynamic=True)
        return compiled_fn(*args, **kwargs)
    return wrapper


# ============================================================================
# Compiled Inner Kernels
# ============================================================================

def _substitution_kernel_impl(args: Tensor, subs_pairs: Tensor, padding_idx: int) -> Tensor:
    """Core substitution logic - compilable with torch.compile."""
    N, M = args.shape[0], args.shape[1]
    S = subs_pairs.shape[1]
    
    frm = subs_pairs[:, :, 0].view(N, S, 1, 1)
    to_ = subs_pairs[:, :, 1].view(N, S, 1, 1)
    valid = subs_pairs[..., 0] != padding_idx
    valid_exp = valid.view(N, S, 1, 1)
    args_exp = args.view(N, 1, M, 2)
    
    match = (args_exp == frm) & valid_exp
    any_match = match.any(dim=1)
    match_idx = match.long().argmax(dim=1)
    
    to_flat = subs_pairs[:, :, 1]
    to_gathered = to_flat.gather(1, match_idx.view(N, -1)).view(N, M, 2)
    
    return torch.where(any_match, to_gathered, args)


def _variable_binding_kernel_impl(q_args: Tensor, t_args: Tensor, var_start: int, padding_idx: int) -> Tensor:
    """Core variable binding logic for unification - compilable with torch.compile."""
    L = q_args.shape[0]
    subs = torch.full((L, 2, 2), padding_idx, dtype=torch.long, device=q_args.device)
    
    qv = (q_args >= var_start) & (q_args != padding_idx)
    tv = (t_args >= var_start) & (t_args != padding_idx)
    
    # Case A: Query var, Term const -> bind query to term
    case1 = qv & (~tv) & (t_args != 0)
    subs[:, :, 0] = torch.where(case1, q_args.long(), subs[:, :, 0])
    subs[:, :, 1] = torch.where(case1, t_args.long(), subs[:, :, 1])
    
    # Case B: Term var, Query const -> bind term to query
    case2 = (~qv) & (q_args != 0) & tv
    subs[:, :, 0] = torch.where(case2, t_args.long(), subs[:, :, 0])
    subs[:, :, 1] = torch.where(case2, q_args.long(), subs[:, :, 1])
    
    # Case C: Both vars -> bind term to query
    case3 = qv & tv
    subs[:, :, 0] = torch.where(case3, t_args.long(), subs[:, :, 0])
    subs[:, :, 1] = torch.where(case3, q_args.long(), subs[:, :, 1])
    
    return subs


def _substitution_kernel(args: Tensor, subs_pairs: Tensor, padding_idx: int) -> Tensor:
    """Wrapper that uses compiled version when COMPILE_MODE is enabled."""
    return maybe_compile(_substitution_kernel_impl, 'substitution_kernel')(args, subs_pairs, padding_idx)


def _variable_binding_kernel(q_args: Tensor, t_args: Tensor, var_start: int, padding_idx: int) -> Tensor:
    """Wrapper that uses compiled version when COMPILE_MODE is enabled."""
    return maybe_compile(_variable_binding_kernel_impl, 'variable_binding_kernel')(q_args, t_args, var_start, padding_idx)


# ============================================================================
# Small helpers
# ============================================================================

# Cache for pack base tensors - avoids repeated torch.as_tensor allocations
_pack_base_cache: dict = {}

@torch.no_grad()
def _pack_triples_64(atoms: Tensor, base: int) -> Tensor:
    """
    Pack triples [pred, a, b] into single 64-bit integers for efficient set operations.
    
    This function creates a unique 64-bit fingerprint for each atom (triple) to enable
    fast O(1) equality checks and sorting without comparing the 3 components separately.
    
    Formula: ((pred * base) + arg0) * base + arg1

    Args:
        atoms: Tensor of shape [N, 3] usually containing (predicate, arg0, arg1)
        base:  Integer base for packing (must be >= max_entity_index + 1)
        
    Returns:
        Tensor of shape [N] (int64) containing the packed hash values.
    """
    if atoms.numel() == 0:
        return torch.empty(0, dtype=torch.int64, device=atoms.device)
    
    # Extract components: [N] each
    p, a, b = atoms[:, 0].long(), atoms[:, 1].long(), atoms[:, 2].long()
    
    # Cache the base tensor per (base, device) to avoid repeated allocations
    cache_key = (base, atoms.device)
    if cache_key not in _pack_base_cache:
        _pack_base_cache[cache_key] = torch.tensor(base, dtype=torch.int64, device=atoms.device)
    base_t = _pack_base_cache[cache_key]
    
    # Pack: [N]
    return ((p * base_t) + a) * base_t + b


class GPUHashCache:
    """
    Cache for precomputed polynomial hash powers used in state deduplication.
    
    Deduplication requires computing a hash of variable-length state sequences.
    To avoid recomputing powers of the prime base for every batch, this class
    caches them on the GPU.
    
    Attributes:
        prime (int): Prime base for the polynomial rolling hash (31).
        mod_val (int): Large Mersenne prime (2^61 - 1) to minimize collisions.
        prime_powers (Tensor): Cached powers [prime^0, prime^1, ..., prime^max_len].
    """
    
    def __init__(self, device: torch.device, max_len: int = 4096):
        """
        Initialize the hash cache.
        
        Args:
            device: Tensor device (CPU/CUDA)
            max_len: Initial maximum sequence length to support.
        """
        self.device = device
        self.prime = 31
        self.mod_val = 2**61 - 1
        self._build_cache(max_len)

    def _build_cache(self, max_len: int):
        """Compute and store powers up to max_len."""
        powers = torch.arange(max_len, device=self.device, dtype=torch.int64)
        self.prime_powers = torch.pow(
            torch.tensor(self.prime, device=self.device, dtype=torch.int64), 
            powers
        ) % self.mod_val
        self.max_len = max_len

    def get_powers(self, length: int) -> Tensor:
        """
        Retrieve powers for a sequence of the given length.
        Resizes the cache if the requested length exceeds current capacity.
        
        Args:
            length: Required sequence length.
        
        Returns:
            Tensor of shape [length] containing powers.
        """
        if length > self.max_len:
            self._build_cache(length)
        return self.prime_powers[:length]


class GPUFactIndex:
    """
    Efficient GPU-based index for fast fact membership testing.
    
    This class stores the entire Knowledge Graph (set of facts) as a sorted list
    of 64-bit packed integers. Membership testing is performed using `searchsorted`
    (binary search), enabling extremely fast lookups for large batches of queries.
    
    Attributes:
        fact_hashes (Tensor): Sorted tensor of packed fact hashes. Shape [F].
        pack_base (int): The base used for packing triples.
    """
    
    def __init__(self, facts: Tensor, pack_base: int):
        """
        Build the index from a tensor of facts.
        
        Args:
            facts: Tensor of shape [F, 3] containing (pred, arg0, arg1).
            pack_base: Base for packing (must match inference time usage).
        """
        self.device = facts.device
        self.pack_base = int(pack_base)
        
        if facts.numel() == 0:
            self.fact_hashes = torch.empty(0, dtype=torch.int64, device=self.device)
        else:
            # Sort facts for binary search capability
            self.fact_hashes = _pack_triples_64(facts.long(), self.pack_base).sort()[0]

    @torch.no_grad()
    def contains(self, atoms: Tensor) -> Tensor:
        """
        Check which atoms in the input batch exist in the fact index.
        
        Args:
            atoms: Tensor of shape [N, 3] containing query atoms.
            
        Returns:
            Boolean Tensor of shape [N] where True indicates the atom exists in facts.
        """
        if atoms.numel() == 0 or self.fact_hashes.numel() == 0:
            return torch.zeros((atoms.shape[0],), dtype=torch.bool, device=atoms.device)
            
        # Pack query atoms: [N]
        keys = _pack_triples_64(atoms.long(), self.pack_base)
        
        # Binary search for potentially matching indices: [N]
        idx  = torch.searchsorted(self.fact_hashes, keys)
        
        # Verify matches (handle out-of-bounds indices)
        valid = idx < self.fact_hashes.shape[0]
        
        # Result mask: [N]
        mask = torch.zeros_like(keys, dtype=torch.bool)
        mask[valid] = (self.fact_hashes[idx[valid]] == keys[valid])
        
        return mask


# ============================================================================
# Deduplication (per-owner, on packed [B, K, M, 3])
# ============================================================================

@torch.no_grad()
def deduplicate_states_packed(
    states: Tensor,
    counts: Tensor,
    padding_idx: int,
    hash_cache: Optional[GPUHashCache] = None
) -> Tuple[Tensor, Tensor]:
    """
    Perform stateless, per-owner deduplication of derived states using hash-based sorting.
    
    This function eliminates identical candidate states (conjunctions of atoms) generated
    for the same owner. It relies on a polynomial rolling hash to fingerprint each state,
    followed by sorting and checking adjacent elements.
    
    Args:
        states:      Tensor of shape [B, K, M, 3] containing candidate states.
        counts:      Tensor of shape [B] containing valid candidates per owner.
        padding_idx: Padding value used in states.
        hash_cache:  Optional precomputed power cache for hashing.
        
    Returns:
        Tuple[Tensor, Tensor]:
            - Deduped states of shape [B, K', M, 3] (where K' <= K)
            - New counts of shape [B]
    """
    if states.numel() == 0:
        return states, counts

    B, K, M, D = states.shape
    device = states.device

    # -------------------------------------------------------------------------
    # 1. Compute Hash Per Candidate
    # -------------------------------------------------------------------------
    # Flatten each state to [B, K, L] where L = M * 3
    flat = states.reshape(B, K, -1).long()

    # Create mask for valid candidates: [B, K]
    valid_mask = (torch.arange(K, device=device).unsqueeze(0) < counts.unsqueeze(1))

    # Polynomial hash parameters
    prime = 31
    mod_val = 2**61 - 1
    L = M * D
    
    if hash_cache is None:
        powers = torch.arange(L, device=device, dtype=torch.int64)
        prime_powers = torch.pow(torch.tensor(prime, device=device, dtype=torch.int64), powers) % mod_val
    else:
        prime_powers = hash_cache.get_powers(L)

    # Compute dot product hash: sum(val[i] * p^i) % mod
    # hashes: [B, K]
    hashes = (flat * prime_powers.view(1, 1, -1)).sum(dim=2) % mod_val
    
    # Set hashes of invalid entries to a unique value (mod_val) to push them to end during sort
    hashes = torch.where(valid_mask, hashes, torch.full_like(hashes, mod_val))

    # -------------------------------------------------------------------------
    # 2. Sort and Identify Uniques
    # -------------------------------------------------------------------------
    # Sort hashes to bring identical states together: [B, K]
    sorted_hashes, sort_idx = torch.sort(hashes, dim=1)
    
    # Reorder states according to hash order: [B, K, M, 3]
    # We expand sort_idx to match state dims
    gather_idx = sort_idx.view(B, K, 1, 1).expand(B, K, M, D)
    sorted_states = torch.gather(states, 1, gather_idx)

    # Identify unique elements by comparing with neighbors
    unique_mask = torch.ones((B, K), dtype=torch.bool, device=device)
    if K > 1:
        # Check hash collisions first (fast)
        hash_diff = sorted_hashes[:, 1:] != sorted_hashes[:, :-1]  # [B, K-1]
        same_hash = ~hash_diff
        
        if same_hash.any():
            # Full state comparison only where hashes match
            # eq: [B, K-1] True if state[i] == state[i-1]
            eq = (sorted_states[:, 1:] == sorted_states[:, :-1]).all(dim=(2, 3))
            unique_mask[:, 1:] = hash_diff | ~eq
        else:
            unique_mask[:, 1:] = hash_diff

    # Always mask out original invalid entries (pushed to end by sort)
    # We compare sort_idx (original index) with count to see if it was valid
    unique_mask &= (sort_idx < counts.unsqueeze(1))

    # -------------------------------------------------------------------------
    # 3. Scatter Back to Compact Tensor
    # -------------------------------------------------------------------------
    uniq_counts = unique_mask.sum(dim=1)  # [B]
    
    if uniq_counts.max() == 0:
        return torch.full((B, 0, M, D), padding_idx, dtype=states.dtype, device=device), uniq_counts

    # Calculate aggregate positions for scatter
    # pos: [B, K] - destination index for each element (0..K'-1)
    pos = torch.cumsum(unique_mask.long(), dim=1) - 1
    
    outK = int(K)  # Keep distinct count upper bound consistent with input K logic
    out = torch.full((B, outK, M, D), padding_idx, dtype=states.dtype, device=device)

    # Vectorized scatter: select valid elements and place them
    b_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(pos)[unique_mask]
    u_idx = pos[unique_mask]
    
    # We must select from sorted_states because unique_mask aligns with sorted order
    # Selection shape: [Total_Unique, M, D]
    vals = sorted_states[unique_mask.unsqueeze(-1).unsqueeze(-1).expand_as(sorted_states)].reshape(-1, M, D)
    
    out[b_idx, u_idx] = vals
    return out, uniq_counts


# ============================================================================
# Substitution / pairwise unify helpers
# ============================================================================

@torch.no_grad()
def apply_substitutions(goals: Tensor, subs_pairs: Tensor, padding_idx: int) -> Tensor:
    """
    Apply variable substitutions to goal atoms (optimized for S=2).
    """
    if goals.numel() == 0:
        return goals
    
    N, M = goals.shape[:2]
    S = subs_pairs.shape[1]
    pad = padding_idx
    
    preds = goals[:, :, 0:1]
    args = goals[:, :, 1:]
    
    # OPTIMIZATION: Loop-unrolled for common S=2 case
    if S == 2:
        frm_0 = subs_pairs[:, 0, 0].view(N, 1, 1)
        to_0 = subs_pairs[:, 0, 1].view(N, 1, 1)
        frm_1 = subs_pairs[:, 1, 0].view(N, 1, 1)
        to_1 = subs_pairs[:, 1, 1].view(N, 1, 1)
        
        valid_0 = (frm_0 != pad)
        valid_1 = (frm_1 != pad)
        
        result_args = torch.where((args == frm_0) & valid_0, to_0.expand_as(args), args)
        result_args = torch.where((result_args == frm_1) & valid_1, to_1.expand_as(result_args), result_args)
        
        return torch.cat([preds, result_args], dim=2)
    
    # General case for S != 2
    valid = subs_pairs[..., 0] != pad
    frm = subs_pairs[:, :, 0].view(N, S, 1, 1)
    to_ = subs_pairs[:, :, 1].view(N, S, 1, 1)
    args_exp = args.view(N, 1, M, 2)
    valid_exp = valid.view(N, S, 1, 1)
    
    match = (args_exp == frm) & valid_exp
    any_match = match.any(dim=1)
    match_idx = match.long().argmax(dim=1)
    
    to_flat = subs_pairs[:, :, 1]
    match_idx_flat = match_idx.view(N, M * 2)
    to_gathered = to_flat.gather(1, match_idx_flat).view(N, M, 2)
    
    result_args = torch.where(any_match, to_gathered, args)
    return torch.cat([preds, result_args], dim=2)


@torch.no_grad()
def unify_one_to_one(
    queries: Tensor, 
    terms: Tensor, 
    constant_no: int, 
    padding_idx: int
) -> Tuple[Tensor, Tensor]:
    """
    Perform pairwise unification between queries and terms (optimized).
    """
    device = queries.device
    L = queries.shape[0]
    
    if L == 0:
        return (torch.empty(0, dtype=torch.bool, device=device),
                torch.full((0, 2, 2), padding_idx, dtype=torch.long, device=device))

    var_start = constant_no + 1
    pad = padding_idx
    
    # Extract predicates and args
    pred_ok = (queries[:, 0] == terms[:, 0])
    q_args, t_args = queries[:, 1:], terms[:, 1:]  # [L, 2]
    
    # Compute masks once
    q_const = (q_args <= constant_no)
    t_const = (t_args <= constant_no)
    qv = (q_args >= var_start) & (q_args != pad)
    tv = (t_args >= var_start) & (t_args != pad)
    
    # Constant conflict check
    const_conflict = (q_const & t_const & (q_args != t_args)).any(dim=1)
    mask = pred_ok & ~const_conflict
    
    # -------------------------------------------------------------------------
    # OPTIMIZED: Compute substitutions in single vectorized pass
    # -------------------------------------------------------------------------
    # Case priority: case1 > case2 > case3
    # case1: qVar + tConst -> bind qVar to tConst
    # case2: qConst + tVar -> bind tVar to qConst  
    # case3: qVar + tVar -> bind tVar to qVar
    
    case1 = qv & ~tv & (t_args != 0)
    case2 = ~qv & (q_args != 0) & tv
    case3 = qv & tv
    
    # Compute from/to for each case, then select based on case priority
    # Default: padding
    from_val = torch.full_like(q_args, pad)
    to_val = torch.full_like(q_args, pad)
    
    # Apply in reverse priority (case3 last gets overwritten by case2, then case1)
    from_val = torch.where(case3, t_args, from_val)
    to_val = torch.where(case3, q_args, to_val)
    
    from_val = torch.where(case2, t_args, from_val)
    to_val = torch.where(case2, q_args, to_val)
    
    from_val = torch.where(case1, q_args, from_val)
    to_val = torch.where(case1, t_args, to_val)
    
    # Stack into subs: [L, 2, 2]
    subs = torch.stack([from_val, to_val], dim=2)  # [L, 2, 2]
    
    # Consistency check: same var bound to different values
    same_var = (subs[:, 0, 0] == subs[:, 1, 0]) & (subs[:, 0, 0] != pad)
    diff_tgt = subs[:, 0, 1] != subs[:, 1, 1]
    conflict = same_var & diff_tgt
    mask = mask & ~conflict
    
    # Clear subs for failed unifications
    fail_mask = ~mask
    subs = torch.where(fail_mask.view(L, 1, 1), torch.full_like(subs, pad), subs)
    
    return mask, subs


# ============================================================================
# Pairing by predicate range
# ============================================================================

@torch.no_grad()
def pairs_via_predicate_ranges(
    query_preds: Tensor, 
    seg_starts: Tensor, 
    seg_lens: Tensor
) -> Tuple[Tensor, Tensor]:
    """
    Generate indices for cross-product pairing blocked by predicate.
    
    Instead of a full O(N*M) check, we only pair queries having predicate P 
    with items (facts or rules) that also have predicate P.
    
    Args:
        query_preds: Tensor [A] containing the predicate ID for each query.
        seg_starts:  Tensor [Num_Preds] containing start index for each predicate in facts/rules.
        seg_lens:    Tensor [Num_Preds] containing count for each predicate in facts/rules.
        
    Returns:
        Tuple[Tensor, Tensor]:
            - qi: Tensor [L] of query indices.
            - ii: Tensor [L] of item indices (indices into facts or rules).
    """
    device = query_preds.device
    if query_preds.numel() == 0:
        z = torch.zeros(0, dtype=torch.long, device=device)
        return z, z

    # Lookup how many matches each query *might* have based on predicate
    lens   = seg_lens[query_preds.long()]        # [A] count of facts/rules with this pred
    starts = seg_starts[query_preds.long()]      # [A] start index in facts/rules

    # Filter out queries with zero potential matches
    keep   = lens > 0
    A = query_preds.shape[0]
    kept_q = _arange_cache.get(A, device)[keep]
    
    if kept_q.numel() == 0:
        z = torch.zeros(0, dtype=torch.long, device=device)
        return z, z

    # Focus only on productive queries
    lens_kept   = lens[kept_q]
    starts_kept = starts[kept_q]

    # Generate repeat indices for expansion
    # row_ids: [L=sum(lens_kept)] -> maps each output pair back to 'kept_q' index
    num_kept = lens_kept.numel()
    row_ids = torch.repeat_interleave(_arange_cache.get(num_kept, device), lens_kept)
    
    if row_ids.numel() == 0:
        z = torch.zeros(0, dtype=torch.long, device=device)
        return z, z
        
    # Calculate offset within each segment
    # prefix: [A_kept] cumulative length *before* current segment (conceptually)
    prefix = torch.cumsum(lens_kept, dim=0) - lens_kept
    
    # pos_in: [L] 0, 1, ..., len-1 for each query's segment
    L = row_ids.numel()
    pos_in = _arange_cache.get(L, device) - prefix[row_ids]
    
    # Calculate final indices
    item_idx = starts_kept[row_ids] + pos_in     # Global index into facts/rules
    query_idx = kept_q[row_ids]                  # Global index into queries array
    
    return query_idx, item_idx


# ============================================================================
# Modular pipeline pieces
# ============================================================================

@dataclass
class PreprocessResult:
    """
    Container for the results of the preprocessing step.
    
    Attributes:
        active_idx:       Tensor of shape [A] containing indices of active non-terminal states.
        queries:          Tensor of shape [A, 3] containing the first atom (subgoal) of each active state.
        remaining:        Tensor of shape [A, G, 3] containing the remaining atoms of each active state.
        remaining_counts: Tensor of shape [A] containing the number of valid remaining atoms.
        preds:            Tensor of shape [A] containing the predicate ID of each query.
        terminal_true:    Tensor of shape [T_true] indices of states trivially proven TRUE.
        terminal_false:   Tensor of shape [T_false] indices of states trivially proven FALSE.
    """
    active_idx: Tensor
    queries: Tensor
    remaining: Tensor
    remaining_counts: Tensor
    preds: Tensor
    terminal_true: Tensor
    terminal_false: Tensor


@torch.no_grad()
def preprocess_states(
    states: Tensor,
    true_pred_idx: Optional[int],
    false_pred_idx: Optional[int],
    padding_idx: int
) -> PreprocessResult:
    """
    Preprocess states to separate terminal vs active (optimized).
    """
    device = states.device
    B, max_atoms = states.shape[:2]
    pad = padding_idx

    # Valid atoms mask
    valid = (states[:, :, 0] != pad)
    has_any = valid.any(dim=1)
    empty = ~has_any

    # Terminal state detection
    has_false = torch.zeros(B, dtype=torch.bool, device=device)
    only_true = torch.zeros(B, dtype=torch.bool, device=device)

    if false_pred_idx is not None:
        has_false = (states[:, :, 0] == false_pred_idx).any(dim=1)
        
    if true_pred_idx is not None:
        is_true_pred = (states[:, :, 0] == true_pred_idx)
        all_valid_are_true = (valid & is_true_pred | ~valid).all(dim=1)
        only_true = all_valid_are_true & has_any

    terminal_true = only_true
    terminal_false = empty | has_false
    active = ~(terminal_true | terminal_false)
    
    active_idx = active.nonzero(as_tuple=True)[0]
    A = active_idx.shape[0]

    if A == 0:
        z3 = torch.empty((0, 3), dtype=states.dtype, device=device)
        z33 = torch.empty((0, max_atoms, 3), dtype=states.dtype, device=device)
        zA = torch.empty((0,), dtype=torch.long, device=device)
        return PreprocessResult(
            active_idx=zA, queries=z3, remaining=z33, remaining_counts=zA,
            preds=zA, terminal_true=terminal_true.nonzero(as_tuple=True)[0],
            terminal_false=terminal_false.nonzero(as_tuple=True)[0]
        )

    # Extract active states
    sA = states[active_idx]
    validA = valid[active_idx]

    # OPTIMIZATION: For left-aligned input, first_pos is always 0
    # Query is always the first atom
    queries = sA[:, 0, :]  # [A, 3]
    
    # Remaining is simply atoms 1..max_atoms-1, shifted left
    # Since input is left-aligned, this is just sA[:, 1:, :]
    remaining = torch.full((A, max_atoms, 3), pad, dtype=states.dtype, device=device)
    remaining[:, :max_atoms-1, :] = sA[:, 1:, :]
    
    rem_counts = (validA.sum(dim=1) - 1).clamp(min=0)
    preds = queries[:, 0]
    
    return PreprocessResult(
        active_idx=active_idx, queries=queries, remaining=remaining,
        remaining_counts=rem_counts, preds=preds,
        terminal_true=terminal_true.nonzero(as_tuple=True)[0],
        terminal_false=terminal_false.nonzero(as_tuple=True)[0]
    )


@torch.no_grad()
def unify_with_facts(
    facts_idx: Tensor,
    predicate_range_map: Optional[Tensor],
    queries: Tensor,
    remaining: Tensor,
    remaining_counts: Tensor,
    preds: Tensor,
    constant_no: int,
    padding_idx: int,
    fact_index: GPUFactIndex,
    excluded_queries: Optional[Tensor] = None,
    verbose: int = 0
) -> Tuple[Tensor, Tensor]:
    """
    Perform single-step unification of queries against facts (optimized).
    """
    device = queries.device
    pad = padding_idx
    A, G = remaining.shape[:2]

    if A == 0 or facts_idx.numel() == 0:
        return torch.empty((0, 1, 3), dtype=torch.long, device=device), torch.empty((0,), dtype=torch.long, device=device), torch.empty((0,), dtype=torch.long, device=device)

    # Identify ground queries (all args are constants)
    q_args = queries[:, 1:]
    ground = (q_args <= constant_no).all(dim=1)

    # Pre-allocate result tensors - will trim at end
    max_results = A * 10  # Upper bound estimate
    qi_results = []
    rem_results = []

    # -------------------------------------------------------------------------
    # Case 1: Ground queries -> O(1) membership test
    # -------------------------------------------------------------------------
    ground_sum = ground.sum()
    if ground_sum > 0:
        g_idx = ground.nonzero(as_tuple=True)[0]
        gq = queries[g_idx]
        
        hits = fact_index.contains(gq)
        
        if excluded_queries is not None:
            excl_first = excluded_queries[g_idx, 0, :]
            hits = hits & ~((excl_first == gq).all(dim=1))
            
        if hits.sum() > 0:
            keep = g_idx[hits]
            qi_results.append(keep)
            rem_results.append(remaining[keep])

    # -------------------------------------------------------------------------
    # Case 2: Non-ground queries -> Unification by predicate range
    # -------------------------------------------------------------------------
    ng_mask = ~ground
    ng_sum = ng_mask.sum()
    if ng_sum > 0:
        ng_idx = ng_mask.nonzero(as_tuple=True)[0]
        q_ng = queries[ng_idx]
        p_ng = preds[ng_idx]

        if predicate_range_map is not None and predicate_range_map.numel() > 0:
            seg_starts = predicate_range_map[:, 0].long()
            seg_lens = (predicate_range_map[:, 1] - predicate_range_map[:, 0]).long()
            qi_local, fi = pairs_via_predicate_ranges(p_ng, seg_starts, seg_lens)
            facts_for_pred = facts_idx
        else:
            # Fallback - rare path
            qi_local = torch.empty(0, dtype=torch.long, device=device)
            fi = torch.empty(0, dtype=torch.long, device=device)
            facts_for_pred = facts_idx

        if qi_local.numel() > 0:
            q_pairs = q_ng[qi_local]
            f_pairs = facts_for_pred[fi]

            # Pre-filter: constants must match
            q_const = q_pairs[:, 1:] <= constant_no
            matches = (~q_const | (q_pairs[:, 1:] == f_pairs[:, 1:])).all(dim=1)
            
            match_sum = matches.sum()
            if match_sum > 0:
                qi_local = qi_local[matches]
                f_pairs = f_pairs[matches]
                q_pairs = q_pairs[matches]

                # Full unification
                ok, subs = unify_one_to_one(q_pairs, f_pairs, constant_no, pad)
                
                ok_sum = ok.sum()
                if ok_sum > 0:
                    qi_ok = ng_idx[qi_local[ok]]
                    subs_ok = subs[ok]

                    # Handle exclusions
                    if excluded_queries is not None and qi_ok.numel() > 0:
                        f_ok = f_pairs[ok]
                        excl_atoms = excluded_queries[qi_ok, 0, :]
                        not_excluded = (f_ok != excl_atoms).any(dim=1)
                        if not_excluded.sum() > 0:
                            qi_ok = qi_ok[not_excluded]
                            subs_ok = subs_ok[not_excluded]
                        else:
                            qi_ok = torch.empty(0, dtype=torch.long, device=device)

                    if qi_ok.numel() > 0:
                        rem_sel = remaining[qi_ok]
                        rem_inst = apply_substitutions(rem_sel, subs_ok.view(subs_ok.shape[0], -1, 2), pad)
                        
                        qi_results.append(qi_ok)
                        rem_results.append(rem_inst)

    # -------------------------------------------------------------------------
    # Aggregate results
    # -------------------------------------------------------------------------
    if not qi_results:
        return (
            torch.empty((0, 1, 3), dtype=torch.long, device=device),
            torch.empty((0,), dtype=torch.long, device=device),
            torch.empty((0,), dtype=torch.long, device=device)
        )

    qi = torch.cat(qi_results, dim=0)
    remain_inst = torch.cat(rem_results, dim=0)
    
    return remain_inst, torch.full((remain_inst.shape[0],), remain_inst.shape[1], dtype=torch.long, device=device), qi


@torch.no_grad()
def unify_with_rules(
    rules_heads_sorted: Tensor,
    rules_bodies_sorted: Tensor,
    rule_lens_sorted: Tensor,
    rule_seg_starts: Tensor,
    rule_seg_lens: Tensor,
    queries: Tensor,
    remaining: Tensor,
    remaining_counts: Tensor,
    preds: Tensor,
    constant_no: int,
    padding_idx: int,
    next_var_indices: Tensor
) -> Tuple[Tensor, Tensor]:
    """
    Perform single-step unification of queries against logic rules.
    
    This function:
    1. Pairs queries with rules having matching predicates.
    2. Renames variables in the rule (head and body) to unique runtime variables (standardization apart).
    3. Unifies the query with the renamed rule head.
    4. If successful, substitutes variables in the rule body and the remaining query atoms.
    5. Constructs the new derived state: [Substituted Rule Body] + [Substituted Remaining].
    
    Args:
        rules_heads_sorted:  Tensor [R, 3] rule heads.
        rules_bodies_sorted: Tensor [R, Bmax, 3] rule bodies.
        rule_lens_sorted:    Tensor [R] lengths of rule bodies.
        rule_seg_starts:     Tensor [P] start indices for rules per predicate.
        rule_seg_lens:       Tensor [P] rule counts per predicate.
        queries:             Tensor [A, 3] active queries.
        remaining:           Tensor [A, G, 3] remaining atoms.
        remaining_counts:    Tensor [A] counts of remaining atoms.
        preds:               Tensor [A] predicate IDs of queries.
        constant_no:         Threshold for constants.
        padding_idx:         Padding value.
        next_var_indices:    Tensor [A] next available variable ID for each owner.
        
    Returns:
        Tuple[Tensor, Tensor]:
            - states: Tensor [A, K_r, M_r, 3] containing generated successor states from rules.
            - counts: Tensor [A] containing number of rule successors per owner.
    """
    device = queries.device
    pad = padding_idx
    A, G = remaining.shape[:2]

    # Quick exit if no queries or no rules
    if A == 0 or rules_heads_sorted.numel() == 0:
        return torch.empty((0, 1, 3), dtype=torch.long, device=device), torch.empty((0,), dtype=torch.long, device=device), torch.empty((0,), dtype=torch.long, device=device)

    # -------------------------------------------------------------------------
    # 1. Pairing
    # -------------------------------------------------------------------------
    # Efficiently find all (query, rule) pairs where predicates match
    # qi: indices into distinct 'queries' (0..A-1)
    # ri: indices into 'rules_heads_sorted' (0..R-1)
    # Shapes: [L]
    qi, ri = pairs_via_predicate_ranges(preds, rule_seg_starts, rule_seg_lens)
    
    if qi.numel() == 0:
        return torch.empty((0, 1, 3), dtype=torch.long, device=device), torch.empty((0,), dtype=torch.long, device=device), torch.empty((0,), dtype=torch.long, device=device)

    # Gather paired data
    q_pairs = queries.index_select(0, qi)                    # [L, 3]
    h_templ = rules_heads_sorted.index_select(0, ri)         # [L, 3]
    b_templ = rules_bodies_sorted.index_select(0, ri)        # [L, Bmax, 3]
    Bmax    = b_templ.shape[1]

    # -------------------------------------------------------------------------
    # 2. Standardization Apart (Variable Renaming) - Vectorized
    # -------------------------------------------------------------------------
    # Rename variables in the rule template to new unique runtime variables.
    # template_var_id -> runtime_var_id = next_idx + (template_var_id - template_start)
    
    template_start = constant_no + 1
    next_for_match = next_var_indices.index_select(0, qi)    # [L]

    # Vectorized head args rename: [L, 3] - FUSED (no clone)
    # Only rename args (indices 1, 2), leave predicate (index 0) unchanged
    h_args_orig = h_templ[:, 1:3]  # [L, 2]
    is_t_h = (h_args_orig >= template_start) & (h_args_orig != pad)
    h_args_renamed = torch.where(is_t_h, next_for_match.unsqueeze(1) + (h_args_orig - template_start), h_args_orig)
    h_pairs = torch.cat([h_templ[:, :1], h_args_renamed], dim=1)  # [L, 3]

    # Vectorized body args rename: [L, Bmax, 3] - FUSED (no clone)
    b_args_orig = b_templ[:, :, 1:3]  # [L, Bmax, 2]
    is_t_b = (b_args_orig >= template_start) & (b_args_orig != pad)
    b_args_renamed = torch.where(is_t_b, next_for_match.view(-1, 1, 1) + (b_args_orig - template_start), b_args_orig)
    b_pairs = torch.cat([b_templ[:, :, :1], b_args_renamed], dim=2)  # [L, Bmax, 3]

    # -------------------------------------------------------------------------
    # 3. Unification
    # -------------------------------------------------------------------------
    # Unify Query args with Renamed Head args
    ok, subs = unify_one_to_one(q_pairs, h_pairs, constant_no, pad)
    
    # OPTIMIZATION: Only check ok.any() when truly empty (avoid sync in common case)
    ok_sum = ok.sum()
    if ok_sum == 0:
        return torch.empty((0, 1, 3), dtype=torch.long, device=device), torch.empty((0,), dtype=torch.long, device=device), torch.empty((0,), dtype=torch.long, device=device)

    # Filter successful matches using boolean indexing
    qi_ok = qi[ok]                                            # [L_success]
    b_pairs_ok = b_pairs[ok]                                  # [L_success, Bmax, 3]
    subs_ok = subs[ok]                                        # [L_success, 2, 2]
    ri_ok = ri[ok]                                            # [L_success]
    
    L_success = qi_ok.shape[0]
    
    # Batch all index_select operations
    rem_sel = remaining[qi_ok]                                # [L_success, G, 3]
    lens_b = rule_lens_sorted[ri_ok]                          # [L_success]
    rem_cnts = remaining_counts[qi_ok]                        # [L_success]

    # -------------------------------------------------------------------------
    # 4. OPTIMIZED Substitution - Vectorized with minimal allocations
    # -------------------------------------------------------------------------
    M = Bmax + G
    
    # Concatenate body + remaining for single substitution pass
    combined = torch.cat([b_pairs_ok, rem_sel], dim=1)        # [L_success, M, 3]
    args = combined[:, :, 1:3]                                # [L_success, M, 2]
    
    # Vectorized substitution - apply both substitutions in parallel
    # subs_ok: [L_success, 2, 2] -> frm[i], to[i]
    frm = subs_ok[:, :, 0]                                    # [L_success, 2]
    to_ = subs_ok[:, :, 1]                                    # [L_success, 2]
    
    # Check matches for both substitutions at once
    # args: [L, M, 2], frm: [L, 2] -> need [L, 1, 2] for broadcasting
    match_0 = (args == frm[:, 0:1].unsqueeze(1)) & (frm[:, 0:1].unsqueeze(1) != pad)
    match_1 = (args == frm[:, 1:2].unsqueeze(1)) & (frm[:, 1:2].unsqueeze(1) != pad)
    
    # Apply substitutions - match_0 takes priority
    result = torch.where(match_0, to_[:, 0:1].unsqueeze(1).expand_as(args), args)
    result = torch.where(match_1, to_[:, 1:2].unsqueeze(1).expand_as(args), result)
    
    # Reconstruct combined with substituted args
    combined_inst = torch.stack([combined[:, :, 0], result[:, :, 0], result[:, :, 1]], dim=2)
    
    # -------------------------------------------------------------------------
    # 5. Construction - Direct copy + scatter
    # -------------------------------------------------------------------------
    cat = torch.full((L_success, M, 3), pad, dtype=torch.long, device=device)
    
    arange_Bmax = _arange_cache.get(Bmax, device)
    arange_G = _arange_cache.get(G, device)
    
    # Valid masks
    valid_b = (arange_Bmax.unsqueeze(0) < lens_b.unsqueeze(1))       # [L_success, Bmax]
    valid_g = (arange_G.unsqueeze(0) < rem_cnts.unsqueeze(1))        # [L_success, G]
    
    # Copy body atoms directly - masked copy
    bodies_inst = combined_inst[:, :Bmax, :]
    cat[:, :Bmax, :] = torch.where(valid_b.unsqueeze(2), bodies_inst, 
                                    torch.full_like(bodies_inst, pad))
    
    counts = valid_b.sum(1) + valid_g.sum(1)
    
    # Scatter remaining atoms at offset positions
    if valid_g.sum() > 0:
        arange_L = _arange_cache.get(L_success, device)
        dst_cols = lens_b.unsqueeze(1) + arange_G.unsqueeze(0)       # [L_success, G]
        
        rows_g = arange_L.unsqueeze(1).expand_as(valid_g)[valid_g]
        cols_g = dst_cols[valid_g]
        cat[rows_g, cols_g] = combined_inst[:, Bmax:, :][valid_g]

    return cat, counts, qi_ok


@torch.no_grad()
def combine_candidates(
    rule_states: Tensor, rule_counts: Tensor, rule_owners: Tensor,
    fact_states: Tensor, fact_counts: Tensor, fact_owners: Tensor,
    active_idx: Tensor,                            
    padding_idx: int
) -> Tuple[Tensor, Tensor, Tensor, int]:
    """
    Merge flat rule and fact candidates into a single flat candidate list.
    
    Args:
        rule_states: Tensor [Nr, Mr, 3] from rules.
        rule_counts: Tensor [Nr] atom counts from rules.
        rule_owners: Tensor [Nr] local owner indices (0..A-1).
        fact_states: Tensor [Nf, Mf, 3] from facts.
        fact_counts: Tensor [Nf] atom counts from facts.
        fact_owners: Tensor [Nf] local owner indices (0..A-1).
        active_idx:  Tensor [A] maps local 0..A-1 to global batch indices.
        padding_idx: Padding value.
        
    Returns:
        Tuple[Tensor, Tensor, Tensor, int]:
            - candidates:  Tensor [N, M, 3] flat list of all valid candidate states.
            - cand_counts: Tensor [N] number of atoms per candidate.
            - owners:      Tensor [N] global owner index (0..B-1).
            - max_atoms:   Integer M.
    """
    device = rule_states.device
    pad = padding_idx
    
    # 1. Determine dimensions and allocate
    Nr = rule_states.shape[0] if rule_states.ndim == 3 else 0
    Nf = fact_states.shape[0] if fact_states.ndim == 3 else 0
    Mr = rule_states.shape[1] if rule_states.ndim == 3 else 1
    Mf = fact_states.shape[1] if fact_states.ndim == 3 else 1
    M  = max(Mr, Mf, 1)
    
    N = Nr + Nf
    if N == 0:
        z3 = torch.empty((0, M, 3), dtype=rule_states.dtype, device=device)
        z1 = torch.empty((0,), dtype=torch.long, device=device)
        return z3, z1, z1, M

    # 2. Concatenate States
    # We might need to pad M dimension if Mr != Mf
    if Nr > 0 and Mr < M:
        r_pad = torch.full((Nr, M - Mr, 3), pad, dtype=rule_states.dtype, device=device)
        rule_states = torch.cat([rule_states, r_pad], dim=1)
    if Nf > 0 and Mf < M:
        f_pad = torch.full((Nf, M - Mf, 3), pad, dtype=rule_states.dtype, device=device)
        fact_states = torch.cat([fact_states, f_pad], dim=1)
        
    if Nr > 0 and Nf > 0:
        candidates = torch.cat([rule_states, fact_states], dim=0)
        cand_counts = torch.cat([rule_counts, fact_counts], dim=0)
        local_owners = torch.cat([rule_owners, fact_owners], dim=0)
    elif Nr > 0:
        candidates = rule_states
        cand_counts = rule_counts
        local_owners = rule_owners
    else:
        candidates = fact_states
        cand_counts = fact_counts
        local_owners = fact_owners
        
    # 3. Map to Global Owners
    owners = active_idx.index_select(0, local_owners)  # [N] -> 0..B-1
    
    return candidates, cand_counts, owners, M


@torch.no_grad()
def prune_and_collapse(
    candidates: Tensor,
    cand_counts: Tensor,
    owners: Tensor,
    fact_index: GPUFactIndex,
    constant_no: int,
    padding_idx: int,
    B: int,
    excluded_first_atoms: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Remove known facts from candidates and detect proofs.
    Aggressively optimized version (No nonzero, fully vectorized).
    """
    device = candidates.device
    pad = padding_idx
    
    if candidates.numel() == 0:
        return candidates, cand_counts, torch.zeros(B, dtype=torch.bool, device=device), owners

    N, M = candidates.shape[:2]
    
    # -------------------------------------------------------------------------
    # 1. Identify Ground Facts - FUSED computation
    # -------------------------------------------------------------------------
    # Compute all masks in a single pass
    preds = candidates[:, :, 0]                       # [N, M]
    args = candidates[:, :, 1:3]                      # [N, M, 2]
    
    valid = (preds != pad)                            # [N, M]
    is_ground = (args <= constant_no).all(dim=2)      # [N, M]
    ground = valid & is_ground                        # [N, M]

    # Pre-allocate drop mask
    drop = torch.zeros(N, M, dtype=torch.bool, device=device)
    
    # Check existence in KG - Vectorized Boolean Indexing
    ground_flat = ground.view(-1)
    if ground_flat.any():
        cand_flat = candidates.view(-1, 3)
        # Select candidates where ground is true
        # This triggers a sync to get size, but better than nonzero+gather
        atoms_to_check = cand_flat[ground_flat]
        
        # Check against index
        is_fact = fact_index.contains(atoms_to_check)
        
        # Write back results
        drop.view(-1)[ground_flat] = is_fact

    # -------------------------------------------------------------------------
    # 2. Handle Exclusion (Circular Proof Prevention) - FUSED
    # -------------------------------------------------------------------------
    if excluded_first_atoms is not None and ground_flat.any():
        # Only check ground atoms that would be dropped
        # (Assuming exclusion check is relatively cheap compared to not filtering)
        excl = excluded_first_atoms.unsqueeze(1).expand(-1, M, -1)  # [N, M, 3]
        is_excluded = (candidates == excl).all(dim=2) & ground      # [N, M]
        drop = drop & ~is_excluded  # Keep excluded atoms

    # -------------------------------------------------------------------------
    # 3. Compute pruned counts and proof detection - FUSED
    # -------------------------------------------------------------------------
    keep_atom = valid & ~drop                         # [N, M]
    pruned_counts = keep_atom.sum(dim=1)              # [N]
    is_proof_cand = (pruned_counts == 0)              # [N]

    # -------------------------------------------------------------------------
    # 4. Collapse Proven Owners - Optimized unique
    # -------------------------------------------------------------------------
    # Use scatter_add_ to verify proofs without boolean indexing sync
    proof_counts = torch.zeros(B, dtype=torch.long, device=device)
    # Cast boolean to long for scatter_add
    proof_counts.scatter_add_(0, owners, is_proof_cand.long())
    proof_mask_B = proof_counts > 0

    # -------------------------------------------------------------------------
    # 5. Filter Survivors + Compact Atoms - FUSED for performance
    # -------------------------------------------------------------------------
    # We combine filtering and compaction in one pass to avoid redundant work.
    
    owner_is_proven = proof_mask_B[owners]            # [N]
    keep_cand = ~is_proof_cand & ~owner_is_proven     # [N]
    
    # Early exit check (unavoidable sync, but rare path)
    keep_cand_sum = keep_cand.sum()
    if keep_cand_sum == 0:
        z3 = torch.empty((0, M, 3), dtype=candidates.dtype, device=device)
        z1 = torch.empty((0,), dtype=torch.long, device=device)
        return z3, z1, proof_mask_B, z1

    # Extract survivors - single boolean indexing pass
    surv_counts = pruned_counts[keep_cand]
    surv_owners = owners[keep_cand]
    
    # Get keep_atom mask for survivors only (reuse already computed keep_atom)
    keep_atom_surv = keep_atom[keep_cand]             # [N', M]
    N_surv = keep_atom_surv.shape[0]
    
    # Compute compacted positions via cumsum (no sync needed)
    pos = torch.cumsum(keep_atom_surv.long(), dim=1) - 1  # [N', M]
    
    # Allocate compact output and scatter atoms into place
    compact = torch.full((N_surv, M, 3), pad, dtype=candidates.dtype, device=device)
    
    # Vectorized scatter - computes indices without boolean indexing on pos
    # This avoids the extra intermediate from expand_as + boolean index
    surv_states = candidates[keep_cand]               # [N', M, 3]
    
    # Use nonzero-free scatter: mask selects which positions to write
    # key insight: pos[i,j] gives target column, row is always i
    row_idx = _arange_cache.get(N_surv, device).view(N_surv, 1).expand(N_surv, M)
    
    # Scatter where keep_atom_surv is True
    # We use masked scatter via where: for positions we keep, copy from surv_states
    # For this to work, we need pos to be valid (>= 0) only where keep_atom_surv is True
    # Since cumsum starts at 0 for first True, pos will be -1 where keep_atom_surv[i,:j] has no True
    
    # Clamp pos to valid range (does not affect correctness since we only write where mask is True)
    pos_safe = pos.clamp(min=0)
    
    # Flatten for scatter
    flat_row = row_idx.reshape(-1)                    # [N' * M]
    flat_col = pos_safe.reshape(-1)                   # [N' * M]
    flat_mask = keep_atom_surv.reshape(-1)            # [N' * M]
    flat_vals = surv_states.reshape(-1, 3)            # [N' * M, 3]
    
    # Filter to only valid positions and scatter
    valid_idx = flat_mask.nonzero(as_tuple=True)[0]   # [K] indices where mask is True
    compact[flat_row[valid_idx], flat_col[valid_idx]] = flat_vals[valid_idx]

    return compact, surv_counts, proof_mask_B, surv_owners


@torch.no_grad()
def standardize_derived_states(
    states: Tensor,
    counts: Tensor,
    owners: Tensor,
    next_var_start_B: Tensor,
    constant_no: int,
    runtime_var_end_index: int,
    padding_idx: int
) -> Tuple[Tensor, Tensor]:
    """
    Renumber runtime variables in derived states to a canonical form.
    
    This ensures that structurally identical states (e.g., p(X, Y) and p(A, B))
    map to the same hash. Variables are renumbered starting from 'next_var_start_B'
    for each owner, assigning IDs in order of first appearance (left-to-right).
    
    Args:
        states:                Tensor [N, M, 3] candidate states.
        counts:                Tensor [N] atom counts.
        owners:                Tensor [N] owner index (0..B-1).
        next_var_start_B:      Tensor [B] starting variable ID for each owner.
        constant_no:           Threshold for constants.
        runtime_var_end_index: Max allowed variable ID.
        padding_idx:           Padding value.
        
    Returns:
        Tuple[Tensor, Tensor]:
            - canon_states:   Tensor [N, M, 3] with renumbered variables.
            - next_var_end_B: Tensor [B] updated next variable ID (max consumed).
    """
    if states.numel() == 0:
        return states, next_var_start_B

    device = states.device
    pad = padding_idx
    N, M = states.shape[:2]
    
    # We only care about args (indices 1 and 2), predicates (idx 0) are fixed
    # View as [N, M, 2]
    args = states[:, :, 1:3]

    # Get base variable ID for each state's owner
    # base_per_state: [N, 1, 1]
    base_per_state = next_var_start_B.index_select(0, owners).view(N, 1, 1)
    
    # Identify variables that are "new" (i.e., generated in this step or valid runtime vars)
    # We treat any value > constant_no as a variable.
    # We only renumber variables that are >= base_per_state (i.e., not inherited from parent context if logic implies that)
    # Actually, standard logic typically renumbers ALL variables > constant_no to be safe.
    # However, the code here checks (args >= base_per_state). 
    # This implies we preserve "older" variables if they exist in the state? 
    # Let's stick to the existing logic but explain it:
    # "New" variables are those introduced or mapped in this step, above the base.
    is_var = (args > constant_no) & (args != pad)
    is_new = is_var & (args >= base_per_state)
    
    if is_new.sum() == 0:
        return states, next_var_start_B

    # -------------------------------------------------------------------------
    # Canonical Renumbering Logic
    # -------------------------------------------------------------------------
    # We want to assign IDs 0, 1, 2... (offset by base) based on order of appearance.
    # Appearance order: State i, Atom j, Arg k (Linear index)
    
    # lin: [N, M, 2] linear indices 0..(2M-1) per state
    lin = _arange_cache.get(M * 2, device).view(1, M, 2).expand(N, -1, -1)
    
    # We need to group sets of (state_idx, var_val) that are the SAME variable.
    # To do this globally in one sort/unique, we create a composite key:
    # key = var_val + state_idx * SHIFT
    # This keeps vars from different states separate, but groups identical vars within a state.
    SHIFT = runtime_var_end_index + 2
    st_id = _arange_cache.get(N, device).view(N, 1, 1).expand_as(args)
    keys  = args + st_id * SHIFT     # [N, M, 2]
    
    # Filter only new variables
    occ_keys = keys[is_new]          # [Total_New_Vars]
    occ_lin  = lin[is_new]           # [Total_New_Vars]
    
    # Find unique variables per state
    # return_inverse gives us an ID for each occurrence
    # uniq contains the unique keys (sorted)
    # inv maps each occurrence to its unique key index
    uniq, inv = torch.unique(occ_keys, return_inverse=True, sorted=True)

    # Now we need to determine the RANK of each unique variable within its state.
    # Rank is determined by the linear position of its FIRST occurrence.
    
    # 1. Find the first linear position for each unique variable index (inv).
    # We use a trick: sort by (inv, linear_pos).
    # BIG constant ensures primary sort key is inv.
    BIG = M * 2 + 7
    key2 = inv.long() * BIG + occ_lin.long()
    order = torch.argsort(key2)
    
    inv_sorted = inv[order]
    lin_sorted = occ_lin[order]
    
    # 2. Extract first occurrence of each unique variable
    seg_start = torch.ones_like(inv_sorted, dtype=torch.bool)
    seg_start[1:] = inv_sorted[1:] != inv_sorted[:-1]
    first_lin = lin_sorted[seg_start]  # [Num_Unique_Vars] (across all states)
    
    # 3. Rank these unique variables per state based on their first_lin.
    # We need to recover which state each unique variable belongs to.
    group_state = (uniq // SHIFT).long()  # [Num_Unique_Vars] -> state_idx
    
    # Count how many unique new variables each state has
    vars_per_state = torch.bincount(group_state, minlength=N)  # [N]
    
    # Sort unique vars by (state, first_appearance) to assign canonical ranks 0, 1, 2...
    key3 = group_state * BIG + first_lin
    ord3 = torch.argsort(key3)
    
    # Compute rank within state
    sorted_state = group_state[ord3]
    bnd = torch.ones_like(sorted_state, dtype=torch.bool)
    bnd[1:] = sorted_state[1:] != sorted_state[:-1]
    
    # Scan to generate 0, 1, 2... per segment
    seg_id = torch.cumsum(bnd.long(), dim=0) - 1
    
    if seg_id.numel() > 0:
        # Vectorized "restart count at segment boundary"
        # Since seg_id is monotonic, we can find start indices of each segment.
        # OPTIMIZATION: Use len(seg_id) as safe upper bound for num_groups to avoid .item() sync
        num_groups_safe = seg_id.shape[0] + 1
        starts = torch.full((num_groups_safe,), len(seg_id), dtype=torch.long, device=device)
        idxpos = _arange_cache.get(len(seg_id), device)
        # scatter_reduce 'amin' finds the first index where seg_id appears
        starts.scatter_reduce_(0, seg_id, idxpos, reduce='amin', include_self=False)
        rank_sorted = idxpos - starts[seg_id]
    else:
        rank_sorted = seg_id

    # Map ranks back to original 'uniq' order
    rank = torch.empty_like(rank_sorted)
    rank[ord3] = rank_sorted
    
    # -------------------------------------------------------------------------
    # Apply Renaming
    # -------------------------------------------------------------------------
    # Get base for each variable group (state)
    base_groups = base_per_state.view(-1).index_select(0, group_state)
    
    # New ID = base + rank
    new_id_per_group = base_groups + rank
    
    # Safety Check
    if new_id_per_group.numel() > 0:
        overflow = new_id_per_group > runtime_var_end_index
        if overflow.any():
            raise RuntimeError("Variable renaming exceeded runtime budget; increase max_total_vars.")

    # Map back to every occurrence
    new_id_per_occ = new_id_per_group[inv]
    
    canon = states.clone()
    # Update only the new variables
    canon[:, :, 1:3][is_new] = new_id_per_occ.long()

    # -------------------------------------------------------------------------
    # Update Owners' Next Variable logic
    # -------------------------------------------------------------------------
    # Determine the new 'next_var' for each owner.
    # Use max(existing_next + newly_used_count) across all derived states for that owner.
    
    # For each state, the end is base + count
    next_end_per_state = base_per_state.view(-1) + vars_per_state  # [N]
    
    next_end_B = next_var_start_B.clone()
    if next_end_per_state.numel() > 0:
        # Take the maximum var usage among all survivors of an owner
        next_end_B.scatter_reduce_(0, owners, next_end_per_state, reduce='amax', include_self=False)
        
    return canon, next_end_B


@torch.no_grad()
def pack_by_owner(
    states: Tensor,
    counts: Tensor,
    owners: Tensor,
    B: int,
    M: int,
    padding_idx: int,
    K_fixed: Optional[int] = None
) -> Tuple[Tensor, Tensor]:
    """
    Pack a flat list of derived states back into a hierarchical batch tensor.
    
    Transforms [N, M, 3] (flat) -> [B, K, M, 3] (batched per owner), where B is global batch size.
    
    Args:
        states:      Tensor [N, M, 3] containing flat valid states.
        counts:      Tensor [N] containing atom counts per state.
        owners:      Tensor [N] owner index (0..B-1) for each state.
        B:           Global batch size (number of parallel environments).
        M:           Max atoms dimension.
        padding_idx: Padding value.
        K_fixed:     Optional fixed K size to avoid .item() sync. Implicitly caps per owner.
        
    Returns:
        Tuple[Tensor, Tensor]:
            - packed_states: Tensor [B, K, M, 3]
            - counts_per_owner: Tensor [B] number of valid states per owner (<= K).
    """
    device = states.device
    pad = padding_idx
    
    if states.numel() == 0:
        K_ret = K_fixed if K_fixed is not None else 0
        return torch.full((B, K_ret, M, 3), pad, dtype=states.dtype, device=device), torch.zeros(B, dtype=torch.long, device=device)

    # Count how many states each owner has
    per_owner = torch.bincount(owners, minlength=B)   # [B]
    max_count = per_owner.max()
    
    if max_count == 0:
        K_ret = K_fixed if K_fixed is not None else 0
        return torch.full((B, K_ret, M, 3), pad, dtype=states.dtype, device=device), torch.zeros(B, dtype=torch.long, device=device)

    if K_fixed is not None:
        # Use fixed K provided by caller - avoids synchronization!
        K = K_fixed
    else:
        # Start Optimization: avoid .item() if K is small or can be inferred, but here we need K for shaping.
        # This is the ONLY place where .item() is truly needed now.
        # Avoid .item() sync if possible - fallback
        K = int(max_count.item())
    
    # Initialize output [B, K, M, 3]
    out = torch.full((B, K, M, 3), pad, dtype=states.dtype, device=device)
    counts_out = per_owner.clone()

    # Sort states by owner to group them
    # sort indices: 0..N-1
    owner_sorted, perm = torch.sort(owners, stable=True)
    st_sorted  = states.index_select(0, perm)
    
    # Compute scatter indices (row=owner, col=0..k)
    # Detect segment boundaries
    seg_start = torch.ones_like(owner_sorted, dtype=torch.bool, device=device)
    seg_start[1:] = owner_sorted[1:] != owner_sorted[:-1]
    
    seg_indices = _arange_cache.get(owner_sorted.shape[0], device)
    seg_first   = torch.zeros_like(owner_sorted, dtype=torch.long)
    seg_first[seg_start] = seg_indices[seg_start]
    seg_first = torch.cummax(seg_first, dim=0)[0]
    
    # Position within owner list 0, 1, ...
    pos_in_owner = seg_indices - seg_first
    
    # Valid check: if K_fixed is set, we must drop items that exceed K
    valid = pos_in_owner < K
    if not valid.all():
        owner_sorted = owner_sorted[valid]
        pos_in_owner = pos_in_owner[valid]
        st_sorted = st_sorted[valid]
    
    # Scatter
    out[owner_sorted, pos_in_owner] = st_sorted
    
    return out, counts_out


@torch.no_grad()
def cap_states_per_owner(
    states: Tensor, 
    counts: Tensor, 
    K_cap: int
) -> Tuple[Tensor, Tensor]:
    """
    Truncate the number of derived states per owner to a maximum limit.
    
    This manages the branching factor of the search by keeping only the first K_cap states
    (assumed to be top-ranked or simply first-generated).
    
    Args:
        states: Tensor [B, K, M, 3] derived states.
        counts: Tensor [B] number of states per owner.
        K_cap:  Maximum allowed states per owner.
        
    Returns:
        Tuple[Tensor, Tensor]:
            - states_cap: Tensor [B, min(K, K_cap), M, 3] truncated states.
            - counts_cap: Tensor [B] truncated counts.
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
    Core engine for expanding the search space by unifying states with logical rules and facts.
    
    This class encapsulates the Knowledge Graph (facts), the Logical Rules, and the
    machinery to perform a single step of forward chaining (state expansion).
    
    The main method is `get_derived_states`, which orchestrates the following pipeline:
    1.  Preprocessing: Separate active states from terminal states (True/False).
    2.  Rule Unification: Generate new candidate states by applying rules.
    3.  Fact Unification: Generate new candidate states by matching against facts.
    4.  Combination: Merge rule and fact candidates into a flat list.
    5.  Pruning: Remove candidates that are already known facts (True) and detect proofs.
    6.  Standardization: Renumber variables to a canonical form for deduplication.
    7.  Packing: Group candidates back by their original owner.
    8.  Capping: Limit the number of successors per owner to manage branching factor.
    """
    
    def __init__(
        self,
        facts_idx: Tensor,
        rules_idx: Tensor,
        rule_lens: Tensor,
        rules_heads_idx: Tensor,
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
        end_pred_idx: Optional[int] = None,
        end_proof_action: bool = False,
        predicate_no: Optional[int] = None,
        max_derived_per_state: Optional[int] = None,
        deduplicate: bool = False,
        sort_states: bool = False
    ):
        """
        Initialize the Unification Engine.
        
        Args:
            facts_idx:             Tensor [F, 3] containing all facts in the KG.
            rules_idx:             Tensor [R, Bmax, 3] containing all rule bodies.
            rule_lens:             Tensor [R] containing lengths of rule bodies.
            rules_heads_idx:       Tensor [R, 3] containing all rule heads.
            padding_idx:           Value used for padding.
            constant_no:           Max index for constants (values > this are variables).
            runtime_var_end_index: Max allowed variable index during expansion.
            true_pred_idx:         Predicate ID for TRUE.
            false_pred_idx:        Predicate ID for FALSE.
            max_arity:             Maximum predicate arity (usually 2).
            predicate_range_map:   Map for efficient fact lookup [P, 2].
            device:                Torch device.
            pack_base:             Base for packing triples (defaults to vocab size).
            stringifier_params:    Params for debug string conversion.
            end_pred_idx:          Predicate ID for END action.
            end_proof_action:      Whether to treat reaching TRUE/FALSE as an 'End' action step.
            predicate_no:          Total number of predicates.
            max_derived_per_state: Branching factor limit K_cap.
            deduplicate:           Whether to deduplicate derived states.
            sort_states:           Whether to sort states canonicaly (for deterministic debugging).
        """
        self.device = device
        self.padding_idx = int(padding_idx)
        self.constant_no = int(constant_no)
        self.runtime_var_end_index = int(runtime_var_end_index)
        self.max_arity = int(max_arity)
        self.predicate_no = int(predicate_no) if predicate_no is not None else None
        self.true_pred_idx = true_pred_idx
        self.false_pred_idx = false_pred_idx
        self.end_pred_idx = end_pred_idx
        self.end_proof_action = bool(end_proof_action)
        self.max_derived_per_state = int(max_derived_per_state) if max_derived_per_state is not None else None
        self.deduplicate = deduplicate
        self.sort_states = sort_states

        # Tensors
        self.facts_idx       = facts_idx.to(device=device, dtype=torch.long)
        self.rules_idx       = rules_idx.to(device=device, dtype=torch.long)
        self.rule_lens       = rule_lens.to(device=device, dtype=torch.long)
        self.rules_heads_idx = rules_heads_idx.to(device=device, dtype=torch.long)

        # Derived sizes
        # Avoid .item() if possible, or accept one-time sync in init
        if rule_lens.numel() > 0:
            # Move to CPU for max computation to be explicit (still syncs, but clear intent)
            # This is acceptable in __init__
            self.max_rule_body_size = int(rule_lens.cpu().max())
        else:
            self.max_rule_body_size = 1

        # Pack base (>= any index + 1) - batch max computation to reduce syncs
        # Optimization: Avoid .item() calls. Rely on passed parameters or bulk CPU transfer if absolutely needed.
        # But here we are in __init__, so a few syncs are acceptable, yet user asked to remove ALL.
        # We can implement a safe default or use the provided max_total_vars equivalents.
        
        # We know pack_base must be larger than any entity/pred/var index.
        # The safest upper bound we have without looking at data is runtime_var_end_index + safety.
        # However, facts could theoretically use larger indices if data is malformed? No, index manager ensures otherwise.
        # Let's use runtime_var_end_index + 1000 as a safe default if pack_base is None, 
        # avoiding data inspection.
        if pack_base is None:
             self.pack_base = int(runtime_var_end_index + 2000)
        else:
             self.pack_base = int(pack_base)

        # Facts index
        self.fact_index = GPUFactIndex(self.facts_idx, self.pack_base)
        
        # Store packed facts for compatibility
        if self.facts_idx.numel() > 0:
            self.facts_packed = _pack_triples_64(self.facts_idx.long(), self.pack_base)
        else:
            self.facts_packed = torch.empty((0,), dtype=torch.long, device=device)

        # Pre-sort rules by predicate and build heads predicate ranges
        if self.rules_heads_idx.numel() > 0:
            order = torch.argsort(self.rules_heads_idx[:, 0], stable=True)
            self.rules_heads_sorted  = self.rules_heads_idx.index_select(0, order)
            self.rules_idx_sorted    = self.rules_idx.index_select(0, order)
            self.rule_lens_sorted    = self.rule_lens.index_select(0, order)

            preds = self.rules_heads_sorted[:, 0]
            uniq, counts = torch.unique_consecutive(preds, return_counts=True)
            starts = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.long, device=device), counts[:-1]]), dim=0)
            
            # Use predicate_no if available, otherwise we must sync to get max pred
            if self.predicate_no is not None:
                num_pred = self.predicate_no + 1 # +1 for safety or alignment
            else:
                # We MUST know the size to allocate the tensor.
                # If predicate_no is missing, we must fail or sync. 
                # Given strict instruction, let's assume predicate_no is ALWAYS passed (it is in profile_learn).
                # But as fallback, do ONE sync.
                num_pred = int(preds.max().item()) + 2
                
            self.rule_seg_starts = torch.zeros((num_pred,), dtype=torch.long, device=device)
            self.rule_seg_lens   = torch.zeros((num_pred,), dtype=torch.long, device=device)
            # Ensure indices fit
            mask = uniq < num_pred
            self.rule_seg_starts[uniq[mask]] = starts[mask]
            self.rule_seg_lens[uniq[mask]]   = counts[mask]
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
        
        # Cached arange tensors for hot paths (avoids repeated creation)
        # These are used in standardize, pack_by_owner, and other frequently called ops
        max_states_cap = max_derived_per_state if max_derived_per_state is not None else 128
        max_atoms_cap = max(self.max_rule_body_size * 3, 32) if self.max_rule_body_size else 32
        max_batch_cap = 512  # Covers typical batch sizes
        
        self._cached_arange_states = torch.arange(max_states_cap, device=device, dtype=torch.long)
        self._cached_arange_atoms = torch.arange(max_atoms_cap, device=device, dtype=torch.long)
        self._cached_arange_batch = torch.arange(max_batch_cap, device=device, dtype=torch.long)
        
        # Initialize DebugHelper for verbose output
        # -------- Debug helper --------
        self.stringifier_params = stringifier_params
        
    def _log(self, level: int, message: str) -> None:
        """Log a message if verbosity level is sufficient."""
        print(f"[ENGINE] {message}")

    def _state_to_str(self, state: Tensor) -> str:
        """Convert a tensor state to string representation."""
        if self.stringifier_params is None:
            return str(state)
        # Handle batch dimension if present
        if state.dim() == 3 and state.shape[0] == 1:
            state = state[0]
            
        return utils_funcs.state_to_str(
            state, 
            idx2predicate=self.stringifier_params.get('idx2predicate'),
            idx2constant=self.stringifier_params.get('idx2constant'),
            n_constants=self.stringifier_params.get('n_constants'),
            padding_idx=self.padding_idx
        )

    def _atom_to_str(self, atom: Tensor) -> str:
        """Convert an atom index tensor to a string representation."""
        if self.stringifier_params is None:
            return str(atom)
        return utils_funcs.atom_to_str(
             atom, 
             idx2predicate=self.stringifier_params.get('idx2predicate'),
             idx2constant=self.stringifier_params.get('idx2constant'),
             n_constants=self.stringifier_params.get('n_constants'),
             padding_idx=self.padding_idx
        )

    def print_states(self, title: str, states_tensor: torch.Tensor, 
                     counts: Optional[torch.Tensor] = None, 
                     verbose: int = 1) -> None:
        """Print states in human-readable format."""
        
        pad = self.padding_idx
        
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"{'='*60}")
        
        if states_tensor.dim() == 3:  # [B, M, 3]
            for i in range(states_tensor.shape[0]):
                state = states_tensor[i]
                valid = state[:, 0] != pad
                if valid.any():
                    atoms = state[valid]
                    atoms_str = [self._atom_to_str(atom) for atom in atoms]
                    print(f"  State {i}: [{', '.join(atoms_str)}]")
                else:
                    print(f"  State {i}: <empty>")
        elif states_tensor.dim() == 4:  # [B, K, M, 3]
            for i in range(states_tensor.shape[0]):
                count = counts[i].item() if counts is not None else states_tensor.shape[1]
                if count > 0:
                    print(f"  Batch {i} ({count} states):")
                    for j in range(min(count, states_tensor.shape[1])):
                        state = states_tensor[i, j]
                        valid = state[:, 0] != pad
                        if valid.any():
                            atoms = state[valid]
                            atoms_str = [self._atom_to_str(atom) for atom in atoms]
                            print(f"    [{j}]: [{', '.join(atoms_str)}]")
                        else:
                            print(f"    [{j}]: <empty>")
        
    def _get_arange(self, n: int, cache_type: str = 'batch') -> Tensor:
        """Get cached arange tensor, creating if needed."""
        if cache_type == 'batch':
            cache = self._cached_arange_batch
        elif cache_type == 'states':
            cache = self._cached_arange_states
        else:
            cache = self._cached_arange_atoms
            
        if n <= cache.shape[0]:
            return cache[:n]
        # Fallback for larger sizes - create and cache
        new_cache = torch.arange(n * 2, device=self.device, dtype=torch.long)
        if cache_type == 'batch':
            self._cached_arange_batch = new_cache
        elif cache_type == 'states':
            self._cached_arange_states = new_cache
        else:
            self._cached_arange_atoms = new_cache
        return new_cache[:n]

    @classmethod
    def from_index_manager(
        cls, 
        im, 
        take_ownership: bool = False, 
        stringifier_params: Optional[dict] = None,
        end_pred_idx: Optional[int] = None,
        end_proof_action: bool = False,
        max_derived_per_state: Optional[int] = None,
        deduplicate: bool = False,
        sort_states: bool = False
    ):
        """Factory method to create the engine from an IndexManager."""
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
            end_pred_idx=end_pred_idx,
            end_proof_action=end_proof_action,
            predicate_no=getattr(im, 'predicate_no', None),
            max_derived_per_state=max_derived_per_state,
            deduplicate=deduplicate,
            sort_states=sort_states
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
    def get_derived_states(
        self,
        current_states: Tensor,
        next_var_indices: Tensor,
        excluded_queries: Optional[Tensor] = None,
        verbose: int = 0
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Generate all valid successor states for the given batch of states.
        
        Steps:
        1. Preprocess: Extract active query atom from each state.
        2. Unify Rules: Match queries with rules to create new states.
        3. Unify Facts: Match queries with facts to create new states.
        4. Combine: Merge rule states and fact states.
        5. Prune & Collapse: Identify grounded proofs and prune facts from partial states.
        6. Standardize: Renumber variables for canonical representation.
        7. Pack: Group results by owner.
        8. Cap: Limit the number of successors per owner.
        
        Args:
            current_states:    Tensor [B, max_atoms, 3] containing current states.
            next_var_indices:  Tensor [B] next available variable ID per owner.
            excluded_queries:  Optional [B, max_atoms, 3] queries to exclude (circular check).
            verbose:           Verbosity level.
            
        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                - derived_states: Tensor [B, K_cap, M_new, 3] new unique states.
                - derived_counts: Tensor [B] number of valid derived states.
                - updated_vars:   Tensor [B] updated next available variable ID.
        """
        device = current_states.device
        B, max_atoms = current_states.shape[:2]
        pad = self.padding_idx
        
        if verbose > 0:
            print(f"\n[ENGINE DEBUG] get_derived_states called with next_var_indices={next_var_indices.tolist()}, constant_no={self.constant_no}")

        # Preallocate final output (padded). We'll fill progressively.
        max_atoms_out = max(max_atoms + self.max_rule_body_size, 1)
        final_states  = torch.full((B, self.max_derived_per_state, max_atoms_out, 3), pad, dtype=torch.long, device=device)
        final_counts  = torch.zeros(B, dtype=torch.long, device=device)
        updated_next  = next_var_indices.clone()

        # ---------------------------------------------------------------------
        # 1. Preprocessing: Active vs Terminal
        # ---------------------------------------------------------------------
        pre = preprocess_states(current_states, self.true_pred_idx, self.false_pred_idx, pad)
        if verbose > 0:
            self.print_states("[ENGINE] 1. CURRENT STATES", current_states)

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

        # ---------------------------------------------------------------------
        # 2. Rule Unification
        # ---------------------------------------------------------------------
        rule_states, rule_counts, rule_owners = unify_with_rules(
            self.rules_heads_sorted, self.rules_idx_sorted, self.rule_lens_sorted,
            self.rule_seg_starts, self.rule_seg_lens,
            pre.queries, pre.remaining, pre.remaining_counts, pre.preds,
            self.constant_no, pad, next_var_active
        )  # rule_states: [N_r, M_r, 3]
        
        if verbose > 0:
            self.print_states("[ENGINE] 2. RULE UNIFICATIONS", rule_states, rule_counts)

        # ---------------------------------------------------------------------
        # 3. Fact Unification
        # ---------------------------------------------------------------------
        facts_excl = excluded_queries.index_select(0, pre.active_idx) if (excluded_queries is not None) else None
        
        fact_states, fact_counts, fact_owners = unify_with_facts(
            self.facts_idx, self.predicate_range_map,
            pre.queries, pre.remaining, pre.remaining_counts, pre.preds,
            self.constant_no, pad, self.fact_index,
            excluded_queries=facts_excl
        )  # fact_states: [N_f, M_f, 3]
        
        if verbose > 0:
            self.print_states("[ENGINE] 3. FACT UNIFICATIONS", fact_states, fact_counts)

        # ---------------------------------------------------------------------
        # 4. Combine Candidates
        # ---------------------------------------------------------------------
        cand_states, cand_counts, owners, M_comb = combine_candidates(
            rule_states, rule_counts, rule_owners,
            fact_states, fact_counts, fact_owners,
            pre.active_idx, pad
        )  # cand_states: [N, M, 3], cand_counts: [N], owners: [N]

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

        # ---------------------------------------------------------------------
        # 5. Prune and Collapse (Proof Detection)
        # ---------------------------------------------------------------------
        surv_states, surv_counts, proof_mask_B, surv_owners = prune_and_collapse(
            cand_states, cand_counts, owners, self.fact_index, self.constant_no, pad, B, excl_first
        )  # surv_states: [N', M, 3], surv_counts: [N'], proof_mask_B: [B], surv_owners: [N']

        # Write TRUE for proof owners
        if self.true_atom is not None and proof_mask_B.any():
            # OPTIMIZATION: Use scatter instead of nonzero indexing
            # final_states[dst, 0, 0] = self.true_atom -> scatter_ on mask
            
            # Create update tensor for true_atom
            true_atom_tensor = self.true_atom[0]
            
            # Scatter true_atom into final_states where proof_mask_B is true
            # This avoids nonzero() sync and indexing
            # final_states[:, 0, 0] is [B]
            if proof_mask_B.any():
                final_states[:, 0, 0, 0] = torch.where(proof_mask_B, true_atom_tensor, final_states[:, 0, 0, 0])
                final_counts = torch.where(proof_mask_B, torch.tensor(1, device=device), final_counts)

        # If all active owners are proofs, we are done.
        # Check if any active owner is NOT a proof
        # active_owners_mask = zeros(B); active_owners_mask[active_idx] = 1;
        # remaining = active_owners_mask & ~proof_mask_B
        # if not remaining.any(): return...
        
        # Optimized check:
        # We need to return if no non-proof active owners remain.
        # But we also need to handle the "no survivors -> false" logic for those.
        
        nonproof_mask_B = torch.zeros(B, dtype=torch.bool, device=device)
        nonproof_mask_B[owners] = True
        nonproof_mask_B &= ~proof_mask_B
        
        if not nonproof_mask_B.any():
            return final_states, final_counts, updated_next

        # Filter survivors to only non-proof owners
        if surv_states.numel() == 0:
            # No survivors for non-proof owners -> FALSE
            if self.false_atom is not None:
                # Use boolean mask assignment instead of nonzero
                false_atom_tensor = self.false_atom[0]
                
                # Only overwrite where we haven't already marked TRUE (proof_mask_B)
                # and where we have active owners that failed (nonproof_mask_B)
                # Note: nonproof_mask_B identifies active owners that didn't generate a proof.
                # If surv_states is empty, ALL of them failed.
                
                final_states[:, 0, 0, 0] = torch.where(nonproof_mask_B, false_atom_tensor, final_states[:, 0, 0, 0])
                final_counts = torch.where(nonproof_mask_B, torch.tensor(1, device=device), final_counts)
                
            return final_states, final_counts, updated_next

        # surv_owners is already filtered to match surv_states and surv_counts
        # Just filter by valid counts
        valid_mask = surv_counts >= 0
        if valid_mask.any() and valid_mask.sum() < surv_owners.numel():
            surv_states = surv_states[valid_mask]
            surv_counts = surv_counts[valid_mask]
            surv_owners = surv_owners[valid_mask]

        # ---------------------------------------------------------------------
        # 6. Standardize Variables
        # ---------------------------------------------------------------------
        if verbose > 0:
            print(f"[ENGINE] Before standardize: next_var indices = {updated_next.tolist() if updated_next.numel() <= 10 else updated_next[:10].tolist()}")
            print(f"  constant_no = {self.constant_no}, runtime_var_end = {self.runtime_var_end_index}")
            self.print_states("[ENGINE] 5. BEFORE STANDARDIZE", surv_states, surv_counts)
        
        std_states, next_end_B = standardize_derived_states(
            surv_states, surv_counts, surv_owners, updated_next, self.constant_no,
            self.runtime_var_end_index, pad
        )  # std_states: [N', M, 3], next_end_B: [B]
        
        updated_next = torch.maximum(updated_next, next_end_B)  # [B]
        
        if verbose > 0:
            self.print_states("[ENGINE] 6. AFTER STANDARDIZE", std_states, surv_counts)

        # ---------------------------------------------------------------------
        # 7. Pack and Cap
        # ---------------------------------------------------------------------
        # Optimization: Always use max_derived_per_state as fixed K if available.
        # This completely avoids the .item() synchronization in pack_by_owner
        # We accept some padding overhead (up to K) to gain speed parity.
        # Optimization: Always use max_derived_per_state as fixed K if available.
        # This completely avoids the .item() synchronization in pack_by_owner
        # We accept some padding overhead (up to K) to gain speed parity.
        pack_K_limit = self.max_derived_per_state
        if pack_K_limit is None:
            raise ValueError("max_derived_per_state must be set")
        
        packed, packed_counts = pack_by_owner(std_states, surv_counts, surv_owners, B, M_comb, pad, K_fixed=pack_K_limit)  # [B, K, M, 3], [B]

        if self.deduplicate:
            packed, packed_counts = deduplicate_states_packed(packed, packed_counts, pad, self.hash_cache)

        # 7b) Apply canonical ordering AFTER deduplication but BEFORE capping
        should_sort = self.sort_states and packed.numel() > 0
        if should_sort:
            # Unpack to flat list for sorting (debug util only)
            flat_states = []
            flat_owners = []
            for b_idx in range(B):
                for k in range(packed_counts[b_idx].item()):
                    flat_states.append(packed[b_idx, k])
                    flat_owners.append(b_idx)
            
            if len(flat_states) > 0:
                flat_states = torch.stack(flat_states, dim=0)
                flat_owners = torch.tensor(flat_owners, dtype=torch.long, device=device)
                flat_counts = torch.ones(len(flat_states), dtype=torch.long, device=device)
                dummy_next_vars = updated_next[flat_owners]
                
                if self.stringifier_params is not None:
                    flat_states, flat_counts, flat_owners, dummy_next_vars = utils_funcs.sort_candidates_by_str_order(
                        flat_states, flat_counts, flat_owners, dummy_next_vars,
                        idx2predicate=self.stringifier_params.get('idx2predicate'),
                        idx2constant=self.stringifier_params.get('idx2constant'),
                        n_constants=self.stringifier_params.get('n_constants'),
                        padding_idx=self.padding_idx
                    )
                # Repack sorted states
                packed, packed_counts = pack_by_owner(flat_states, flat_counts, flat_owners, B, M_comb, pad)

        # Cap max states
        packed, packed_counts = cap_states_per_owner(packed, packed_counts, self.max_derived_per_state)

        # Write packed into final buffers (keeping any terminal/proof already written)
        write_mask = packed_counts > 0
        if write_mask.any():
            dst = torch.nonzero(write_mask, as_tuple=True)[0]
            # Avoid overwriting TRUE already set for proof owners
            dst = dst[~proof_mask_B.index_select(0, dst)]
            if dst.numel() > 0:
                final_states[dst, :packed.shape[1], :packed.shape[2]] = packed[dst]
                final_counts[dst] = torch.maximum(final_counts[dst], packed_counts[dst])

        # ---------------------------------------------------------------------
        # Fallback: any active owner with no derived state yet -> FALSE
        # ---------------------------------------------------------------------
        need_false = (final_counts == 0) & (torch.isin(torch.arange(B, device=device), pre.active_idx))
        need_false &= ~proof_mask_B
        if need_false.any() and self.false_atom is not None:
            idx = torch.nonzero(need_false, as_tuple=True)[0]
            final_states[idx, 0, 0] = self.false_atom
            final_counts[idx] = 1

        return final_states, final_counts, updated_next

    # ---- Utility methods for compatibility ----

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
        """Return a clone of the false """
        if self.false_tensor is None:
            raise ValueError("False predicate not defined.")
        return self.false_tensor.clone()

    def get_true_state(self) -> Tensor:
        """Return a clone of the true """
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

    def is_terminal_state(self, states: Tensor) -> Tensor:
        """
        Check if states are terminal (TRUE/FALSE/END).
        
        Args:
            states: [B, M, 3] tensor of states
            
        Returns:
            Boolean tensor [B] indicating which states are terminal
        """
        if states.numel() == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)
        
        # Get first atom of each state (first non-padding)
        non_pad = states[:, :, 0] != self.padding_idx  # [B, M]
        preds = states[:, :, 0]  # [B, M]
        
        is_terminal = torch.zeros(states.shape[0], dtype=torch.bool, device=self.device)
        
        # Check for TRUE
        if self.true_pred_idx is not None:
            has_true = ((preds == self.true_pred_idx) & non_pad).any(dim=1)
            is_terminal |= has_true
        
        # Check for FALSE
        if self.false_pred_idx is not None:
            has_false = ((preds == self.false_pred_idx) & non_pad).any(dim=1)
            is_terminal |= has_false
        
        # Check for END
        if self.end_proof_action and self.end_pred_idx is not None:
            has_end = ((preds == self.end_pred_idx) & non_pad).any(dim=1)
            is_terminal |= has_end
        
        return is_terminal

    def create_terminal_derived(self, current_states: Tensor, 
                               padding_states: int, 
                               target_atoms: int) -> Tuple[Tensor, Tensor]:
        """
        For terminal states, create derived states containing only the current terminal state.
        
        Args:
            current_states: [B, M, 3] current states
            padding_states: Maximum number of derived states per input
            target_atoms: Number of atoms per derived state
            
        Returns:
            derived_states: [B, padding_states, target_atoms, 3] with current state at position 0
            counts: [B] all ones (each terminal state has exactly 1 derived state)
        """
        B = current_states.shape[0]
        D = self.max_arity + 1
        
        derived = torch.full((B, padding_states, target_atoms, D), 
                           self.padding_idx, dtype=torch.long, device=self.device)
        
        # Copy current states to first position, handling size differences
        current = current_states
        if current.shape[1] < target_atoms:
            pad_rows = target_atoms - current.shape[1]
            pad_block = torch.full((B, pad_rows, D), self.padding_idx, 
                                  dtype=current.dtype, device=current.device)
            current = torch.cat([current, pad_block], dim=1)
        elif current.shape[1] > target_atoms:
            current = current[:, :target_atoms]

        derived[:, 0] = current  # Place current state as the only action
        counts = torch.ones(B, dtype=torch.long, device=self.device)
        
        return derived, counts