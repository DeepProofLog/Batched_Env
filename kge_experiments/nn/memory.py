import torch
from torch import Tensor
from typing import Optional, Tuple


class BloomFilter:
    """
    Per-environment Bloom filter for memory pruning/tabling.
    """

    def __init__(
        self,
        batch_size: int,
        device: torch.device,
        memory_bits_pow: int,
        memory_hashes: int,
        padding_atoms: int,
        max_arity: int,
        total_vocab_size: int,
        padding_idx: int,
        true_pred_idx: Optional[int] = None,
        false_pred_idx: Optional[int] = None,
        end_pred_idx: Optional[int] = None,
    ):
        self.batch_size = batch_size
        self._device = device
        self.true_pred_idx = true_pred_idx
        self.false_pred_idx = false_pred_idx
        self.end_pred_idx = end_pred_idx
        self.mem_bits_pow = int(memory_bits_pow)
        self.mem_bits = 1 << self.mem_bits_pow
        self.mem_mask = self.mem_bits - 1
        self.mem_hashes = int(memory_hashes)
        self._word_bits = 64
        self._word_mask = self._word_bits - 1
        self.mem_words = (self.mem_bits + self._word_bits - 1) // self._word_bits
        # Bitset: [B, mem_words] int64
        self._mem_bloom = torch.zeros((batch_size, self.mem_words), dtype=torch.long, device=device)
        # Per-env salt to decorrelate across episodes
        self._mem_salt = torch.randint(0, (1 << 61) - 1, (batch_size,), dtype=torch.long, device=device)

        # Hash support for states
        self._pack_base = total_vocab_size + 1  # matches IndexManager.pack_base default
        L = padding_atoms * (max_arity + 1)
        # Use (ar + 1) to avoid ar[0]=0 which would make _pos_vec1[0] == _pos_vec2[0] == 0
        # This caused h1 == h2 for single-atom states, making double-hashing degenerate
        ar = torch.arange(L, device=device, dtype=torch.long) + 1
        self._pos_vec1 = (ar * 0x9E3779B97F4A7C15) & ((1 << 63) - 1)  # 63-bit to keep it positive
        self._pos_vec2 = (ar * 0xC2B2AE3D27D4EB4F) & ((1 << 63) - 1)
        self._hash_idx = torch.arange(self.mem_hashes, device=device, dtype=torch.long)
        self.padding_idx = padding_idx

    def reset(self, rows: Tensor):
        """Clear bloom rows and refresh salts for selected env indices."""
        self._mem_bloom.index_fill_(0, rows, 0)
        self._mem_salt.index_copy_(0, rows, torch.randint(0, (1 << 61) - 1, (rows.shape[0],), dtype=torch.long, device=self._device))

    def _state_hash64(self, states: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Hash states (padded) into two 63-bit values via sorted packed atoms (position-independent).
        Fully vectorized, no loops.
        shapes:
            [N, M, D]  -> returns (h1[N], h2[N])
            [A, K, M, D] -> returns (h1[A,K], h2[A,K])
        """
        base = torch.as_tensor(self._pack_base, dtype=torch.long, device=self._device)
        pad = self.padding_idx
        mask63 = (1 << 63) - 1
        invalid_value = torch.iinfo(torch.long).max
        if states.dim() == 3:
            N, M, D = states.shape
            s = states.long()
            packed = ((s[..., 0] * base + s[..., 1]) * base + s[..., 2]) & mask63  # [N, M]
            valid_mask = s[..., 0] != pad  # [N, M]
            packed_masked = torch.where(valid_mask, packed, invalid_value)
            # Optimized: No sort. Use order-independent Sum/XOR hash.
            # h(state) = sum(hash(atom))
            # We use a mixer to better distribute atom hashes before summing
            packed_mixed1 = (packed * 0x9E3779B97F4A7C15) & mask63
            packed_mixed2 = (packed * 0xC2B2AE3D27D4EB4F) & mask63
            
            h1 = torch.where(valid_mask, packed_mixed1, 0).sum(dim=1) & mask63
            h2 = torch.where(valid_mask, packed_mixed2, 0).sum(dim=1) & mask63
            return h1, h2
        elif states.dim() == 4:
            A, K, M, D = states.shape
            # Reshape to [A*K, M, D] and compute as 3D
            states_flat = states.view(A * K, M, D)
            h1_flat, h2_flat = self._state_hash64(states_flat)  # [A*K], [A*K]
            h1 = h1_flat.view(A, K)
            h2 = h2_flat.view(A, K)
            return h1, h2
        else:
            raise ValueError("states must be [N,M,3] or [A,K,M,3]")

    def membership(self, states: Tensor, owners: Tensor) -> Tensor:
        """
        Membership test for a batch of states belonging to per-env Bloom rows.
        Filters out terminal predicates to match add_current() behavior.
        states: [A, K, M, D]
        owners: [A] env indices in full batch
        Returns:
            visited: [A, K] bool
        """
        A, K, M, D = states.shape
        
        # Filter out terminal predicates (must match add_current behavior)
        # Use torch.where without cloning - creates new tensor only where needed
        pad = self.padding_idx
        preds = states[:, :, :, 0]  # [A, K, M]
        is_terminal = torch.zeros_like(preds, dtype=torch.bool)
        if self.true_pred_idx is not None:
            is_terminal = is_terminal | (preds == self.true_pred_idx)
        if self.false_pred_idx is not None:
            is_terminal = is_terminal | (preds == self.false_pred_idx)
        if self.end_pred_idx is not None:
            is_terminal = is_terminal | (preds == self.end_pred_idx)
        
        # Zero out terminal atoms by setting all components to padding
        # Expand mask to cover all D dimensions: [A, K, M] -> [A, K, M, D]
        is_terminal_expanded = is_terminal.unsqueeze(-1).expand_as(states)
        states_filtered = torch.where(is_terminal_expanded, pad, states)  # No clone needed
        
        h1, h2 = self._state_hash64(states_filtered)                      # [A, K], [A, K]
        salt = self._mem_salt.index_select(0, owners).view(A, 1)          # [A,1]
        # double hashing with salt
        h2s = (h2 ^ salt) & ((1 << 63) - 1)
        idxs = (h1.unsqueeze(-1) + self._hash_idx.view(1, 1, -1) * h2s.unsqueeze(-1)) & self.mem_mask  # [A, K, k]
        word_idx = (idxs >> 6).long()                                    # [A, K, k]
        bit_off = (idxs & self._word_mask).long()                        # [A, K, k]

        # Gather words from each owner's Bloom row
        bloom_rows = self._mem_bloom.index_select(0, owners)             # [A, W]
        words = bloom_rows.gather(1, word_idx.view(A, -1)).view(A, K, -1)  # [A, K, k]

        # Check all k bits
        mask = (torch.bitwise_and(words, torch.bitwise_left_shift(torch.ones_like(bit_off, dtype=torch.long), bit_off))) != 0           # [A, K, k]
        visited = mask.all(dim=2)                                        # [A, K]
        return visited

    def add_current(self, rows: Tensor, current_queries: Tensor):
        """
        Insert current_queries[rows] into the Bloom filter for each env row.
        Filters out terminal predicates to match ExactMemory and str_env behavior.
        rows: 1D Long tensor of env indices, shape [N]
        current_queries: [B, M, D]
        """
        if rows.numel() == 0:
            return
        
        # Check if current_queries is the full batch (indexed by env_idx) or a subset (aligned with rows)
        if current_queries.shape[0] == self.batch_size:
            states = current_queries.index_select(0, rows)  # [N, M, D] - no clone needed
        elif current_queries.shape[0] == rows.shape[0]:
            states = current_queries  # [N, M, D] - no clone needed
        else:
            raise ValueError(f"Shape mismatch: rows={rows.shape}, current_queries={current_queries.shape}, batch_size={self.batch_size}")
        
        # Filter out terminal predicates (match ExactMemory and str_env behavior)
        # Replace terminal atoms with padding to exclude them from the hash
        pad = self.padding_idx
        preds = states[:, :, 0]  # [N, M]
        is_terminal = torch.zeros_like(preds, dtype=torch.bool)
        if self.true_pred_idx is not None:
            is_terminal = is_terminal | (preds == self.true_pred_idx)
        if self.false_pred_idx is not None:
            is_terminal = is_terminal | (preds == self.false_pred_idx)
        if self.end_pred_idx is not None:
            is_terminal = is_terminal | (preds == self.end_pred_idx)
        
        # Zero out terminal atoms by setting all components to padding
        # Use torch.where for consistent behavior with membership()
        is_terminal_expanded = is_terminal.unsqueeze(-1).expand(states.shape[0], states.shape[1], states.shape[2])
        states = torch.where(is_terminal_expanded, pad, states)  # Creates new tensor, no mutation
        
        h1, h2 = self._state_hash64(states)                              # [N], [N]
        salt = self._mem_salt.index_select(0, rows)                      # [N]
        h2s = (h2 ^ salt) & ((1 << 63) - 1)
        idxs = (h1.unsqueeze(-1) + self._hash_idx.view(1, -1) * h2s.unsqueeze(-1)) & self.mem_mask  # [N, k]
        word_idx = (idxs >> 6).long()                                    # [N, k]
        bit_off = (idxs & self._word_mask).long()                        # [N, k]
        
        # DEBUG: Print first state being added
        if False and rows.numel() > 0:  # Disabled
            print(f"[BloomFilter.add_current] Adding {rows.numel()} states")
            print(f"  First state shape: {states[0].shape}")
            print(f"  First state h1={h1[0].item()}, h2={h2[0].item()}, salt={salt[0].item()}")
            print(f"  First state idxs: {idxs[0].tolist()}")
        
        # Build masks and OR into Bloom
        row_exp = rows.view(-1, 1).expand_as(word_idx)                   # [N, k]
        old = self._mem_bloom[row_exp, word_idx]
        self._mem_bloom[row_exp, word_idx] = old | torch.bitwise_left_shift(torch.ones_like(bit_off, dtype=torch.long), bit_off)

    def add_states(self, states: Tensor, owners: Tensor, valid_mask: Optional[Tensor] = None):
        """
        Insert multiple states into the Bloom filter for specified env rows.
        states: [A, K, M, D] - batch of states
        owners: [A] - env indices for each row in states
        valid_mask: [A, K] - optional mask indicating which states are valid (non-padding)
        """
        if states.numel() == 0:
            return
        
        A, K, M, D = states.shape
        if valid_mask is None:
            valid_mask = torch.ones((A, K), dtype=torch.bool, device=states.device)
        
        # Only process valid states
        if not valid_mask.any():
            return
        
        # Compute hashes for all states
        h1, h2 = self._state_hash64(states)                              # [A, K], [A, K]
        salt = self._mem_salt.index_select(0, owners).view(A, 1)         # [A, 1]
        h2s = (h2 ^ salt) & ((1 << 63) - 1)                              # [A, K]
        idxs = (h1.unsqueeze(-1) + self._hash_idx.view(1, 1, -1) * h2s.unsqueeze(-1)) & self.mem_mask  # [A, K, k]
        word_idx = (idxs >> 6).long()                                    # [A, K, k]
        bit_off = (idxs & self._word_mask).long()                        # [A, K, k]
        
        # Flatten to process only valid states
        valid_flat = valid_mask.flatten()                                # [A*K]
        if not valid_flat.any():
            return
        
        owner_exp = owners.view(A, 1).expand(A, K).flatten()             # [A*K]
        word_idx_flat = word_idx.view(A * K, -1)                         # [A*K, k]
        bit_off_flat = bit_off.view(A * K, -1)                           # [A*K, k]
        
        # Filter to valid states only
        owner_valid = owner_exp[valid_flat]                              # [V]
        word_idx_valid = word_idx_flat[valid_flat]                       # [V, k]
        bit_off_valid = bit_off_flat[valid_flat]                         # [V, k]
        
        # Expand for scatter
        owner_exp2 = owner_valid.view(-1, 1).expand_as(word_idx_valid)  # [V, k]
        old = self._mem_bloom[owner_exp2, word_idx_valid]
        self._mem_bloom[owner_exp2, word_idx_valid] = old | torch.bitwise_left_shift(
            torch.ones_like(bit_off_valid, dtype=torch.long), bit_off_valid
        )







class ExactMemory:
    """
    Exact, Python-set-based memory backend used for debugging / equivalence tests.
    Mirrors the semantics of the string environment's `_state_to_hashable`:
      - States are treated as order-independent sets of atoms.
      - Current states are added to memory with terminal predicates removed.
      - Membership for derived states is checked on the full state (including terminals).

    This backend is intentionally CPU-only and loop-based; it is only enabled
    when `use_exact_memory=True` (e.g. in tests), and the default batched
    environment still uses the GPU BloomFilter for training.
    """

    def __init__(
        self,
        batch_size: int,
        padding_idx: int,
        true_pred_idx: Optional[int],
        false_pred_idx: Optional[int],
        end_pred_idx: Optional[int],
    ):
        self.batch_size = int(batch_size)
        self.padding_idx = int(padding_idx)
        self.true_pred_idx = true_pred_idx
        self.false_pred_idx = false_pred_idx
        self.end_pred_idx = end_pred_idx
        # Per-env Python set of frozenset(atom-tuples)
        self._mem = [set() for _ in range(self.batch_size)]

    def reset(self, rows: Tensor) -> None:
        """Clear memory for selected env indices."""
        if rows.numel() == 0:
            return
        for idx in rows.view(-1).tolist():
            if 0 <= idx < self.batch_size:
                self._mem[idx] = set()

    def _state_to_key(self, state: Tensor, ignore_terminals: bool) -> frozenset:
        """
        Convert a padded tensor state [M, D] into an order-independent key.
        """
        if state.dim() != 2:
            raise ValueError("ExactMemory expects states with shape [M, D]")

        pad = self.padding_idx
        preds = state[:, 0]
        valid = preds != pad

        if ignore_terminals:
            if self.true_pred_idx is not None:
                valid = valid & (preds != self.true_pred_idx)
            if self.false_pred_idx is not None:
                valid = valid & (preds != self.false_pred_idx)
            if self.end_pred_idx is not None:
                valid = valid & (preds != self.end_pred_idx)

        if not valid.any():
            return frozenset()

        atoms = state[valid]
        tuples = [tuple(int(x) for x in atom.tolist()) for atom in atoms]
        return frozenset(tuples)

    def add_current(self, rows: Tensor, current_queries: Tensor) -> None:
        """
        Add current_queries[rows] to memory, filtering out terminal atoms
        to match the string environment's behavior.
        """
        if rows.numel() == 0:
            return
        
        rows_list = rows.view(-1).tolist()
        
        if current_queries.shape[0] == self.batch_size:
            # Full batch, indexed by env_idx
            for idx in rows_list:
                if 0 <= idx < self.batch_size:
                    state = current_queries[idx]
                    key = self._state_to_key(state, ignore_terminals=True)
                    self._mem[idx].add(key)
        elif current_queries.shape[0] == rows.shape[0]:
            # Subset, aligned with rows
            for i, idx in enumerate(rows_list):
                if 0 <= idx < self.batch_size:
                    state = current_queries[i]
                    key = self._state_to_key(state, ignore_terminals=True)
                    self._mem[idx].add(key)
        else:
             raise ValueError(f"Shape mismatch: rows={rows.shape}, current_queries={current_queries.shape}, batch_size={self.batch_size}")

    def membership(self, states: Tensor, owners: Tensor) -> Tensor:
        """
        Exact membership test.
        states: [A, K, M, D]
        owners: [A]
        Returns visited: [A, K] bool 
        """
        if states.numel() == 0:
            return torch.zeros(
                (states.shape[0], states.shape[1]),
                dtype=torch.bool,
                device=states.device,
            )

        A, K, M, D = states.shape
        visited = torch.zeros((A, K), dtype=torch.bool, device=states.device)
        pad = self.padding_idx

        owner_list = owners.view(-1).tolist()

        for a, env_idx in enumerate(owner_list):
            if not (0 <= env_idx < self.batch_size):
                continue
            mem_set = self._mem[env_idx]
            if not mem_set:
                continue

            for k in range(K):
                # Skip padded slots
                if states[a, k, 0, 0].item() == pad:
                    continue
                # CRITICAL: Must ignore_terminals=False to match sb3_env behavior
                # sb3_env checks full derived state (with terminals) against memory (which has terminals removed)
                key = self._state_to_key(states[a, k], ignore_terminals=False)
                if key in mem_set:
                    visited[a, k] = True

        return visited


class GPUExactMemory:
    """
    GPU-accelerated exact memory backend using sorted hash tables.
    
    This class provides exact membership testing (like ExactMemory) but runs entirely
    on GPU using packed 64-bit hashes and torch.searchsorted for O(log N) lookup.
    
    Key differences from ExactMemory:
    - No Python loops or .tolist() calls
    - All operations are vectorized on GPU
    - Uses sorted hash tensors per environment for exact membership
    
    Maintains semantic parity with ExactMemory:
    - States are treated as order-independent sets of atoms
    - Terminal predicates are filtered when adding states
    - Membership checks include terminals for derived state comparison
    """

    def __init__(
        self,
        batch_size: int,
        device: torch.device,
        padding_idx: int,
        padding_atoms: int,
        max_arity: int,
        total_vocab_size: int,
        true_pred_idx: Optional[int] = None,
        false_pred_idx: Optional[int] = None,
        end_pred_idx: Optional[int] = None,
        initial_capacity: int = 1024,
    ):
        """
        Initialize GPU exact memory.
        
        Args:
            batch_size: Number of environments
            device: GPU device
            padding_idx: Index used for padding
            padding_atoms: Maximum atoms per state
            max_arity: Maximum predicate arity
            total_vocab_size: Size of vocabulary (for hash base)
            true_pred_idx: Index of True predicate (terminal)
            false_pred_idx: Index of False predicate (terminal)
            end_pred_idx: Index of Endf predicate (terminal)
            initial_capacity: Initial hash table capacity per environment
        """
        self.batch_size = int(batch_size)
        self._device = device
        self.padding_idx = int(padding_idx)
        self.true_pred_idx = true_pred_idx
        self.false_pred_idx = false_pred_idx
        self.end_pred_idx = end_pred_idx
        self._capacity = initial_capacity
        
        # Hash packing base
        self._pack_base = total_vocab_size + 1
        self._mask63 = (1 << 63) - 1
        
        # Position vectors for order-independent hashing (same as BloomFilter)
        L = padding_atoms * (max_arity + 1)
        ar = torch.arange(L, device=device, dtype=torch.long) + 1
        self._pos_vec = (ar * 0x9E3779B97F4A7C15) & self._mask63
        
        # Per-environment sorted hash storage
        # Shape: [batch_size, capacity] - stores sorted hashes
        self._mem_hashes = torch.full(
            (batch_size, initial_capacity), 
            torch.iinfo(torch.long).max,  # Use max as sentinel
            dtype=torch.long, 
            device=device
        )
        # Count of valid entries per environment
        self._mem_counts = torch.zeros(batch_size, dtype=torch.long, device=device)
        
    def reset(self, rows: Tensor) -> None:
        """Clear memory for selected env indices (fully vectorized)."""
        if rows.numel() == 0:
            return
        # Reset counts and fill with sentinel
        self._mem_counts.index_fill_(0, rows, 0)
        self._mem_hashes.index_fill_(0, rows, torch.iinfo(torch.long).max)
    
    def _state_hash64(self, states: Tensor, ignore_terminals: bool) -> Tensor:
        """
        Compute order-independent 64-bit hash for states.
        
        Args:
            states: [N, M, D] or [A, K, M, D] tensor of states
            ignore_terminals: If True, exclude terminal predicates from hash
            
        Returns:
            Hash tensor of shape [N] or [A, K]
        """
        if states.dim() == 4:
            A, K, M, D = states.shape
            states_flat = states.view(A * K, M, D)
            hashes_flat = self._state_hash64(states_flat, ignore_terminals)
            return hashes_flat.view(A, K)
        
        # states: [N, M, D]
        N, M, D = states.shape
        s = states.long()
        pad = self.padding_idx
        
        # Create validity mask
        preds = s[:, :, 0]  # [N, M]
        valid = preds != pad
        
        if ignore_terminals:
            if self.true_pred_idx is not None:
                valid = valid & (preds != self.true_pred_idx)
            if self.false_pred_idx is not None:
                valid = valid & (preds != self.false_pred_idx)
            if self.end_pred_idx is not None:
                valid = valid & (preds != self.end_pred_idx)
        
        # Pack atoms: pred * base^2 + arg0 * base + arg1
        base = self._pack_base
        packed = ((s[:, :, 0] * base + s[:, :, 1]) * base + s[:, :, 2]) & self._mask63  # [N, M]
        
        # Optimized: No sort. Use order-independent Sum/XOR hash.
        # h(state) = sum(mix(atom))
        mix_const = 0x9E3779B97F4A7C15
        val = (packed * mix_const) & self._mask63
        h = torch.where(valid, val, 0).sum(dim=1) & self._mask63
        
        return h
    
    def _ensure_capacity(self, rows: Tensor, additional: int) -> None:
        """Ensure capacity for additional entries (grows if needed)."""
        max_needed = (self._mem_counts[rows] + additional).max().item()
        if max_needed > self._capacity:
            new_capacity = max(self._capacity * 2, int(max_needed * 1.5))
            new_hashes = torch.full(
                (self.batch_size, new_capacity),
                torch.iinfo(torch.long).max,
                dtype=torch.long,
                device=self._device
            )
            new_hashes[:, :self._capacity] = self._mem_hashes
            self._mem_hashes = new_hashes
            self._capacity = new_capacity
    
    def add_current(self, rows: Tensor, current_queries: Tensor) -> None:
        """
        Add current_queries[rows] to memory (fully GPU-vectorized).
        Filters out terminal atoms to match ExactMemory behavior.
        
        Args:
            rows: [N] env indices to update
            current_queries: [B, M, D] or [N, M, D] states
        """
        if rows.numel() == 0:
            return
        
        # Get states for the specified rows
        if current_queries.shape[0] == self.batch_size:
            states = current_queries.index_select(0, rows)  # [N, M, D]
        elif current_queries.shape[0] == rows.shape[0]:
            states = current_queries  # [N, M, D]
        else:
            raise ValueError(f"Shape mismatch: rows={rows.shape}, current_queries={current_queries.shape}")
        
        # Compute hashes (ignoring terminals)
        hashes = self._state_hash64(states, ignore_terminals=True)  # [N]
        
        # Check for uniqueness of rows (common case optimization)
        # If all rows are unique, we can perform a fully vectorized batched update
        # This is strictly faster than looping over unique_rows
        if rows.shape[0] == rows.unique().shape[0]:
            self._add_current_unique(rows, hashes)
        else:
            # Fallback for duplicate rows (rare in this codebase)
            unique_rows, inverse = torch.unique(rows, return_inverse=True)
            for env_idx in unique_rows:
                mask = (rows == env_idx)
                new_hashes = hashes[mask]
                self._update_single_env(env_idx, new_hashes)

    def _add_current_unique(self, rows: Tensor, new_hashes: Tensor) -> None:
        """
        Vectorized update for unique environment rows.
        Concatenates new hashes to existing ones, sorts, and dedups in parallel.
        """
        # 1. Gather existing hashes: [N, Capacity]
        existing = self._mem_hashes[rows]
        
        # 2. Append new hashes: [N, Capacity + 1]
        # new_hashes is [N], unsqueeze to [N, 1]
        combined = torch.cat([existing, new_hashes.unsqueeze(1)], dim=1)
        
        # 3. Sort to bring duplicates together and sentinels to end: [N, Capacity + 1]
        sorted_hashes, _ = torch.sort(combined, dim=1)
        
        # 4. Identify unique values (valid and not duplicate of previous)
        is_duplicate = torch.zeros_like(sorted_hashes, dtype=torch.bool)
        is_duplicate[:, 1:] = sorted_hashes[:, 1:] == sorted_hashes[:, :-1]
        
        is_sentinel = sorted_hashes == torch.iinfo(torch.long).max
        keep_mask = (~is_duplicate) & (~is_sentinel)
        
        # 5. Calculate new counts
        new_counts = keep_mask.sum(dim=1)
        
        # 6. Check capacity and resize if needed
        max_needed = new_counts.max().item()
        if max_needed > self._capacity:
            self._ensure_capacity(rows, max_needed - self._mem_counts[rows].min().item())
            # Re-gather existing because _mem_hashes changed size (and address?)
            # NOTE: _mem_hashes is updated in _ensure_capacity.
            # We need to recreate the `combined` tensor or map logic to new capacity.
            # Simpler: just recurse? No.
            # Re-do step 1-3 with new capacity?
            # Existing has grown, but data is same.
            # We can't use `sorted_hashes` directly if we need to write to a larger buffer.
            # BUT: update_buffer creation below depends on self._capacity.
            # So as long as we use updated self._capacity, we are fine.
        
        # 7. Compaction: Write kept values to update buffer
        batch_idx = torch.arange(rows.shape[0], device=self._device).unsqueeze(1).expand_as(sorted_hashes)
        scatter_pos = torch.cumsum(keep_mask.long(), dim=1) - 1
        
        # Select validated locations
        target_rows = batch_idx[keep_mask]
        target_cols = scatter_pos[keep_mask]
        
        # Create a clean buffer [N, Capacity]
        update_buffer = torch.full(
            (rows.shape[0], self._capacity), 
            torch.iinfo(torch.long).max, 
            dtype=torch.long, 
            device=self._device
        )
        
        # Write to update_buffer
        update_buffer[target_rows, target_cols] = sorted_hashes[keep_mask]
        
        # 8. Write back to global memory
        self._mem_hashes[rows] = update_buffer
        self._mem_counts[rows] = new_counts

    def _update_single_env(self, env_idx: Tensor, new_hashes: Tensor) -> None:
        """Legacy/Fallback update for a single environment id."""
        count = self._mem_counts[env_idx]
        existing = self._mem_hashes[env_idx, :count]
        combined = torch.cat([existing, new_hashes])
        unique_combined = torch.unique(combined)
        unique_combined = unique_combined[unique_combined != torch.iinfo(torch.long).max]
        new_count = unique_combined.shape[0]
        
        if new_count > self._capacity:
            self._ensure_capacity(env_idx.unsqueeze(0), new_count - count)
            
        self._mem_hashes[env_idx, :new_count] = unique_combined
        if new_count < self._capacity:
            self._mem_hashes[env_idx, new_count:] = torch.iinfo(torch.long).max
        self._mem_counts[env_idx] = new_count
    
    def add_current_batched(self, rows: Tensor, current_queries: Tensor) -> None:
        """
        Batched version of add_current (more efficient for many insertions).
        Uses batch-level operations where possible.
        """
        if rows.numel() == 0:
            return
        
        # Get states for the specified rows
        if current_queries.shape[0] == self.batch_size:
            states = current_queries.index_select(0, rows)
        elif current_queries.shape[0] == rows.shape[0]:
            states = current_queries
        else:
            raise ValueError(f"Shape mismatch")
        
        N = rows.shape[0]
        hashes = self._state_hash64(states, ignore_terminals=True)  # [N]
        
        self._ensure_capacity(rows, 1)
        
        # For efficiency, update unique rows in batch
        unique_rows, inverse = torch.unique(rows, return_inverse=True)
        
        for i, env_idx in enumerate(unique_rows):
            # Get all hashes for this env
            mask = (rows == env_idx)
            new_hashes = hashes[mask]
            
            count = self._mem_counts[env_idx].item()
            existing = self._mem_hashes[env_idx, :count]
            
            # Merge: combine existing and new, unique, sort
            combined = torch.cat([existing, new_hashes])
            unique_combined = torch.unique(combined)
            unique_combined = unique_combined[unique_combined != torch.iinfo(torch.long).max]
            
            new_count = unique_combined.shape[0]
            if new_count > self._capacity:
                self._ensure_capacity(env_idx.unsqueeze(0), new_count - count)
            
            self._mem_hashes[env_idx, :new_count] = unique_combined
            self._mem_hashes[env_idx, new_count:] = torch.iinfo(torch.long).max
            self._mem_counts[env_idx] = new_count
    
    def membership(self, states: Tensor, owners: Tensor) -> Tensor:
        """
        GPU-vectorized exact membership test.
        
        Args:
            states: [A, K, M, D] batch of derived states
            owners: [A] env indices for each row
            
        Returns:
            visited: [A, K] bool tensor
        """
        if states.numel() == 0:
            return torch.zeros(
                (states.shape[0], states.shape[1]),
                dtype=torch.bool,
                device=states.device,
            )
        
        A, K, M, D = states.shape
        
        # Compute hashes for all states (NOT ignoring terminals for membership check)
        hashes = self._state_hash64(states, ignore_terminals=False)  # [A, K]
        
        # Get counts for each owner
        counts = self._mem_counts[owners]  # [A]
        
        # Initialize result
        visited = torch.zeros((A, K), dtype=torch.bool, device=states.device)
        
        # Check padding (skip if first atom is padding)
        is_padding = states[:, :, 0, 0] == self.padding_idx  # [A, K]
        
        # For each unique owner, do batch membership check
        unique_owners = torch.unique(owners)
        
        for env_idx in unique_owners:
            owner_mask = (owners == env_idx)  # [A]
            if not owner_mask.any():
                continue
            
            count = self._mem_counts[env_idx]
            if count == 0:
                continue
            
            mem_hashes = self._mem_hashes[env_idx, :count]  # [count]
            
            # Get hashes for this owner's states
            owner_indices = torch.where(owner_mask)[0]  # indices in A
            owner_hashes = hashes[owner_mask]  # [num_owner, K]
            
            # Flatten for searchsorted
            owner_hashes_flat = owner_hashes.flatten()  # [num_owner * K]
            
            # Binary search
            pos = torch.searchsorted(mem_hashes, owner_hashes_flat)  # [num_owner * K]
            pos = pos.clamp(max=count - 1)  # Clamp to valid range
            
            # Check if found
            found = (mem_hashes[pos] == owner_hashes_flat)  # [num_owner * K]
            found = found.view(-1, K)  # [num_owner, K]
            
            # Update visited for this owner
            visited[owner_mask] = found
        
        # Don't mark padding as visited
        visited = visited & ~is_padding
        
        return visited