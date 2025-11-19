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
        ar = torch.arange(L, device=device, dtype=torch.long)
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
            packed_sorted, _ = torch.sort(packed_masked, dim=1)  # [N, M]
            num_valid = valid_mask.sum(dim=1)  # [N]
            valid_sorted_mask = torch.arange(M, device=self._device).unsqueeze(0) < num_valid.unsqueeze(1)  # [N, M]
            pv1_exp = self._pos_vec1[:M].unsqueeze(0).expand(N, -1)  # [N, M]
            pv2_exp = self._pos_vec2[:M].unsqueeze(0).expand(N, -1)  # [N, M]
            h1 = torch.where(valid_sorted_mask, packed_sorted ^ pv1_exp, 0).sum(dim=1) & mask63
            h2 = torch.where(valid_sorted_mask, packed_sorted ^ pv2_exp, 0).sum(dim=1) & mask63
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
        states_filtered = states.clone()
        pad = self.padding_idx
        preds = states_filtered[:, :, :, 0]  # [A, K, M]
        is_terminal = torch.zeros_like(preds, dtype=torch.bool)
        if self.true_pred_idx is not None:
            is_terminal |= (preds == self.true_pred_idx)
        if self.false_pred_idx is not None:
            is_terminal |= (preds == self.false_pred_idx)
        if self.end_pred_idx is not None:
            is_terminal |= (preds == self.end_pred_idx)
        
        # Zero out terminal atoms by setting all components to padding
        # Expand mask to cover all D dimensions: [A, K, M] -> [A, K, M, D]
        is_terminal_expanded = is_terminal.unsqueeze(-1).expand_as(states_filtered)
        states_filtered = torch.where(is_terminal_expanded, pad, states_filtered)
        
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
        states = current_queries.index_select(0, rows).clone()      # [N, M, D]
        
        # Filter out terminal predicates (match ExactMemory and str_env behavior)
        # Replace terminal atoms with padding to exclude them from the hash
        pad = self.padding_idx
        preds = states[:, :, 0]  # [N, M]
        is_terminal = torch.zeros_like(preds, dtype=torch.bool)
        if self.true_pred_idx is not None:
            is_terminal |= (preds == self.true_pred_idx)
        if self.false_pred_idx is not None:
            is_terminal |= (preds == self.false_pred_idx)
        if self.end_pred_idx is not None:
            is_terminal |= (preds == self.end_pred_idx)
        
        # Zero out terminal atoms by setting all components to padding
        # Use torch.where for consistent behavior with membership()
        is_terminal_expanded = is_terminal.unsqueeze(-1).expand(states.shape[0], states.shape[1], states.shape[2])
        states = torch.where(is_terminal_expanded, pad, states)
        
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
        for idx in rows.view(-1).tolist():
            if 0 <= idx < self.batch_size:
                state = current_queries[idx]
                key = self._state_to_key(state, ignore_terminals=True)
                self._mem[idx].add(key)

    def membership(self, states: Tensor, owners: Tensor) -> Tensor:
        """
        Exact membership test.
        states: [A, K, M, D]
        owners: [A]
        Returns visited: [A, K] bool tensor.
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
                # CRITICAL: Must ignore_terminals=True to match how states are added to memory
                # This ensures consistency between add_current() and membership()
                key = self._state_to_key(states[a, k], ignore_terminals=True)
                if key in mem_set:
                    visited[a, k] = True

        return visited