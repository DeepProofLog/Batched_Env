
import torch
from torch import Tensor
from typing import Dict, List, Optional, Tuple
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torch.nn.utils.rnn import pad_sequence

from atom_stringifier import AtomStringifier  # API parity
from unification_engine import UnificationEngine


def _safe_device(device: Optional[torch.device]) -> torch.device:
    if device is not None:
        return device
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class BatchedEnv(EnvBase):
    """
    Vectorized, GPU-first environment with:
      • Memory pruning/tabling via per-env GPU Bloom filters (loop-free membership checks)
      • Optional skip-unary closure using a *batched* bounded loop over just the unary subset
      • No Python loops in hot paths (_reset, _step, _postprocess_batched); skip-unary loop is subset-only
    """

    def __init__(
        self,
        batch_size: int,
        queries: List[Tensor],
        labels: List[int],
        query_depths: List[int],
        unification_engine: UnificationEngine,
        sampler=None,
        mode: str = 'train',
        # Idx params
        max_arity: int = 2,
        padding_idx: int = 0,
        runtime_var_start_index: int = 1,
        total_vocab_size: int = 10000,
        padding_atoms: int = 10,
        padding_states: int = 20,
        true_pred_idx: Optional[int] = None,
        false_pred_idx: Optional[int] = None,
        end_pred_idx: Optional[int] = None,
        atom_stringifier: Optional[AtomStringifier] = None,
        # Corruption params
        corruption_mode: bool = False,
        corruption_scheme: List[str] = ('head', 'tail'),
        train_neg_ratio: float = 1.0,
        # Env related params
        max_depth: int = 10,
        memory_pruning: bool = True,
        end_proof_action: bool = False,
        skip_unary_actions: bool = False,
        reward_type: int = 1,
        verbose: int = 0,
        prover_verbose: int = 0,
        device: Optional[torch.device] = None,
        # Memory pruning config (Bloom filter)
        memory_bits_pow: int = 18,     # 2**18 bits per env (32 KB) -> good default
        memory_hashes: int = 3,        # k hash functions
    ):
        # Configure device and batch size
        self.batch_size_int = int(batch_size)
        device = _safe_device(device)
        super().__init__(device=device, batch_size=torch.Size([batch_size]))

        # Core params
        self._device = device
        self.mode = mode
        self.max_depth = int(max_depth)
        self.padding_atoms = int(padding_atoms)
        self.padding_states = int(padding_states)
        self.reward_type = int(reward_type)
        self.verbose = int(verbose)
        self.prover_verbose = int(prover_verbose)

        # Index/vocab params
        self.max_arity = int(max_arity)
        self.padding_idx = int(padding_idx)
        self.true_pred_idx = true_pred_idx
        self.false_pred_idx = false_pred_idx
        self.end_pred_idx = end_pred_idx
        self.runtime_var_start_index = int(runtime_var_start_index)
        self.total_vocab_size = int(total_vocab_size)

        # External engines / helpers
        self.unification_engine = unification_engine
        self.atom_stringifier = atom_stringifier

        # Negative sampling
        self.corruption_mode = bool(corruption_mode)
        self.corruption_scheme = tuple(corruption_scheme) if corruption_scheme is not None else ('head','tail')
        self.train_neg_ratio = float(train_neg_ratio)
        self.sampler = sampler

        # Action modifiers
        self.end_proof_action = bool(end_proof_action)
        self.skip_unary_actions = bool(skip_unary_actions)
        self.max_skip_unary_iters = 20

        # End tensor (optional)
        self.end_tensor = None
        if (self.end_pred_idx is not None) and (self.end_pred_idx >= 0):
            self.end_tensor = torch.tensor(
                [[self.end_pred_idx, self.padding_idx, self.padding_idx]],
                dtype=torch.long, device=self._device
            )

        # -------- Dataset materialization (vectorized) --------
        self._all_queries_padded, self._all_first_atoms = self._materialize_queries(queries)
        self._all_labels = torch.as_tensor(labels, dtype=torch.long, device=self._device)
        self._all_depths = torch.as_tensor(query_depths, dtype=torch.long, device=self._device)
        self._num_all = int(self._all_queries_padded.shape[0])

        # Sampling pointer for eval mode
        self.counter = 0

        # -------- Runtime state tensors (GPU) --------
        arity_plus = self.max_arity + 1
        B = self.batch_size_int
        self.current_queries = torch.full(
            (B, self.padding_atoms, arity_plus), self.padding_idx, dtype=torch.long, device=self._device
        )
        self.current_labels = torch.zeros(B, dtype=torch.long, device=self._device)
        self.current_depths = torch.zeros(B, dtype=torch.long, device=self._device)
        self.next_var_indices = torch.full(
            (B,), self.runtime_var_start_index, dtype=torch.long, device=self._device
        )
        self.original_queries = torch.full(
            (B, arity_plus), self.padding_idx, dtype=torch.long, device=self._device
        )

        # Derived states [B, S, M, D]
        self.derived_states_batch = torch.full(
            (B, self.padding_states, self.padding_atoms, arity_plus),
            self.padding_idx, dtype=torch.long, device=self._device
        )
        self.derived_states_counts = torch.zeros(B, dtype=torch.long, device=self._device)

        # -------- Memory pruning (Bloom filter per env) --------
        # Disable memory pruning in eval to preserve pure evaluation unless requested
        self.memory_pruning = bool(memory_pruning) if mode == 'train' else False
        self.mem_bits_pow = int(memory_bits_pow)
        self.mem_bits = 1 << self.mem_bits_pow
        self.mem_mask = self.mem_bits - 1
        self.mem_hashes = int(memory_hashes)
        self._word_bits = 64
        self._word_mask = self._word_bits - 1
        self.mem_words = (self.mem_bits + self._word_bits - 1) // self._word_bits
        # Bitset: [B, mem_words] int64
        self._mem_bloom = torch.zeros((B, self.mem_words), dtype=torch.long, device=self._device)
        # Per-env salt to decorrelate across episodes
        self._mem_salt = torch.randint(0, (1 << 61) - 1, (B,), dtype=torch.long, device=self._device)

        # Hash support for states
        self._pack_base = self.total_vocab_size + 1  # matches IndexManager.pack_base default
        L = self.padding_atoms * (self.max_arity + 1)
        ar = torch.arange(L, device=self._device, dtype=torch.long)
        self._pos_vec1 = (ar * 0x9E3779B97F4A7C15) & ((1 << 63) - 1)  # 63-bit to keep it positive
        self._pos_vec2 = (ar * 0xC2B2AE3D27D4EB4F) & ((1 << 63) - 1)
        self._hash_idx = torch.arange(self.mem_hashes, device=self._device, dtype=torch.long)

        # Build specs
        self._make_specs()

    # ---------------------------------------------------------------------
    # Specs
    # ---------------------------------------------------------------------
    def _make_specs(self):
        from torchrl.data import Composite, Bounded

        max_vocab_size = int(self.total_vocab_size)
        B = self.batch_size_int
        A = self.padding_atoms
        D = self.max_arity + 1
        S = self.padding_states

        self.observation_spec = Composite(
            sub_index=Bounded(
                low=-1, high=max_vocab_size,
                shape=torch.Size([B, 1, A, D]), dtype=torch.int64, device=self._device,
            ),
            derived_sub_indices=Bounded(
                low=-1, high=max_vocab_size,
                shape=torch.Size([B, S, A, D]), dtype=torch.int64, device=self._device,
            ),
            action_mask=Bounded(
                low=0, high=1, shape=torch.Size([B, S]), dtype=torch.bool, device=self._device
            ),
            shape=torch.Size([B]),
        )
        self.action_spec = Bounded(
            low=0, high=self.padding_states - 1, shape=torch.Size([B]), dtype=torch.int64, device=self._device
        )
        self.reward_spec = Bounded(
            low=-float('inf'), high=float('inf'), shape=torch.Size([B, 1]), dtype=torch.float32, device=self._device
        )
        self.done_spec = Bounded(
            low=0, high=1, shape=torch.Size([B, 1]), dtype=torch.bool, device=self._device
        )
        self.truncated_spec = Bounded(
            low=0, high=1, shape=torch.Size([B, 1]), dtype=torch.bool, device=self._device
        )

    # ---------------------------------------------------------------------
    # Core Env API
    # ---------------------------------------------------------------------
    def _reset(self, tensordict: Optional[TensorDict] = None) -> TensorDict:
        """
        Vectorized reset with optional negative sampling. Resets per-env Bloom
        filters and inserts the initial state into memory when memory_pruning is on.
        """
        B = self.batch_size_int
        device = self._device
        pad = self.padding_idx
        A = self.padding_atoms
        D = self.max_arity + 1

        # Reset mask
        if tensordict is not None and "_reset" in tensordict.keys():
            reset_mask = tensordict["_reset"].squeeze(-1)
            if reset_mask.dtype != torch.bool:
                reset_mask = reset_mask.bool()
            if not reset_mask.any():
                raise ValueError("No environments to reset in partial reset.")
        else:
            reset_mask = torch.ones(B, dtype=torch.bool, device=device)

        env_idx = reset_mask.nonzero(as_tuple=True)[0]
        N = env_idx.shape[0]
        if N == 0:
            return self._create_observation()

        # Sample indices
        if self.mode == 'train':
            idxs = torch.randint(0, self._num_all, (N,), device=device)
        elif self.mode == 'eval':
            base = torch.arange(N, device=device) + int(self.counter)
            idxs = torch.remainder(base, self._num_all)
            self.counter = int((int(self.counter) + N) % self._num_all)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        # Gather batch queries/labels/depths
        batch_q = self._all_queries_padded.index_select(0, idxs)                 # [N, A, D]
        batch_labels = self._all_labels.index_select(0, idxs)                    # [N]
        batch_depths = self._all_depths.index_select(0, idxs)                    # [N]
        batch_first_atoms = self._all_first_atoms.index_select(0, idxs)          # [N, D]

        # Vectorized negative sampling
        if (self.mode == 'train') and self.corruption_mode and (self.sampler is not None) and (self.train_neg_ratio > 0):
            p_neg = float(self.train_neg_ratio) / (1.0 + float(self.train_neg_ratio))
            neg_mask = torch.rand(N, device=device) < p_neg
            if neg_mask.any():
                atoms = batch_first_atoms[neg_mask]                               # [U, D]
                corrupted = self.sampler.corrupt(atoms, num_negatives=1, device=device)
                if corrupted.dim() == 3:
                    corrupted = corrupted[:, 0]                                    # [U, D]
                neg_states = torch.full((corrupted.shape[0], A, D), pad, dtype=torch.long, device=device)
                neg_states[:, 0] = corrupted
                batch_q = batch_q.clone()
                batch_q[neg_mask] = neg_states
                batch_labels = batch_labels.clone()
                batch_labels[neg_mask] = 0

        # Write into runtime buffers
        self.current_queries.index_copy_(0, env_idx, batch_q)
        self.current_labels.index_copy_(0, env_idx, batch_labels)
        self.current_depths.index_fill_(0, env_idx, 0)
        self.next_var_indices.index_fill_(0, env_idx, self.runtime_var_start_index)

        # Reset per-env memory & add the starting state
        if self.memory_pruning:
            self._bloom_reset(env_idx)
            self._bloom_add_current(env_idx)

        # original_queries = first non-padding atom of current state
        self._update_original_queries_for_indices(env_idx.tolist())

        # Compute derived only for reset envs
        self._compute_derived_states(active_mask=reset_mask, clear_inactive=False)
        return self._create_observation()

    def _step(self, tensordict: TensorDict) -> TensorDict:
        """Vectorized step. Adds visited states to Bloom memory when enabled."""
        # OPTIMIZATION: Actions should already be on correct device/dtype from actor
        actions = tensordict["action"]
        if actions.dtype != torch.long:
            actions = actions.long()

        # Validate actions
        invalid_mask = actions >= self.derived_states_counts
        if invalid_mask.any():
            bad_a = actions[invalid_mask].tolist()
            bad_c = self.derived_states_counts[invalid_mask].tolist()
            raise ValueError(f"Invalid actions: {bad_a} with counts {bad_c}")

        # Apply
        B = self.batch_size_int
        batch_ids = torch.arange(B, device=self._device)
        self.current_queries = self.derived_states_batch[batch_ids, actions]
        self.current_depths = self.current_depths + 1

        # Add new current state to memory
        if self.memory_pruning:
            self._bloom_add_current(batch_ids)

        # Reward + done
        rewards, terminated, truncated, is_success = self._get_done_reward()
        dones = terminated | truncated

        # Next derived states for active envs
        active_mask = ~dones.squeeze(-1)
        self._compute_derived_states(active_mask=active_mask, clear_inactive=True)

        # Pack observation
        obs = self._create_observation_dict()
        obs['is_success'] = is_success.squeeze(-1)
        td = TensorDict(
            {**obs, "reward": rewards, "done": dones, "terminated": terminated, "truncated": truncated},
            batch_size=self.batch_size, device=self._device
        )
        return td

    # ---------------------------------------------------------------------
    # Derived-state computation
    # ---------------------------------------------------------------------
    def _compute_derived_states(self, active_mask: Optional[torch.Tensor] = None, clear_inactive: bool = False):
        if active_mask is None:
            active_mask = torch.ones(self.batch_size_int, dtype=torch.bool, device=self._device)

        idx = active_mask.nonzero(as_tuple=True)[0]
        batched_derived = self.derived_states_batch.clone()
        counts = self.derived_states_counts.clone()

        if idx.numel() > 0:
            all_derived, derived_counts_subset, updated_var_indices = self.unification_engine.get_derived_states(
                current_states=self.current_queries.index_select(0, idx),
                next_var_indices=self.next_var_indices.index_select(0, idx),
                excluded_queries=self.original_queries.index_select(0, idx),
                verbose=self.prover_verbose,
            )
            self.next_var_indices.index_copy_(0, idx, updated_var_indices)

            # Optional skip-unary closure on the active subset
            if self.skip_unary_actions:
                all_derived, derived_counts_subset = self._skip_unary(idx, all_derived, derived_counts_subset)

            # Vectorized postprocess (with memory pruning)
            all_derived, derived_counts_subset = self._postprocess(
                idx, all_derived, derived_counts_subset, stage="final"
            )

            # Fit to env buffers [S, A] dimensions
            A, K_sub, M_sub, D = all_derived.shape if all_derived.numel() > 0 else (idx.numel(), 0, self.padding_atoms, self.max_arity+1)
            K_full = batched_derived.shape[1]
            M_full = batched_derived.shape[2]

            # Adjust atoms dim
            if M_sub < M_full:
                pad_atoms = torch.full((A, K_sub, M_full - M_sub, D), self.padding_idx, dtype=all_derived.dtype, device=self._device)
                all_derived = torch.cat([all_derived, pad_atoms], dim=2)
            elif M_sub > M_full:
                all_derived = all_derived[:, :, :M_full]

            # Adjust K dim
            if K_sub < K_full:
                pad_states = torch.full((A, K_full - K_sub, M_full, D), self.padding_idx, dtype=all_derived.dtype, device=self._device)
                all_derived = torch.cat([all_derived, pad_states], dim=1)
            elif K_sub > K_full:
                all_derived = all_derived[:, :K_full]
                derived_counts_subset = torch.clamp(derived_counts_subset, max=K_full)

            # OPTIMIZATION: Ensure dtypes match to avoid conversion
            if all_derived.dtype != batched_derived.dtype:
                all_derived = all_derived.to(dtype=batched_derived.dtype)
            if derived_counts_subset.dtype != counts.dtype:
                derived_counts_subset = derived_counts_subset.to(dtype=counts.dtype)
                
            batched_derived.index_copy_(0, idx, all_derived)
            counts.index_copy_(0, idx, derived_counts_subset)

        if clear_inactive:
            inactive = ~active_mask
            if inactive.any():
                batched_derived[inactive] = self.padding_idx
                counts[inactive] = 0

        # Safety: any active env with zero states -> insert FALSE
        need_false = active_mask & (counts == 0)
        if need_false.any():
            false_state = self._create_false_state()                  # [M, D]
            B = need_false.sum()
            expanded = false_state.unsqueeze(0).expand(B, -1, -1)     # [B, M, D]
            dst_rows = need_false.nonzero(as_tuple=True)[0]
            batched_derived[dst_rows, 0] = expanded
            counts[dst_rows] = 1

        self.derived_states_batch = batched_derived
        self.derived_states_counts = counts

    # ---------------------------------------------------------------------
    # Skip-unary closure (batched; bounded loop on subset only)
    # ---------------------------------------------------------------------
    def _skip_unary(self, idx_subset: Tensor, derived_states: Tensor, derived_counts: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Promote unique non-terminal successors repeatedly for the *subset* of envs
        whose derived_counts == 1, up to max_skip_unary_iters. Each iteration is
        vectorized over the active subset. This preserves the "just walk through
        chains of single successors" behavior without touching non-unary envs.
        """
        if derived_states.numel() == 0:
            return derived_states, derived_counts

        pad = self.padding_idx
        A, K, M, D = derived_states.shape

        iters = 0
        # Helper: mask of unary & non-terminal
        def _unary_nonterminal(ds, dc):
            if ds.numel() == 0:
                return torch.zeros((A,), dtype=torch.bool, device=self._device)
            is_single = (dc == 1)
            fp = ds[:, 0, 0, 0]  # first predicate of the unique child
            is_term = torch.zeros_like(is_single, dtype=torch.bool)
            if self.true_pred_idx is not None:
                is_term |= (fp == self.true_pred_idx)
            if self.false_pred_idx is not None:
                is_term |= (fp == self.false_pred_idx)
            if self.end_pred_idx is not None:
                is_term |= (fp == self.end_pred_idx)
            return is_single & ~is_term

        unary_mask = _unary_nonterminal(derived_states, derived_counts)

        # Bounded iterations; each loop is a *single* batched UE call on the subset
        while unary_mask.any() and iters < self.max_skip_unary_iters:
            uidx = unary_mask.nonzero(as_tuple=True)[0]          # [U]
            # OPTIMIZATION: idx_subset should already be long dtype
            env_rows = idx_subset.index_select(0, uidx)
            if env_rows.dtype != torch.long:
                env_rows = env_rows.long()

            # Promote the single child to current state for those envs
            # OPTIMIZATION: derived_states should match current_queries dtype
            promoted = derived_states[uidx, 0]
            if promoted.dtype != self.current_queries.dtype:
                promoted = promoted.long()
            self.current_queries.index_copy_(0, env_rows, promoted)

            # Add promoted states to memory to respect tabling
            if self.memory_pruning:
                self._bloom_add_current(env_rows)

            # Re-expand only those envs
            sub_derived, sub_counts, sub_next = self.unification_engine.get_derived_states(
                current_states=self.current_queries.index_select(0, env_rows),
                next_var_indices=self.next_var_indices.index_select(0, env_rows),
                excluded_queries=self.original_queries.index_select(0, env_rows),
                verbose=self.prover_verbose,
            )
            self.next_var_indices.index_copy_(0, env_rows, sub_next)

            # Intermediate postprocess (memory pruning + budget only)
            sub_derived, sub_counts = self._postprocess(env_rows, sub_derived, sub_counts, stage="intermediate")

            # Write back into "slot" for those A-rows
            # Adjust shapes to K, M
            U, Ks, Ms, Ds = sub_derived.shape if sub_derived.numel() > 0 else (uidx.numel(), 0, M, D)

            # Pad/truncate to current K / M
            if Ms < M:
                pad_atoms = torch.full((U, Ks, M - Ms, D), pad, dtype=sub_derived.dtype, device=self._device)
                sub_derived = torch.cat([sub_derived, pad_atoms], dim=2)
            elif Ms > M:
                sub_derived = sub_derived[:, :, :M]
            if Ks < K:
                pad_states = torch.full((U, K - Ks, M, D), pad, dtype=sub_derived.dtype, device=self._device)
                sub_derived = torch.cat([sub_derived, pad_states], dim=1)
            elif Ks > K:
                sub_derived = sub_derived[:, :K]
                sub_counts = torch.clamp(sub_counts, max=K)

            # Scatter back - OPTIMIZATION: Only convert if needed
            if sub_derived.dtype != derived_states.dtype:
                sub_derived = sub_derived.to(dtype=derived_states.dtype)
            if sub_counts.dtype != derived_counts.dtype:
                sub_counts = sub_counts.to(dtype=derived_counts.dtype)
                
            derived_states.index_copy_(0, uidx, sub_derived)
            derived_counts.index_copy_(0, uidx, sub_counts)

            # Recompute unary mask
            unary_mask = _unary_nonterminal(derived_states, derived_counts)
            iters += 1

        return derived_states, derived_counts

    # ---------------------------------------------------------------------
    # Postprocess (memory pruning + compaction + truncation)
    # ---------------------------------------------------------------------
    def _postprocess(self, env_indices: Tensor, states: Tensor, counts: Tensor, stage: str = "final") -> Tuple[Tensor, Tensor]:
        """
        Vectorized successor filtering/compaction/truncation with GPU Bloom-based
        memory pruning (tabling). Environment-level memory differs from UE dedup:
        UE removes duplicates *within this step*, while memory pruning removes
        any state previously *visited* for this query. See docs in UnificationEngine. 
        """
        device = self._device
        pad = self.padding_idx

        if states.numel() == 0:
            return states, counts

        A, K, M, D = states.shape
        has_states = counts > 0

        # 1) Drop empty & beyond-count & atom budget
        valid_atom = states[:, :, :, 0] != pad                          # [A, K, M]
        state_nonempty = valid_atom.any(dim=2)                           # [A, K]
        atom_counts = valid_atom.sum(dim=2)                              # [A, K]
        within_budget = atom_counts <= self.padding_atoms                # [A, K]
        within_count = torch.arange(K, device=device).view(1, K).expand(A, K) < counts.view(A, 1)
        base_valid = state_nonempty & within_budget & within_count       # [A, K]

        # 2) Memory pruning (GPU Bloom): remove states seen before for this query
        if self.memory_pruning:
            # owners: env rows in full batch order for this A-subset
            # OPTIMIZATION: env_indices should already be long on device
            owners = env_indices
            if owners.dtype != torch.long:
                owners = owners.long()
            # Evaluate membership for all [A,K] candidates where base_valid is True
            visited = torch.zeros((A, K), dtype=torch.bool, device=device)
            if base_valid.any():
                cand = states.clone()  # [A, K, M, D]
                # Compute membership for all candidates
                visited = self._bloom_membership(cand, owners)  # [A, K]
                # Ignore invalid positions
                visited = visited & base_valid
            # Drop visited
            keep_mask = base_valid & ~visited
        else:
            keep_mask = base_valid

        # Ensure at least one state per active row (inject FALSE later if needed)
        new_counts = keep_mask.sum(dim=1)                                # [A]
        needs_false = (new_counts == 0) & has_states                     # [A]

        # 3) Compact kept states - OPTIMIZATION: Avoid .item() by clamping to reasonable max
        # Use a conservative upper bound instead of calling .item()
        max_new = torch.clamp(new_counts.max() if new_counts.numel() > 0 else torch.tensor(0, device=device), min=1, max=K)
        max_new_int = max_new.item()  # Only one .item() call instead of in loop

        pos = torch.cumsum(keep_mask.long(), dim=1) - 1                  # [A, K]
        pos = torch.clamp(pos, min=0, max=max_new_int - 1)
        batch_idx = torch.arange(A, device=device).view(A, 1).expand(A, K)
        compact = torch.full((A, max_new_int, M, D), pad, dtype=states.dtype, device=device)
        mask_flat = keep_mask
        if mask_flat.any():
            compact[batch_idx[mask_flat], pos[mask_flat]] = states[batch_idx[mask_flat], torch.arange(K, device=device).view(1, K).expand(A, K)[mask_flat]]

        counts_out = new_counts.clone()

        # 4) Inject FALSE where needed
        if needs_false.any():
            false_state = self._create_false_state()     # [M, D]
            idx = needs_false.nonzero(as_tuple=True)[0]
            compact[idx, 0] = false_state
            counts_out[idx] = 1

        # 5) Intermediate stage ends here
        if stage != "final":
            return compact, counts_out

        # 6) Final: truncate by number of states (reserve END if needed)
        reserve_end = torch.zeros(A, dtype=torch.bool, device=device)
        if self.end_proof_action:
            first_preds = compact[:, :, 0, 0]                                        # [A, max_new]
            is_term = torch.zeros_like(first_preds, dtype=torch.bool)
            if self.true_pred_idx is not None:
                is_term = is_term | (first_preds == self.true_pred_idx)
            if self.false_pred_idx is not None:
                is_term = is_term | (first_preds == self.false_pred_idx)
            is_term = is_term | (first_preds == pad)

            valid_state_mask = torch.arange(max_new, device=device).view(1, -1).expand(A, -1) < counts_out.view(A, 1)
            all_terminal = (is_term | ~valid_state_mask).all(dim=1)
            reserve_end = ~all_terminal & (counts_out > 0)

        limit = torch.full((A,), self.padding_states, dtype=torch.long, device=device)
        limit = torch.where(reserve_end, limit - 1, limit)
        limit = torch.clamp(limit, min=0)

        need_trunc = counts_out > limit
        if need_trunc.any():
            scores = (compact[:, :, :, 0] != pad).sum(dim=2)                              # [A, max_new_int]
            valid_mask = torch.arange(max_new_int, device=device).view(1, -1).expand(A, -1) < counts_out.view(A, 1)
            big = torch.full_like(scores, 10**9)
            scores_masked = torch.where(valid_mask, scores, big)

            sort_idx = torch.argsort(scores_masked, dim=1, stable=True)                  # [A, max_new_int]
            gather_idx = sort_idx.unsqueeze(2).unsqueeze(3).expand(A, max_new_int, M, D)
            sorted_states = torch.gather(compact, 1, gather_idx)

            # OPTIMIZATION: Minimize .item() calls
            Kmax = torch.clamp(limit.max(), min=1, max=self.padding_states)
            Kmax_int = Kmax.item()  # Single .item() call
            take_mask = torch.arange(max_new_int, device=device).view(1, -1).expand(A, -1) < limit.view(A, 1)
            out = torch.full((A, Kmax_int, M, D), pad, dtype=compact.dtype, device=device)
            if take_mask.any():
                row = torch.arange(A, device=device).view(A, 1).expand(A, max_new_int)[take_mask]
                col = torch.arange(max_new_int, device=device).view(1, max_new_int).expand(A, max_new_int)[take_mask]
                pos2 = torch.cumsum(take_mask.long(), dim=1) - 1
                pos2 = torch.clamp(pos2, min=0, max=Kmax_int - 1)
                out[row, pos2[take_mask]] = sorted_states[row, col]
            compact = out
            counts_out = torch.minimum(counts_out, limit)

        # 7) Add END action if configured and room available
        if self.end_proof_action:
            can_add_end = reserve_end & (counts_out < self.padding_states)
            if can_add_end.any():
                end_state = self._create_end_state()  # [M, D]
                rows = can_add_end.nonzero(as_tuple=True)[0]
                pos3 = counts_out[rows]
                compact[rows, pos3] = end_state
                counts_out[rows] = pos3 + 1

        return compact, counts_out

    # ---------------------------------------------------------------------
    # Reward / Done
    # ---------------------------------------------------------------------
    def _get_done_reward(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        B = self.batch_size_int
        device = self._device
        pad = self.padding_idx

        rewards = torch.zeros(B, 1, dtype=torch.float32, device=device)
        terminated = torch.zeros(B, 1, dtype=torch.bool, device=device)
        truncated = torch.zeros(B, 1, dtype=torch.bool, device=device)
        is_success = torch.zeros(B, 1, dtype=torch.bool, device=device)

        states = self.current_queries                                     # [B, A, D]
        non_pad = states[:, :, 0] != pad                                   # [B, A]
        preds = states[:, :, 0]                                            # [B, A]

        all_true = torch.zeros(B, dtype=torch.bool, device=device)
        any_false = torch.zeros(B, dtype=torch.bool, device=device)
        is_end = torch.zeros(B, dtype=torch.bool, device=device)

        if self.true_pred_idx is not None:
            true_mask = (preds == self.true_pred_idx) | ~non_pad
            all_true = true_mask.all(dim=1) & non_pad.any(dim=1)

        if self.false_pred_idx is not None:
            any_false = (preds == self.false_pred_idx).any(dim=1)

        if self.end_proof_action:
            single_pred = non_pad.sum(dim=1) == 1
            first_pos = non_pad.long().argmax(dim=1)
            first_pred = preds[torch.arange(B, device=device), first_pos]
            is_end = single_pred & (first_pred == self.end_pred_idx)
            all_true = all_true | is_end
            any_false = any_false & ~is_end

        depth_exceeded = self.current_depths >= self.max_depth
        natural_term = all_true | any_false
        terminated[:, 0] = natural_term
        truncated[:, 0] = depth_exceeded & ~natural_term
        done = natural_term | depth_exceeded

        success = all_true
        labels = self.current_labels

        if self.reward_type == 0:
            reward_mask = done & success & (labels == 1)
            rewards[reward_mask, 0] = 1.0
        elif self.reward_type == 1:
            tp = done & success & (labels == 1)
            fp = done & success & (labels == 0)
            rewards[tp, 0] = 1.0
            rewards[fp, 0] = -1.0
        elif self.reward_type == 2:
            tp = done & success & (labels == 1)
            tn = done & ~success & (labels == 0)
            rewards[tp, 0] = 1.0
            rewards[tn, 0] = 1.0
        elif self.reward_type == 3:
            pos = labels == 1
            neg = labels == 0
            tp = done & success & pos
            fn = done & ~success & pos
            fp = done & success & neg
            tn = done & ~success & neg
            rewards[tp, 0] = 1.0
            rewards[fn, 0] = -0.5
            rewards[fp, 0] = -1.5
            rewards[tn, 0] = 1.0
        elif self.reward_type == 4:
            pos = labels == 1
            neg = labels == 0
            tp = done & success & pos
            fn = done & ~success & pos
            fp = done & success & neg
            tn = done & ~success & neg
            rewards[tp, 0] = 1.0
            rewards[fn, 0] = -1.0
            rewards[fp, 0] = -1.0
            rewards[tn, 0] = 1.0
        else:
            raise ValueError(f"Invalid reward_type: {self.reward_type}. Choose 0-4.")

        is_success[:, 0] = success
        return rewards, terminated, truncated, is_success

    # ---------------------------------------------------------------------
    # Observation packing
    # ---------------------------------------------------------------------
    def _create_observation_dict(self) -> Dict:
        B = self.batch_size_int
        S = self.padding_states
        device = self._device
        action_indices = torch.arange(S, device=device).view(1, S).expand(B, S)
        counts = self.derived_states_counts.view(B, 1)
        action_mask = action_indices < counts

        return {
            'sub_index': self.current_queries.unsqueeze(1),
            'derived_sub_indices': self.derived_states_batch,
            'action_mask': action_mask,
            'label': self.current_labels,
            'query_depth': self.current_depths,
        }

    def _create_observation(self) -> TensorDict:
        B = self.batch_size_int
        device = self._device
        obs = self._create_observation_dict()
        obs['done'] = torch.zeros(B, 1, dtype=torch.bool, device=device)
        obs['terminated'] = torch.zeros(B, 1, dtype=torch.bool, device=device)
        obs['truncated'] = torch.zeros(B, 1, dtype=torch.bool, device=device)
        return TensorDict(obs, batch_size=self.batch_size, device=device)

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------
    def _materialize_queries(self, queries: List[Tensor]) -> Tuple[Tensor, Tensor]:
        device = self._device
        pad = self.padding_idx
        A = self.padding_atoms
        D = self.max_arity + 1

        if len(queries) == 0:
            empty_queries = torch.full((0, A, D), pad, dtype=torch.long, device=device)
            empty_atoms = torch.full((0, D), pad, dtype=torch.long, device=device)
            return empty_queries, empty_atoms

        # OPTIMIZATION: Check first query as heuristic - if it's already correct, assume all are
        # Most queries should already be in correct format from index_manager
        first_q = queries[0]
        if first_q.device != device or first_q.dtype != torch.long:
            q_batched = pad_sequence([q.to(dtype=torch.long, device=device) for q in queries], batch_first=True, padding_value=pad)
        else:
            q_batched = pad_sequence(queries, batch_first=True, padding_value=pad)
        Lmax = q_batched.shape[1]
        if Lmax >= A:
            q_batched = q_batched[:, :A]
        else:
            pad_tail = torch.full((q_batched.shape[0], A - Lmax, D), pad, dtype=torch.long, device=device)
            q_batched = torch.cat([q_batched, pad_tail], dim=1)

        valid = q_batched[:, :, 0] != pad
        has_valid = valid.any(dim=1)
        first_pos = valid.long().argmax(dim=1)
        batch_ids = torch.arange(q_batched.shape[0], device=device)
        first_atoms = q_batched[batch_ids, first_pos]
        if (~has_valid).any():
            first_atoms = first_atoms.clone()
            first_atoms[~has_valid] = torch.full((D,), pad, dtype=torch.long, device=device)

        return q_batched, first_atoms

    def _pad_state(self, state: Tensor) -> Tensor:
        device = self._device
        pad = self.padding_idx
        A = self.padding_atoms
        D = self.max_arity + 1

        # OPTIMIZATION: State should already be on correct device
        if state.device != device:
            state = state.to(device=device)
        if state.dtype != torch.long:
            state = state.long()
            
        if state.shape[0] >= A:
            return state[:A]
        pad_rows = A - state.shape[0]
        tail = torch.full((pad_rows, D), pad, dtype=torch.long, device=device)
        return torch.cat([state, tail], dim=0)

    def _create_false_state(self) -> Tensor:
        if self.false_pred_idx is None:
            raise ValueError("False predicate index is not defined.")
        state = torch.full((1, self.max_arity + 1), self.padding_idx, dtype=torch.long, device=self._device)
        state[0, 0] = int(self.false_pred_idx)
        return self._pad_state(state)

    def _create_end_state(self) -> Tensor:
        if self.end_pred_idx is None:
            raise ValueError("End predicate index is not defined.")
        state = torch.full((1, self.max_arity + 1), self.padding_idx, dtype=torch.long, device=self._device)
        state[0, 0] = int(self.end_pred_idx)
        return self._pad_state(state)

    def _update_original_queries_for_indices(self, indices: List[int]):
        if not indices:
            return
        device = self._device
        pad = self.padding_idx

        rows = torch.as_tensor(indices, dtype=torch.long, device=device)
        states = self.current_queries.index_select(0, rows)               # [N, A, D]
        masks = states[:, :, 0] != pad                                    # [N, A]
        has_valid = masks.any(dim=1)                                      # [N]
        first_pos = masks.long().argmax(dim=1)                            # [N]
        bid = torch.arange(rows.shape[0], device=device)
        first_atoms = states[bid, first_pos]                              # [N, D]
        if (~has_valid).any():
            first_atoms = first_atoms.clone()
            first_atoms[~has_valid] = torch.full((self.max_arity + 1,), pad, dtype=torch.long, device=device)
        self.original_queries.index_copy_(0, rows, first_atoms)

    # ===================== Memory (Bloom) =====================
    def _bloom_reset(self, rows: Tensor):
        """Clear bloom rows and refresh salts for selected env indices."""
        self._mem_bloom.index_fill_(0, rows, 0)
        self._mem_salt.index_copy_(0, rows, torch.randint(0, (1 << 61) - 1, (rows.shape[0],), dtype=torch.long, device=self._device))

    def _state_hash64(self, states: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Hash states (padded) into two 63-bit values via simple positional-xor mix.
        shapes:
            [N, M, D]  -> returns (h1[N], h2[N])
            [A, K, M, D] -> returns (h1[A,K], h2[A,K])
        """
        base = torch.as_tensor(self._pack_base, dtype=torch.long, device=self._device)
        if states.dim() == 3:
            N, M, D = states.shape
            s = states.long()
            packed = ((s[..., 0] * base + s[..., 1]) * base + s[..., 2]) & ((1 << 63) - 1)  # [N, M]
            pv1 = self._pos_vec1[:M].view(1, M)
            pv2 = self._pos_vec2[:M].view(1, M)
            h1 = (packed ^ pv1).sum(dim=1) & ((1 << 63) - 1)
            h2 = (packed ^ pv2).sum(dim=1) & ((1 << 63) - 1)
            return h1, h2
        elif states.dim() == 4:
            A, K, M, D = states.shape
            s = states.long()
            packed = ((s[..., 0] * base + s[..., 1]) * base + s[..., 2]) & ((1 << 63) - 1)  # [A, K, M]
            pv1 = self._pos_vec1[:M].view(1, 1, M)
            pv2 = self._pos_vec2[:M].view(1, 1, M)
            h1 = (packed ^ pv1).sum(dim=2) & ((1 << 63) - 1)                                 # [A, K]
            h2 = (packed ^ pv2).sum(dim=2) & ((1 << 63) - 1)                                 # [A, K]
            return h1, h2
        else:
            raise ValueError("states must be [N,M,3] or [A,K,M,3]")

    def _bloom_membership(self, states: Tensor, owners: Tensor) -> Tensor:
        """
        Membership test for a batch of states belonging to per-env Bloom rows.
        states: [A, K, M, D]
        owners: [A] env indices in full batch
        Returns:
            visited: [A, K] bool
        """
        A, K, M, D = states.shape
        h1, h2 = self._state_hash64(states)                               # [A, K], [A, K]
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

    def _bloom_add_current(self, rows: Tensor):
        """
        Insert current_queries[rows] into the Bloom filter for each env row.
        rows: 1D Long tensor of env indices, shape [N]
        """
        if rows.numel() == 0:
            return
        states = self.current_queries.index_select(0, rows)              # [N, M, D]
        h1, h2 = self._state_hash64(states)                              # [N], [N]
        salt = self._mem_salt.index_select(0, rows)                      # [N]
        h2s = (h2 ^ salt) & ((1 << 63) - 1)
        idxs = (h1.unsqueeze(-1) + self._hash_idx.view(1, -1) * h2s.unsqueeze(-1)) & self.mem_mask  # [N, k]
        word_idx = (idxs >> 6).long()                                    # [N, k]
        bit_off = (idxs & self._word_mask).long()                        # [N, k]
        # Build masks and OR into Bloom
        row_exp = rows.view(-1, 1).expand_as(word_idx)                   # [N, k]
        old = self._mem_bloom[row_exp, word_idx]
        self._mem_bloom[row_exp, word_idx] = old | torch.bitwise_left_shift(torch.ones_like(bit_off, dtype=torch.long), bit_off)

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
