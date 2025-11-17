
import torch
from torch import Tensor
from typing import Dict, List, Optional, Tuple
from tensordict import TensorDict
from torchrl.envs import EnvBase

from unification_engine import UnificationEngine
from bloom_filter import BloomFilter
from debug_helper import DebugHelper


def _safe_device(device: Optional[torch.device]) -> torch.device:
    if device is not None:
        return device
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


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
                key = self._state_to_key(states[a, k], ignore_terminals=False)
                if key in mem_set:
                    visited[a, k] = True

        return visited


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
        queries: Tensor,
        labels: Tensor,
        query_depths: Tensor,
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
        stringifier_params: Optional[dict] = None,
        # Corruption params
        corruption_mode: bool = False,
        corruption_scheme: List[str] = ('head', 'tail'),
        train_neg_ratio: float = 1.0,
        # Env related params
        max_depth: int = 10,
        memory_pruning: bool = True,
        eval_pruning: bool = False,
        end_proof_action: bool = False,
        skip_unary_actions: bool = False,
        reward_type: int = 0,
        verbose: int = 0,
        prover_verbose: int = 0,
        device: Optional[torch.device] = None,
        # Memory pruning config (Bloom filter)
        memory_bits_pow: int = 18,     # 2**18 bits per env (32 KB) -> good default
        memory_hashes: int = 3,        # k hash functions
        use_exact_memory: bool = False,
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
        self._debug_prefix = "[BatchedEnv]"
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
        self.stringifier_params = stringifier_params
        self._last_reset_rows = None

        # Negative sampling
        self.corruption_mode = bool(corruption_mode)
        self.corruption_scheme = tuple(corruption_scheme) if corruption_scheme is not None else ('head','tail')
        self.train_neg_ratio = float(train_neg_ratio)
        self.sampler = sampler

        # Action modifiers
        self.end_proof_action = bool(end_proof_action)
        self.skip_unary_actions = bool(skip_unary_actions)
        self.max_skip_unary_iters = 20

        # Validate end_proof_action configuration
        if self.end_proof_action:
            if self.end_pred_idx is None:
                raise ValueError("end_pred_idx must be provided when end_proof_action is enabled.")
            if self.false_pred_idx is None:
                raise ValueError("false_pred_idx must be provided when end_proof_action is enabled.")

        # End tensor (optional)
        self.end_tensor = None
        if (self.end_pred_idx is not None) and (self.end_pred_idx >= 0):
            self.end_tensor = torch.tensor(
                [[self.end_pred_idx, self.padding_idx, self.padding_idx]],
                dtype=torch.long,
                device=self._device,
            )

        # -------- Dataset tensors (pre-materialized) --------
        self._all_queries_padded = self._ensure_query_tensor(queries)
        self._all_first_atoms = self._compute_first_atoms(self._all_queries_padded)
        self._all_labels = self._ensure_vector(labels, self._all_queries_padded.shape[0], "labels")
        self._all_depths = self._ensure_vector(query_depths, self._all_queries_padded.shape[0], "query_depths")
        self._num_all = int(self._all_queries_padded.shape[0])

        # Sampling pointer for eval mode
        self.counter = 0

        # -------- Runtime state tensors (GPU) --------
        # B: batch size, A: atoms size, D: arity+1
        arity_plus = self.max_arity + 1
        B = self.batch_size_int
        self.current_queries = torch.full(
            (B, self.padding_atoms, arity_plus),
            self.padding_idx,
            dtype=torch.long,
            device=self._device,
        )
        self.current_labels = torch.zeros(B, dtype=torch.long, device=self._device)
        self.current_depths = torch.zeros(B, dtype=torch.long, device=self._device)
        self.proof_depths = torch.zeros(B, dtype=torch.long, device=self._device)
        self.next_var_indices = torch.full(
            (B,), self.runtime_var_start_index, dtype=torch.long, device=self._device
        )
        self.original_queries = torch.full(
            (B, arity_plus), self.padding_idx, dtype=torch.long, device=self._device
        )

        # Derived states [B, S, A, D]
        self.derived_states_batch = torch.full(
            (B, self.padding_states, self.padding_atoms, arity_plus),
            self.padding_idx,
            dtype=torch.long,
            device=self._device,
        )
        self.derived_states_counts = torch.zeros(B, dtype=torch.long, device=self._device)

        # -------- Memory pruning (Bloom filter per env) --------
        self.eval_pruning = bool(eval_pruning)
        self.memory_pruning = bool(memory_pruning) if mode == 'train' or self.eval_pruning else False
        self.use_exact_memory = bool(use_exact_memory)
        if self.use_exact_memory:
            self.memory_backend = ExactMemory(
                batch_size=batch_size,
                padding_idx=self.padding_idx,
                true_pred_idx=self.true_pred_idx,
                false_pred_idx=self.false_pred_idx,
                end_pred_idx=self.end_pred_idx,
            )
        else:
            self.memory_backend = BloomFilter(
                batch_size=batch_size,
                device=self._device,
                memory_bits_pow=memory_bits_pow,
                memory_hashes=memory_hashes,
                padding_atoms=self.padding_atoms,
                max_arity=self.max_arity,
                total_vocab_size=self.total_vocab_size,
                padding_idx=self.padding_idx,
            )

        # -------- Debug helper --------
        debug_helper_params = {}
        if self.stringifier_params is not None:
            debug_helper_params = self.stringifier_params.copy()

        self.debug_helper = DebugHelper(
            verbose=self.verbose,
            debug_prefix=self._debug_prefix,
            batch_size=self.batch_size_int,
            device=self._device,
            padding_idx=self.padding_idx,
            **debug_helper_params,
        )

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
                low=-1,
                high=max_vocab_size,
                shape=torch.Size([B, 1, A, D]),
                dtype=torch.int64,
                device=self._device,
            ),
            derived_sub_indices=Bounded(
                low=-1,
                high=max_vocab_size,
                shape=torch.Size([B, S, A, D]),
                dtype=torch.int64,
                device=self._device,
            ),
            action_mask=Bounded(
                low=0,
                high=1,
                shape=torch.Size([B, S]),
                dtype=torch.bool,
                device=self._device,
            ),
            shape=torch.Size([B]),
        )
        self.action_spec = Bounded(
            low=0,
            high=self.padding_states - 1,
            shape=torch.Size([B]),
            dtype=torch.int64,
            device=self._device,
        )
        self.reward_spec = Bounded(
            low=-float('inf'),
            high=float('inf'),
            shape=torch.Size([B, 1]),
            dtype=torch.float32,
            device=self._device,
        )
        self.done_spec = Bounded(
            low=0,
            high=1,
            shape=torch.Size([B, 1]),
            dtype=torch.bool,
            device=self._device,
        )
        self.truncated_spec = Bounded(
            low=0,
            high=1,
            shape=torch.Size([B, 1]),
            dtype=torch.bool,
            device=self._device,
        )

    def _reset(self, tensordict: Optional[TensorDict] = None) -> TensorDict:
        device = self._device
        pad = self.padding_idx
        B = self.batch_size_int
        A = self.padding_atoms
        D = self.max_arity + 1

        if self.verbose >= 3:
            if tensordict is not None:
                self.debug_helper._log(3, f"[Reset]  td info:")
                for key in tensordict.keys():
                    self.debug_helper._log(3, f"[Reset]  Key: {key}, Shape: {tensordict[key].shape}, Dtype: {tensordict[key].dtype}, Values: {tensordict[key]}")
            else:
                self.debug_helper._log(3, "[Reset]  No tensordict provided.")

        # Reset mask
        if tensordict is not None:
            if "_reset" not in tensordict.keys():
                raise ValueError("_reset key not found in tensordict for partial reset.")
            reset_mask = tensordict["_reset"].squeeze(-1)  # [B] bool
            assert reset_mask.dtype == torch.bool, "_reset mask must be of boolean dtype"
            env_idx = torch.arange(reset_mask.shape[0], device=device)[reset_mask]  # N Indices to reset
            assert env_idx.numel() > 0, "No environments to reset in partial reset."
            partial_reset = True
        else:
            reset_mask = torch.ones(B, dtype=torch.bool, device=device)
            env_idx = torch.arange(B, device=device)
            partial_reset = False

        if self.verbose >= 1:
            # if there are envs not being reset, log them
            if not reset_mask.shape[0] == B:
                self.debug_helper._log(2, f"[Reset] Not resetting envs: {[(i.item()) for i in torch.arange(B, device=device) if i not in env_idx]}")

        assert env_idx.shape[0] > 0, "No environments to reset."
        N = env_idx.shape[0]

        # Sample indices
        if self.mode == 'train':
            if N <= self._num_all:
                # Sample without replacement when we have enough queries
                perm = torch.randperm(self._num_all, device=device)
                idxs = perm[:N]
            else:
                # Sample with replacement when batch size exceeds dataset size
                idxs = torch.randint(0, self._num_all, (N,), device=device)
        elif self.mode == 'eval':
            assert hasattr(self, "_eval_slot_starts") and self._eval_slot_starts is not None and \
                   hasattr(self, "_eval_slot_lengths") and self._eval_slot_lengths is not None and \
                   hasattr(self, "_eval_slot_ptr") and self._eval_slot_ptr is not None, \
                "Evaluation slots not initialized. Call set_eval_dataset with per_slot_lengths."
            # Per-slot schedule: pick next item for each env row independently
            try:
                env_idx, N, idxs = self._eval_sampling(env_idx)
            except RuntimeError:
                return self._create_observation()
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        # Gather batch queries/labels/depths
        batch_q = self._all_queries_padded.index_select(0, idxs)  # [N, A, D]
        batch_labels = self._all_labels.index_select(0, idxs)  # [N]
        proof_depths = self._all_depths.index_select(0, idxs)  # [N]
        batch_first_atoms = self._all_first_atoms.index_select(0, idxs)  # [N, D]

        # Vectorized negative sampling
        batch_q, batch_labels, proof_depths = self.sample_negatives(N, batch_q, batch_labels, proof_depths, batch_first_atoms, device, pad, A, D)

        # Write into runtime buffers
        self.current_queries.index_copy_(0, env_idx, batch_q)
        self.current_labels.index_copy_(0, env_idx, batch_labels)
        self.proof_depths.index_copy_(0, env_idx, proof_depths)
        # the initial depth are 0 for all reset envs
        self.current_depths.index_fill_(0, env_idx, 0)
        self.next_var_indices.index_fill_(0, env_idx, self.runtime_var_start_index)

        # Reset per-env memory (current state will be added to memory in _postprocess)
        if self.memory_pruning:
            self.memory_backend.reset(env_idx)

        # original_queries = first non-padding atom of current state
        self._update_original_queries_for_indices(env_idx)

        # Apply skip-unary closure to current states (string-env style) when using exact memory.
        if self.skip_unary_actions and self.use_exact_memory:
            self._apply_skip_unary_to_current_state(env_idx)

        # Compute derived only for reset envs
        # Create a mask that only includes the environments that were actually reset (env_idx)
        actual_reset_mask = torch.zeros(B, dtype=torch.bool, device=device)
        actual_reset_mask[env_idx] = True

        self._compute_derived_states(active_mask=actual_reset_mask, clear_inactive=False)

        # IMPORTANT: Initialize inactive slots AFTER _compute_derived_states to avoid being overwritten
        if self.mode == 'eval':
            # For inactive slots in the original reset_mask, set them to a terminal state
            # This needs to happen AFTER _compute_derived_states so it doesn't get overwritten
            if reset_mask is not None:
                original_reset_idx = torch.arange(reset_mask.shape[0], device=device)[reset_mask]
            else:
                original_reset_idx = torch.arange(B, device=device)

            lengths = self._eval_slot_lengths
            inactive_in_reset = original_reset_idx[lengths[original_reset_idx] == 0]
            if inactive_in_reset.numel() > 0:
                # Set inactive slots to terminated state (FALSE + done=True)
                false_state_full = self.unification_engine.get_false_state()  # [A, D] padded
                false_atom = false_state_full[0]  # Just the first atom [D]
                false_queries = torch.full((inactive_in_reset.shape[0], A, D), pad, dtype=torch.long, device=device)
                false_queries[:, 0] = false_atom
                self.current_queries.index_copy_(0, inactive_in_reset, false_queries)
                self.current_labels.index_fill_(0, inactive_in_reset, 0)
                self.current_depths.index_fill_(0, inactive_in_reset, self.max_depth + 1)  # Exceed max_depth to ensure done
                self.proof_depths.index_fill_(0, inactive_in_reset, -1)  # No proof depth for terminated states
                self.next_var_indices.index_fill_(0, inactive_in_reset, self.runtime_var_start_index)

                # CRITICAL: Also set derived states for inactive slots to have at least one action
                # Otherwise they'll trigger "all actions masked" warnings
                # Create a dummy FALSE derived state
                inactive_count = inactive_in_reset.shape[0]
                S = self.padding_states
                dummy_derived = torch.full((inactive_count, S, A, D), pad, dtype=torch.long, device=device)
                dummy_derived[:, 0, 0] = false_atom  # First action is FALSE
                self.derived_states_batch.index_copy_(0, inactive_in_reset, dummy_derived)
                self.derived_states_counts.index_fill_(0, inactive_in_reset, 1)  # Only 1 action available
        if self.verbose >= 1:
            obs_dict = self._create_observation_dict()
            mss = "[partial reset] After: " if partial_reset else "[reset] After: "
            self.debug_helper._dump_states(
                mss, self.current_queries, self.derived_states_batch, self.derived_states_counts,
                current_depths=self.current_depths, proof_depths=self.proof_depths, current_labels=self.current_labels,
                rows=env_idx, level=1, action_mask=obs_dict.get('action_mask')
            )
        return self._create_observation()

    def sample_negatives(self, N, batch_q, batch_labels, proof_depths, batch_first_atoms, device, pad, A, D):
        """
        Vectorized negative sampling with duplicate avoidance.
        """
        if (self.mode == 'train') and self.corruption_mode and (self.train_neg_ratio > 0):
            p_neg = float(self.train_neg_ratio) / (1.0 + float(self.train_neg_ratio))
            neg_mask = torch.rand(N, device=device) < p_neg
            neg_rows = torch.arange(N, device=device)[neg_mask]
            if neg_rows.numel() > 0:
                atoms = batch_first_atoms.index_select(0, neg_rows)               # [U, D]
                corrupted = self.sampler.corrupt(atoms, num_negatives=1, device=device)
                if corrupted.dim() == 3:
                    corrupted = corrupted[:, 0]                                    # [U, D]
                neg_states = torch.full((neg_rows.shape[0], A, D), pad, dtype=torch.long, device=device)
                neg_states[:, 0] = corrupted
                batch_q = batch_q.clone()
                batch_q.index_copy_(0, neg_rows, neg_states)
                batch_labels = batch_labels.clone()
                batch_labels.index_fill_(0, neg_rows, 0)
                # define the depths as -1 for corrupted states
                proof_depths = proof_depths.clone()
                proof_depths.index_fill_(0, neg_rows, -1)

                # Ensure no duplicate negatives (vectorized)
                flattened = batch_q.view(N, -1)
                equal_matrix = (flattened.unsqueeze(0) == flattened.unsqueeze(1)).all(dim=2)  # [N, N]
                duplicate_mask = equal_matrix.sum(dim=1) > 1  # [N]
                duplicate_neg_mask = duplicate_mask[neg_rows]
                duplicate_neg_rows = neg_rows[duplicate_neg_mask]
                if duplicate_neg_rows.numel() > 0:
                    atoms_to_resample = batch_first_atoms.index_select(0, duplicate_neg_rows)
                    new_corrupted = self.sampler.corrupt(atoms_to_resample, num_negatives=1, device=device)
                    if new_corrupted.dim() == 3:
                        new_corrupted = new_corrupted[:, 0]
                    new_neg_states = torch.full((duplicate_neg_rows.shape[0], A, D), pad, dtype=torch.long, device=device)
                    new_neg_states[:, 0] = new_corrupted
                    batch_q.index_copy_(0, duplicate_neg_rows, new_neg_states)

        return batch_q, batch_labels, proof_depths

    def _step(self, tensordict: TensorDict) -> TensorDict:
        """Vectorized step. Adds visited states to Bloom memory when enabled."""
        
        actions = tensordict["action"]
        if actions.dtype != torch.long:
            actions = actions.long()

        counts = self.derived_states_counts
        active_mask = counts > 0

        # Validate actions only for rows that have available successors
        invalid_mask = (actions >= counts) & active_mask
        invalid_idx = torch.arange(actions.shape[0], device=self._device)[invalid_mask]
        if invalid_idx.numel() > 0:
            raise ValueError(
                f"Invalid actions for envs {invalid_idx}: "
                f"actions={actions[invalid_idx]}, counts={self.derived_states_counts[invalid_idx]}"
            )

        # Increase depth for active envs
        B = self.batch_size_int
        batch_ids = torch.arange(B, device=self._device)
        if active_mask.any():
            active_rows = batch_ids[active_mask]
            chosen = self.derived_states_batch[active_rows, actions[active_rows]]
            self.current_queries.index_copy_(0, active_rows, chosen)

            # Optional string-env style skip-unary: follow unary chains on the
            # newly selected current states before expanding successors.
            if self.skip_unary_actions and self.use_exact_memory:
                self._apply_skip_unary_to_current_state(active_rows)

        self.current_depths = self.current_depths + active_mask.long()

        # Note: Current state will be added to memory in _postprocess during _compute_derived_states

        # Compute derived states (and apply skip_unary if enabled)
        # This must be done BEFORE checking termination because skip_unary may reach a terminal state
        active_mask_for_derived = active_mask.clone()
        
        # In eval mode, don't clear inactive (done) envs - they need to keep their terminal state with valid actions
        # In train mode, clearing is OK because envs will be reset immediately
        should_clear_inactive = (self.mode == 'train')
        self._compute_derived_states(active_mask=active_mask_for_derived, clear_inactive=should_clear_inactive)

        # NOW check termination AFTER skip_unary has been applied
        if self.verbose >= 2:
            self.debug_helper._log(2, f"[_step] Before _get_done_reward, current_queries[0,0]: {self.current_queries[0, 0]}")
        rewards, terminated, truncated, is_success = self._get_done_reward()
        dones = terminated | truncated
        if self.verbose >= 2:
            self.debug_helper._log(2, f"[_step] After _get_done_reward: terminated={terminated}, truncated={truncated}, dones={dones}")

        # Active mask for observation packing (envs that are NOT done after termination check)
        active_mask = ~dones.squeeze(-1)
        
        # Derived states already computed above (before termination check)
        # so skip_unary has already been applied
        
        if self.verbose >= 3:
            self.debug_helper._log(3, f"[step] After compute_derived: derived_counts={self.derived_states_counts[:min(8, self.batch_size_int)]}")

        # Pack observation
        obs = self._create_observation_dict()
        obs['is_success'] = is_success.squeeze(-1)
        
        # DEBUG: Print derived counts to diagnose constrained action space
        if self.verbose >= 2:
            sample_counts = self.derived_states_counts[:min(4, B)]
            self.debug_helper._log(2, f"[step] derived_states_counts (first 4): {sample_counts.tolist()}")
        
        td = TensorDict(
            {**obs, "reward": rewards, "done": dones, "terminated": terminated, "truncated": truncated},
            batch_size=self.batch_size, device=self._device
        )
        if self.verbose >= 1:
            self.debug_helper._dump_states(
                "[step] After:", self.current_queries, self.derived_states_batch, self.derived_states_counts,
                current_depths=self.current_depths, proof_depths=self.proof_depths, current_labels=self.current_labels,
                action_mask=obs.get('action_mask'), done=dones, rewards=rewards, level=1
            )
        
        # DEBUG: Check done flag before returning
        if self.verbose >= 2:
            self.debug_helper._log(2, f"[_step] Returning td['done']: {td['done']}, terminated: {td['terminated']}")

        return td

    # ---------------------------------------------------------------------
    # Derived-state computation
    # ---------------------------------------------------------------------
    def _compute_derived_states(self, active_mask: Optional[torch.Tensor] = None, clear_inactive: bool = False):
        """
        Compute derived states following proof-safe logic:
        1) Mark terminal rows (TRUE/FALSE/END)
        2) Expand valid (non-terminal) rows
        3) Postprocess A: memory prune non-terminals, reject over-budget, mark & protect terminals
        4) Terminal handling: emit rewards for rows with terminals
        5) Skip-unary loop on remaining non-terminal rows
        6) Update current states for branching rows
        """
        if active_mask is None:
            active_mask = torch.ones(self.batch_size_int, dtype=torch.bool, device=self._device)

        idx = torch.arange(active_mask.shape[0], device=self._device)[active_mask]
        batched_derived = self.derived_states_batch.clone()
        counts = self.derived_states_counts.clone()
        
        verbose = self.verbose >= 3

        if idx.numel() > 0:
            if verbose:
                self.debug_helper._log(3, f"[compute_derived] env_count={idx.numel()}")
                self.debug_helper._log(3, f"[compute_derived] sample query: {self.current_queries[idx[0]]}")
                self.debug_helper._log(3, f"[compute_derived] next_var_indices: {self.next_var_indices[idx[:min(3, idx.numel())]]}")
            
            # Step 1: Mark which rows are end-of-state (terminal: TRUE/FALSE/END)
            terminal_mask = self.unification_engine.is_terminal_state(self.current_queries[idx])
            non_terminal_idx = idx[~terminal_mask]
            
            if verbose:
                self.debug_helper._log(3, f"[compute_derived] terminal_mask sum: {terminal_mask.sum().item()}/{idx.numel()}")
            
            # Step 2: Expand valid (non-terminal) rows only
            if non_terminal_idx.numel() > 0:
                all_derived, derived_counts_subset, updated_var_indices = self.unification_engine.get_derived_states(
                    current_states=self.current_queries.index_select(0, non_terminal_idx),
                    next_var_indices=self.next_var_indices.index_select(0, non_terminal_idx),
                    excluded_queries=self.original_queries.index_select(0, non_terminal_idx).unsqueeze(1),
                    verbose=self.prover_verbose,
                )
                
                if verbose:
                    self.debug_helper._log(3, f"[compute_derived] after UE shape={all_derived.shape}, counts={derived_counts_subset[:min(3, derived_counts_subset.numel())]}")
                    # Log env 0 specifically
                    env_0_in_idx = (non_terminal_idx == 0).nonzero(as_tuple=False)
                    if env_0_in_idx.numel() > 0:
                        env_0_pos = env_0_in_idx[0].item()
                        self.debug_helper._log(3, f"[compute_derived] Env 0 has {derived_counts_subset[env_0_pos].item()} derived states")
                
                self.next_var_indices.index_copy_(0, non_terminal_idx, updated_var_indices)

                # Step 3: Postprocess A - validity + proof-safe cut
                # This includes: memory prune non-terminals, padding limits, mark terminals, add non-terminals to memory
                if verbose:
                    self.debug_helper._log(3, f"[compute_derived] BEFORE postprocess_A counts={derived_counts_subset[:min(3, derived_counts_subset.numel())]}")
                all_derived, derived_counts_subset = self._postprocess(
                    non_terminal_idx, all_derived, derived_counts_subset
                )
                
                if verbose:
                    self.debug_helper._log(3, f"[compute_derived] AFTER postprocess_A counts={derived_counts_subset[:min(3, derived_counts_subset.numel())]}")
                
                # Step 4: Terminal handling (early, after A)
                # For any row where a terminal is present, we'll handle it in skip_unary or final write
                
                # Step 5: Skip-unary loop (for remaining non-terminal rows)
                # In exact-memory mode we already applied skip-unary directly
                # to current states in _reset/_step, so we only run the
                # batched skip-unary here for the Bloom filter backend.
                if self.skip_unary_actions and not self.use_exact_memory:
                    all_derived, derived_counts_subset = self._skip_unary(non_terminal_idx, all_derived, derived_counts_subset)
                    if verbose:
                        self.debug_helper._log(3, f"[compute_derived] after skip_unary counts={derived_counts_subset}")

                # Step 5.5: Add END action if configured and room available (for non-terminal rows)
                if self.end_proof_action and all_derived.numel() > 0:
                    all_derived, derived_counts_subset = self._add_end_action(all_derived, derived_counts_subset)
                    if verbose:
                        self.debug_helper._log(3, f"[compute_derived] after add_end_action counts={derived_counts_subset}")

                # Step 6: Final handling - fit to buffer dimensions
                all_derived, derived_counts_subset = self._fit_to_buffer(all_derived, derived_counts_subset, non_terminal_idx.numel())
                
                batched_derived.index_copy_(0, non_terminal_idx, all_derived)
                counts.index_copy_(0, non_terminal_idx, derived_counts_subset)
            
            # Handle terminal rows: they keep their current state as the only derived state
            if terminal_mask.any():
                terminal_idx = idx[terminal_mask]
                terminal_derived, terminal_counts = self.unification_engine.create_terminal_derived(
                    self.current_queries[terminal_idx], self.padding_states, self.padding_atoms)
                batched_derived.index_copy_(0, terminal_idx, terminal_derived)
                counts.index_copy_(0, terminal_idx, terminal_counts)

        if clear_inactive:
            inactive_mask = ~active_mask
            inactive_idx = torch.arange(active_mask.shape[0], device=self._device)[inactive_mask]
            if inactive_idx.numel() > 0:
                batched_derived[inactive_idx] = self.padding_idx
                counts[inactive_idx] = 0

        # Safety: any active env with zero states -> insert FALSE
        need_false = active_mask & (counts == 0)
        dst_rows = torch.arange(active_mask.shape[0], device=self._device)[need_false]
        if dst_rows.numel() > 0:
            false_state = self.unification_engine.get_false_state()  # [A, D]
            expanded = false_state.unsqueeze(0).expand(dst_rows.shape[0], -1, -1)
            batched_derived[dst_rows, 0] = expanded
            counts[dst_rows] = 1

        # Write back in-place instead of reassigning
        self.derived_states_batch.copy_(batched_derived)
        self.derived_states_counts.copy_(counts)

    def _fit_to_buffer(self, states: Tensor, counts: Tensor, num_envs: int) -> Tuple[Tensor, Tensor]:
        """
        Fit derived states to the environment buffer dimensions [N, S, A, D].
        Pads or truncates states dimension (S) and atoms dimension (A) as needed.
        """
        if states.numel() == 0:
            return (torch.full((num_envs, self.padding_states, self.padding_atoms, self.max_arity + 1),
                              self.padding_idx, dtype=torch.long, device=self._device),
                    torch.zeros(num_envs, dtype=torch.long, device=self._device))
        
        A, K_sub, M_sub, D = states.shape
        K_full = self.padding_states
        M_full = self.padding_atoms
        
        # Adjust atoms dim
        if M_sub < M_full:
            pad_atoms = torch.full((A, K_sub, M_full - M_sub, D), self.padding_idx, 
                                  dtype=states.dtype, device=self._device)
            states = torch.cat([states, pad_atoms], dim=2)
        elif M_sub > M_full:
            states = states[:, :, :M_full]
        
        # Adjust K dim
        if K_sub < K_full:
            pad_states = torch.full((A, K_full - K_sub, M_full, D), self.padding_idx, 
                                   dtype=states.dtype, device=self._device)
            states = torch.cat([states, pad_states], dim=1)
        elif K_sub > K_full:
            states = states[:, :K_full]
            counts = torch.clamp(counts, max=K_full)
        
        return states, counts

    def _add_end_action(self, states: Tensor, counts: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Add END action to derived states for rows that don't have all terminal states.
        This is called for non-terminal current states after postprocess and skip-unary.
        
        Args:
            states: [A, K, M, D] derived states
            counts: [A] number of valid states per row
        
        Returns:
            Updated (states, counts) with END actions added where appropriate
        """
        if states.numel() == 0:
            return states, counts
        
        # Guard: if end_pred_idx is not set, we cannot add END actions
        if self.end_pred_idx is None:
            raise ValueError("end_pred_idx is not set; cannot add END actions.")
        
        device = self._device
        pad = self.padding_idx
        A, K, M, D = states.shape
        
        # Check which rows have all terminal derived states
        first_preds = states[:, :, 0, 0]  # [A, K]
        is_term = self.unification_engine.is_terminal_pred(first_preds) | (first_preds == pad)
        
        valid_state_mask = torch.arange(K, device=device).view(1, -1).expand(A, -1) < counts.view(A, 1)
        all_terminal = (is_term | ~valid_state_mask).all(dim=1)  # [A]
        
        # Reserve END for rows that are not all terminal and have room
        reserve_end = ~all_terminal & (counts > 0) & (counts < K)
        
        if reserve_end.any():
            rows = torch.arange(A, device=device)[reserve_end]
            # Create END state: first atom is END, rest are padding
            end_state = torch.full((M, D), pad, dtype=torch.long, device=device)
            end_atom = self.unification_engine.get_end_state()  # [1, 3]
            end_state[0] = end_atom[0]  # Copy END atom to first position
            
            pos = counts[rows]
            expanded_end = end_state.unsqueeze(0).expand(rows.shape[0], -1, -1)
            states[rows, pos] = expanded_end
            counts[rows] = pos + 1
        
        return states, counts

    # ---------------------------------------------------------------------
    # Skip-unary closure (batched; bounded loop on subset only)
    # ---------------------------------------------------------------------
    def _apply_skip_unary_to_current_state(self, env_indices: Tensor):
        """
        Apply skip_unary logic to current states immediately after taking an action.
        This follows unary chains until reaching a branching point or terminal state.
        Updates self.current_queries in-place for the given environment indices.
        
        This is called in _step BEFORE checking termination to ensure that terminal
        states reached via unary chains (e.g., False()) are detected immediately.
        """
        if env_indices.numel() == 0:
            return
        
        device = self._device
        pad = self.padding_idx
        max_iters = self.max_skip_unary_iters
        
        # Track which envs are still following unary chains
        active_envs = env_indices.clone()
        hit_limit = False
        
        for iter_count in range(max_iters):
            if active_envs.numel() == 0:
                break
            
            if self.verbose >= 3:
                self.debug_helper._log(3, f"[apply_skip_unary] Iteration {iter_count}: {active_envs.numel()} active envs")
            
            # Check if current states are terminal
            terminal_mask = self.unification_engine.is_terminal_state(self.current_queries[active_envs])
            active_envs = active_envs[~terminal_mask]
            
            if active_envs.numel() == 0:
                break
            
            # Get derived states for active envs
            current_states = self.current_queries.index_select(0, active_envs)  # [N, A, D]
            next_vars = self.next_var_indices.index_select(0, active_envs)
            excluded = self.original_queries.index_select(0, active_envs).unsqueeze(1)
            
            derived_batch, derived_counts, updated_vars = self.unification_engine.get_derived_states(
                current_states=current_states,
                next_var_indices=next_vars,
                excluded_queries=excluded,
                verbose=0,
            )
            
            # Update next_var_indices for envs that got new derived states
            self.next_var_indices.index_copy_(0, active_envs, updated_vars)
            
            if self.verbose >= 2:
                for i in range(active_envs.shape[0]):
                    env_idx = active_envs[i]
                    count = derived_counts[i].item()
                    self.debug_helper._log(2, f"[apply_skip_unary] Iter {iter_count}: env {env_idx.item()} has {count} derived states BEFORE memory pruning")
            
            # Apply memory pruning if enabled (exact/Python or Bloom filter)
            if self.memory_pruning:
                # Check membership for all derived states
                visited = self.memory_backend.membership(
                    derived_batch,  # [N, K, M, D]
                    active_envs  # [N]
                )  # Returns [N, K] bool tensor
                
                # Keep only non-visited states
                for i in range(active_envs.shape[0]):
                    count = derived_counts[i].item()
                    if count == 0:
                        continue
                    
                    # Get mask for this env's derived states
                    not_visited = ~visited[i, :count]
                    
                    if not not_visited.any():
                        # All states were visited, set count to 0
                        derived_counts[i] = 0
                    elif not not_visited.all():
                        # Some states were visited, compact the kept ones
                        kept_count = not_visited.sum().item()
                        kept_states = derived_batch[i, :count][not_visited]
                        derived_batch[i, :kept_count] = kept_states
                        derived_batch[i, kept_count:] = pad
                        derived_counts[i] = kept_count
            
            # Check which have exactly one non-terminal child
            is_single = (derived_counts == 1)
            has_derived = is_single
            
            if self.verbose >= 2:
                for i in range(active_envs.shape[0]):
                    env_idx = active_envs[i]
                    count = derived_counts[i].item()
                    self.debug_helper._log(2, f"[apply_skip_unary] Iter {iter_count}: env {env_idx.item()} has {count} derived states AFTER memory pruning")
            
            if not has_derived.any():
                break
            
            # Get first predicate of the single child for rows with count=1
            first_preds = torch.full((active_envs.shape[0],), pad, dtype=torch.long, device=device)
            single_mask = is_single
            if single_mask.any():
                first_preds[single_mask] = derived_batch[single_mask, 0, 0, 0]
            
            # Check if the single child is terminal
            is_child_terminal = self.unification_engine.is_terminal_pred(first_preds)
            
            # Keep only rows with single non-terminal child
            is_unary_nonterminal = is_single & ~is_child_terminal
            
            if not is_unary_nonterminal.any():
                break
            
            # Promote the single child to current state for unary rows
            unary_envs = active_envs[is_unary_nonterminal]
            unary_local_idx = torch.arange(active_envs.shape[0], device=device)[is_unary_nonterminal]
            
            promoted = derived_batch[unary_local_idx, 0]  # [U, M, D]
            
            # Ensure promoted states match padding_atoms dimension
            if promoted.shape[1] < self.padding_atoms:
                pad_cols = self.padding_atoms - promoted.shape[1]
                pad_tail = torch.full(
                    (promoted.shape[0], pad_cols, promoted.shape[2]),
                    pad, dtype=promoted.dtype, device=device
                )
                promoted = torch.cat([promoted, pad_tail], dim=1)
            elif promoted.shape[1] > self.padding_atoms:
                promoted = promoted[:, :self.padding_atoms]
            
            # Update current_queries for unary envs
            self.current_queries.index_copy_(0, unary_envs, promoted)
            
            # Add promoted states to memory
            if self.memory_pruning:
                self.memory_backend.add_current(unary_envs, self.current_queries)
            
            # Continue with rows that were unary
            active_envs = unary_envs
            
            if self.verbose >= 2:
                for env_idx in active_envs[:min(2, active_envs.numel())]:
                    state_str = self.debug_helper._format_atoms(self.current_queries[env_idx])
                    self.debug_helper._log(2, f"[apply_skip_unary] Iter {iter_count}: env {env_idx.item()} state: {state_str[:100]}...")
        else:
            # Loop completed without breaking - we hit the iteration limit
            hit_limit = True
        
        # Check if we hit the iteration limit
        # If so, inject False() state to match str environment behavior
        if hit_limit and active_envs.numel() > 0:
            if self.verbose >= 1:
                self.debug_helper._log(1, f"[apply_skip_unary] Hit max_iters={max_iters}, injecting False() for {active_envs.numel()} envs")
            
            if self.false_pred_idx is None:
                raise RuntimeError("False predicate index is undefined; cannot inject False() state after skip-unary cap")
            
            # Build False() state
            false_state = torch.full(
                (self.padding_atoms, self.max_arity + 1),
                pad,
                dtype=self.current_queries.dtype,
                device=device
            )
            false_state[0, 0] = self.false_pred_idx
            
            # Set current state to False() for envs that hit the limit
            false_expanded = false_state.unsqueeze(0).expand(active_envs.shape[0], -1, -1)
            self.current_queries.index_copy_(0, active_envs, false_expanded)
        
        if self.verbose >= 2 and env_indices.numel() > 0:
            for i in range(min(3, env_indices.numel())):
                env_idx = env_indices[i]
                state_str = self.debug_helper._format_atoms(self.current_queries[env_idx])
                self.debug_helper._log(2, f"[apply_skip_unary] Final state for env {env_idx.item()}: {state_str}")
    
    def _skip_unary(self, idx_subset: Tensor, derived_states: Tensor, derived_counts: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Skip-unary loop following proof-safe logic (Step 5):
        While a row (current state) is non-terminal and has exactly one non-terminal child:
        - Promote that child → current; add the previous current to memory
        - Expand: derived ← get_derived_states(current)
        - Run postprocess_A (memory pruning, padding, mark terminals, add kept to memory)
        - Terminal handling: if terminal appears, stop processing that row
        - Update current states to the new kept non-terminals
        - Stop the loop on terminal, on first branch (>1 child), or after hop cap
        """
        if derived_states.numel() == 0:
            return derived_states, derived_counts

        pad = self.padding_idx
        A, K, M, D = derived_states.shape
        device = self._device
        verbose = self.verbose >= 3

        iters = 0
        
        # Helper: identify rows with single non-terminal child
        def _unary_nonterminal_rows(ds, dc):
            """Returns mask [A] for rows that have exactly one non-terminal child."""
            if ds.numel() == 0:
                return torch.zeros(A, dtype=torch.bool, device=device)
            
            is_single = (dc == 1)
            fp = ds[:, 0, 0, 0]  # first predicate of the unique child
            is_term = self.unification_engine.is_terminal_pred(fp)
            return is_single & ~is_term
        
        # Start with rows that have single non-terminal children
        unary_mask = _unary_nonterminal_rows(derived_states, derived_counts)
        unary_idx = torch.arange(A, device=device)[unary_mask]

        if self.verbose >= 2:
            for i in range(min(3, unary_idx.numel())):
                row = unary_idx[i]
                env_idx = idx_subset[row].item()
                current_state = self.current_queries[env_idx]
                current_str = self.debug_helper._format_atoms(current_state)
                derived_state = derived_states[row, 0]
                derived_str = self.debug_helper._format_atoms(derived_state)
                self.debug_helper._log(2, f"[skip_unary]  Unary in idx {env_idx}: current={current_str}, derived={derived_str}")

        if verbose:
            self.debug_helper._log(3, f"[skip_unary] Starting with {unary_idx.numel()} unary rows")

        # Bounded iterations
        while unary_idx.numel() > 0 and iters < self.max_skip_unary_iters:
            uidx = unary_idx  # [U] - indices into the A-dimension
            env_rows = idx_subset.index_select(0, uidx)  # [U] - actual env indices
            if env_rows.dtype != torch.long:
                env_rows = env_rows.long()

            if verbose:
                self.debug_helper._log(3, f"[skip_unary] Iteration {iters}: processing {uidx.numel()} rows")

            # Promote the single child to current state (will be added to memory in _postprocess)
            promoted = derived_states[uidx, 0]  # [U, M, D]
            if promoted.dtype != self.current_queries.dtype:
                promoted = promoted.long()
            
            # Ensure promoted states respect env atom budget
            if promoted.shape[1] > self.padding_atoms:
                promoted = promoted[:, :self.padding_atoms]
            elif promoted.shape[1] < self.padding_atoms:
                pad_rows = self.padding_atoms - promoted.shape[1]
                pad_tail = torch.full(
                    (promoted.shape[0], pad_rows, promoted.shape[2]),
                    pad, dtype=promoted.dtype, device=device
                )
                promoted = torch.cat([promoted, pad_tail], dim=1)
            
            self.current_queries.index_copy_(0, env_rows, promoted)

            # Check if promoted states are terminal - if so, stop processing them
            terminal_mask = self.unification_engine.is_terminal_state(self.current_queries[env_rows])
            non_terminal_in_promoted = env_rows[~terminal_mask]
            uidx_non_terminal = uidx[~terminal_mask]
            
            if verbose:
                self.debug_helper._log(3, f"[skip_unary] After promotion: {terminal_mask.sum().item()} became terminal")

            # For terminal rows, set their derived to just themselves
            if terminal_mask.any():
                terminal_uidx = uidx[terminal_mask]
                terminal_env_rows = env_rows[terminal_mask]
                terminal_derived, terminal_counts = self.unification_engine.create_terminal_derived(
                    self.current_queries[terminal_env_rows], K, M)
                
                # Adjust to (K, M, D) for scatter back
                terminal_derived_subset = terminal_derived[:, :K, :M, :]
                derived_states.index_copy_(0, terminal_uidx, terminal_derived_subset)
                derived_counts[terminal_uidx] = terminal_counts

            # Expand non-terminal promoted states
            if non_terminal_in_promoted.numel() > 0:
                sub_derived, sub_counts, sub_next = self.unification_engine.get_derived_states(
                    current_states=self.current_queries.index_select(0, non_terminal_in_promoted),
                    next_var_indices=self.next_var_indices.index_select(0, non_terminal_in_promoted),
                    excluded_queries=self.original_queries.index_select(0, non_terminal_in_promoted).unsqueeze(1),
                    verbose=self.prover_verbose,
                )
                self.next_var_indices.index_copy_(0, non_terminal_in_promoted, sub_next)

                # Postprocess_A: memory pruning, padding limits, mark terminals, add to memory
                sub_derived, sub_counts = self._postprocess(non_terminal_in_promoted, sub_derived, sub_counts)

                if verbose:
                    self.debug_helper._log(3, f"[skip_unary] After postprocess_A: counts={sub_counts[:min(3, sub_counts.shape[0])]}")

                # Fit to current K, M dimensions
                U, Ks, Ms, Ds = sub_derived.shape if sub_derived.numel() > 0 else (uidx_non_terminal.numel(), 0, M, D)

                # Pad/truncate atoms
                if Ms < M:
                    pad_atoms = torch.full((U, Ks, M - Ms, D), pad, dtype=sub_derived.dtype, device=device)
                    sub_derived = torch.cat([sub_derived, pad_atoms], dim=2)
                elif Ms > M:
                    sub_derived = sub_derived[:, :, :M]
                
                # Pad/truncate states
                if Ks < K:
                    pad_states = torch.full((U, K - Ks, M, D), pad, dtype=sub_derived.dtype, device=device)
                    sub_derived = torch.cat([sub_derived, pad_states], dim=1)
                elif Ks > K:
                    sub_derived = sub_derived[:, :K]
                    sub_counts = torch.clamp(sub_counts, max=K)

                # Ensure dtypes match
                if sub_derived.dtype != derived_states.dtype:
                    sub_derived = sub_derived.to(dtype=derived_states.dtype)
                if sub_counts.dtype != derived_counts.dtype:
                    sub_counts = sub_counts.to(dtype=derived_counts.dtype)
                
                # Scatter back
                derived_states.index_copy_(0, uidx_non_terminal, sub_derived)
                derived_counts.index_copy_(0, uidx_non_terminal, sub_counts)
            
            # Recompute unary mask - only non-terminal rows can continue
            unary_mask = _unary_nonterminal_rows(derived_states, derived_counts)
            unary_idx = torch.arange(A, device=device)[unary_mask]
            iters += 1

            if verbose:
                self.debug_helper._log(3, f"[skip_unary] End of iteration {iters}: {unary_idx.numel()} rows remain unary")

        # If we exited the loop because we hit the iteration cap but still have unary rows,
        # align with the string environment semantics: mark those rows as False().
        if unary_idx.numel() > 0:
            if self.false_pred_idx is None:
                raise RuntimeError("False predicate index is undefined; cannot inject False() state after skip-unary cap")

            env_rows = idx_subset.index_select(0, unary_idx)
            if env_rows.dtype != torch.long:
                env_rows = env_rows.long()

            # Build False() state tensors
            false_state = torch.full(
                (self.padding_atoms, self.max_arity + 1),
                pad,
                dtype=self.current_queries.dtype,
                device=device,
            )
            false_state[0, 0] = self.false_pred_idx
            false_batch = false_state.unsqueeze(0).expand(env_rows.shape[0], -1, -1)

            # Overwrite current queries for the exhausted rows
            self.current_queries.index_copy_(0, env_rows, false_batch)

            # Their derived states become the terminal False() state only
            terminal_derived, terminal_counts = self.unification_engine.create_terminal_derived(
                self.current_queries[env_rows], K, M)
            terminal_subset = terminal_derived[:, :K, :M, :]
            derived_states.index_copy_(0, unary_idx, terminal_subset)
            derived_counts.index_copy_(0, unary_idx, terminal_counts)

            if verbose:
                self.debug_helper._log(2, f"[skip_unary] Iteration cap reached; forced {env_rows.shape[0]} rows to False()")

        return derived_states, derived_counts

    def _is_terminal_pred(self, pred_indices: Tensor) -> Tensor:
        """
        Check if predicate indices are terminal (TRUE/FALSE/END).
        pred_indices: [N] tensor of predicate indices
        Returns: [N] boolean mask
        """
        is_term = torch.zeros_like(pred_indices, dtype=torch.bool)
        if self.true_pred_idx is not None:
            is_term |= (pred_indices == self.true_pred_idx)
        if self.false_pred_idx is not None:
            is_term |= (pred_indices == self.false_pred_idx)
        if self.end_pred_idx is not None:
            is_term |= (pred_indices == self.end_pred_idx)
        return is_term

    # ---------------------------------------------------------------------
    # Postprocess (memory pruning + compaction + truncation)
    # ---------------------------------------------------------------------
    def _postprocess(self, env_indices: Tensor, states: Tensor, counts: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Postprocess A - Proof-safe validity and budget cut:
        - Add current state to memory (it's being expanded now)
        - Mark non-terminal rows
        - Memory prune non-terminals using memory (terminals bypass)
        - Reject any state violating padding_cfg (state or atom count) - no semantic truncation
        - Mark terminals and protect them from being dropped
        
        This ensures:
        - Proof safety: terminals are never dropped by memory prune
        - Soundness under padding: any over-budget state is rejected, not truncated
        - Memory/tabling is per-query (episode) and idempotent
        """
        device = self._device
        pad = self.padding_idx

        if states.numel() == 0:
            return states, counts

        # Step 0: Add current state to memory before checking derived states
        # This prevents loops (derived states that match current state will be pruned)
        if self.memory_pruning:
            self.memory_backend.add_current(env_indices, self.current_queries)

        A, K, M, D = states.shape
        has_states = counts > 0

        # Step 1: Mark non-terminal states
        # Get first predicate of each state
        first_preds = states[:, :, 0, 0]  # [A, K]
        is_terminal_state = self.unification_engine.is_terminal_pred(first_preds)  # [A, K]

        # Step 2: Base validity checks (drop empty, beyond-count, over-budget)
        valid_atom = states[:, :, :, 0] != pad  # [A, K, M]
        state_nonempty = valid_atom.any(dim=2)  # [A, K]
        atom_counts = valid_atom.sum(dim=2)  # [A, K]
        
        # CRITICAL: Reject (not truncate) any state exceeding atom budget
        within_atom_budget = atom_counts <= self.padding_atoms  # [A, K]
        within_count = torch.arange(K, device=device).view(1, K).expand(A, K) < counts.view(A, 1)
        base_valid = state_nonempty & within_atom_budget & within_count  # [A, K]

        if self.verbose >= 2:
            rejected_violations = state_nonempty & (~within_count | ~within_atom_budget)
            if rejected_violations.any():
                num_rejected = rejected_violations.sum().item()
                self.debug_helper._log(2, f"[postprocess] Rejected {num_rejected} states violating budget constraints (atom or state count)")
                rejected_indices = rejected_violations.nonzero(as_tuple=False)
                for i in range(min(3, rejected_indices.shape[0])):
                    row, col = rejected_indices[i]
                    env_idx = env_indices[row].item()
                    state = states[row, col]
                    state_str = self.debug_helper._format_atoms(state)
                    atom_count = atom_counts[row, col].item()
                    self.debug_helper._log(2, f"  Rejected in env {env_idx}: {state_str} (atoms: {atom_count})")

        # Step 3: Memory pruning (non-terminals only)
        keep_mask = base_valid.clone()
        
        if self.memory_pruning:
            owners = env_indices
            if owners.dtype != torch.long:
                owners = owners.long()
            
            # Check membership for all base-valid states
            visited = torch.zeros((A, K), dtype=torch.bool, device=device)
            if base_valid.any():
                visited = self.memory_backend.membership(states, owners)  # [A, K]
                visited = visited & base_valid
            
            # Only prune non-terminal visited states (terminals are protected)
            prune_mask = visited & ~is_terminal_state
            keep_mask = base_valid & ~prune_mask
            
            # Verbose logging
            if self.verbose >= 2 and prune_mask.any():
                pruned_indices = prune_mask.nonzero(as_tuple=False)
                for i in range(pruned_indices.shape[0]):
                    row, col = pruned_indices[i]
                    env_idx = env_indices[row].item()
                    current_state_str = self.debug_helper._format_atoms(self.current_queries[env_idx])
                    all_derived = [self.debug_helper._format_atoms(states[row, j]) for j in range(states.shape[1]) if base_valid[row, j]]
                    state = states[row, col]
                    state_str = self.debug_helper._format_atoms(state)
                    remaining_states = states[row][keep_mask[row]]
                    remaining_states = [self.debug_helper._format_atoms(s) for s in remaining_states]
                    self.debug_helper._log(2, f" [postprocess_A]  Pruned in Idx {env_idx}: {state_str}. Remaining: {remaining_states}")

        # Ensure terminals are always kept if they're valid
        keep_mask = keep_mask | (is_terminal_state & base_valid)

        # Step 4: Compact kept states
        new_counts = keep_mask.sum(dim=1)  # [A]
        needs_false = (new_counts == 0) & has_states  # [A]

        pos = torch.cumsum(keep_mask.long(), dim=1) - 1  # [A, K]
        pos = torch.clamp(pos, min=0, max=K - 1)
        batch_idx = torch.arange(A, device=device).view(A, 1).expand(A, K)

        compact = torch.full((A, K, M, D), pad, dtype=states.dtype, device=device)
        if keep_mask.any():
            target_rows = batch_idx[keep_mask]
            target_pos = pos[keep_mask]
            compact[target_rows, target_pos] = states[keep_mask]
        
        counts_out = new_counts.clone()

        # Step 5: Inject FALSE where needed
        if needs_false.any():
            # Create FALSE state with correct M dimension to match compact
            false_state = torch.full((M, D), pad, dtype=states.dtype, device=device)
            false_state[0, 0] = self.false_pred_idx
            num_false = needs_false.sum().item()
            # Expand to [num_false, M, D]
            false_expanded = false_state.unsqueeze(0).expand(num_false, -1, -1)
            compact[needs_false, 0] = false_expanded
            counts_out[needs_false] = 1

        # Note: We do NOT add derived states to memory here.
        # Derived states are added to memory only when they become the current state
        # (in _step or _skip_unary when promoted). This prevents the correct tabling behavior.

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
        if self.verbose >= 3:
            sample = min(2, B)
            self.debug_helper._log(3, f"[reward] preds sample: {preds[:sample, :2]}")
            self.debug_helper._log(3, f"[reward] labels sample: {self.current_labels[:sample]}")
            self.debug_helper._log(3, f"[reward] true_idx={self.true_pred_idx} false_idx={self.false_pred_idx}")

        all_true = torch.zeros(B, dtype=torch.bool, device=device)
        any_false = torch.zeros(B, dtype=torch.bool, device=device)
        is_end = torch.zeros(B, dtype=torch.bool, device=device)

        if self.true_pred_idx is not None:
            true_mask = (preds == self.true_pred_idx) | ~non_pad
            all_true = true_mask.all(dim=1) & non_pad.any(dim=1)
            if self.verbose >= 3:
                self.debug_helper._log(3, f"[reward] all_true raw: {all_true[:sample]}")

        if self.false_pred_idx is not None:
            any_false = (preds == self.false_pred_idx).any(dim=1)
            if self.verbose >= 3:
                self.debug_helper._log(3, f"[reward] any_false raw: {any_false[:sample]}")

        if self.end_proof_action:
            single_pred = non_pad.sum(dim=1) == 1
            first_pos = non_pad.long().argmax(dim=1)
            first_pred = preds[torch.arange(B, device=device), first_pos]
            is_end = single_pred & (first_pred == self.end_pred_idx)
            any_false = any_false | is_end
            all_true = all_true & ~is_end

        depth_exceeded = self.current_depths >= self.max_depth
        natural_term = all_true | any_false
        if self.verbose >= 3:
            self.debug_helper._log(3, f"[reward] natural_term: {natural_term[:sample]}")
        terminated[:, 0] = natural_term
        truncated[:, 0] = depth_exceeded & ~natural_term
        done = natural_term | depth_exceeded
        if self.verbose >= 3:
            self.debug_helper._log(3, f"[reward] terminated: {terminated[:sample, 0]}")
            self.debug_helper._log(3, f"[reward] truncated: {truncated[:sample, 0]}")

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
        action_mask = action_indices < counts # [B, S]
        
        # If everything is masked we need to surface the underlying issue immediately.
        # HOWEVER: If all environments are done, this is normal and we should just return dummy obs
        all_false = (~action_mask).all(dim=1)
        if all_false.any():
            rows = all_false.nonzero(as_tuple=False).view(-1)
            # Check if these rows are all done - if so, it's OK (terminal states don't need actions)
            # For now, check if current state is terminal (TRUE/FALSE/END)
            first_preds = self.current_queries[rows, 0, 0]
            is_terminal = torch.zeros_like(first_preds, dtype=torch.bool)
            # Also treat depth-capped states as terminal: they hit truncation even if query isn't TRUE/FALSE
            depth_capped = self.current_depths[rows] >= self.max_depth
            is_terminal = is_terminal | (first_preds == self.true_pred_idx) | (first_preds == self.false_pred_idx)| depth_capped
            if self.end_pred_idx is not None:
                is_terminal = is_terminal | (first_preds == self.end_pred_idx)          
            non_terminal_masked = rows[~is_terminal]
            if non_terminal_masked.numel() > 0:
                diagnostics = {
                    "rows": non_terminal_masked.tolist(),
                    "derived_counts": self.derived_states_counts[non_terminal_masked].tolist(),
                    "current_depths": self.current_depths[non_terminal_masked].tolist(),
                    "labels": self.current_labels[non_terminal_masked].tolist(),
                }
                raise RuntimeError(
                    "[BatchedEnv] All actions masked for non-terminal states; inspect derived-state generator. "
                    f"Context={diagnostics}"
                )

        return {
            'sub_index': self.current_queries.unsqueeze(1),
            'derived_sub_indices': self.derived_states_batch,
            'action_mask': action_mask,
            'label': self.current_labels,
            'query_depth': self.current_depths,
        }

    def _create_observation(self) -> TensorDict:
        """Create initial observation TensorDict"""
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
    def _ensure_query_tensor(self, queries: Tensor) -> Tensor:
        """Ensure queries tensor is properly padded to [N, padding_atoms, max_arity+1].
        Expects input of shape [N, L, D] where 
        N is number of queries, L is current atom count, D is current width.
        Pads/truncates atoms to padding_atoms and width to max_arity + 1.
        """
        if not isinstance(queries, torch.Tensor) or queries.dim() != 3:
            raise TypeError(
                "Expected `queries` to be a tensor of shape [N, L, max_arity+1]. "
                "Use DataHandler.materialize_indices() to build these tensors."
            )
        queries = queries.to(device=self._device, dtype=torch.long)
        pad = self.padding_idx
        B, cur_atoms, cur_width = queries.shape
        target_width = self.max_arity + 1
        assert cur_width == target_width, "Input queries must have arity+1 dimension of size 3."

        if cur_atoms < self.padding_atoms:
            atom_tail = torch.full(
                (B, self.padding_atoms - cur_atoms, cur_width),
                pad,
                dtype=torch.long,
                device=self._device,
            )
            queries = torch.cat([queries, atom_tail], dim=1)
        elif cur_atoms > self.padding_atoms:
            raise ValueError("Query atom count exceeds padding_atoms.")

        return queries

    def _ensure_vector(self, data: Optional[Tensor], expected_len: int, name: str) -> Tensor:
        """Ensure data is a 1D tensor of expected length on the correct device."""
        if data is None:
            if expected_len == 0:
                return torch.zeros((0,), dtype=torch.long, device=self._device)
            raise ValueError(f"{name} must be provided when queries are non-empty")
        tensor = torch.as_tensor(data, dtype=torch.long, device=self._device).view(-1)
        if tensor.shape[0] != expected_len:
            raise ValueError(
                f"{name} length ({tensor.shape[0]}) does not match number of queries ({expected_len})"
            )
        return tensor

    def _compute_first_atoms(self, padded_queries: Tensor) -> Tensor:
        """Extract first atoms from padded queries tensor.
        Expects shape [N, A, D], returns [N, D].
        N is number of queries, A is padding_atoms, D is max_arity + 1."""
        pad = self.padding_idx
        assert padded_queries.dim() == 3, "padded_queries must be 3D tensor"
        valid = padded_queries[:, :, 0] != pad
        has_valid = valid.any(dim=1) # check if each query has any valid atom
        first_pos = valid.long().argmax(dim=1)
        batch_ids = torch.arange(padded_queries.shape[0], device=self._device)
        first_atoms = padded_queries[batch_ids, first_pos]
        if (~has_valid).any():
            first_atoms = first_atoms.clone()
            first_atoms[~has_valid] = torch.full(
                (self.max_arity + 1,), pad, dtype=torch.long, device=self._device
            )
        return first_atoms

    def _update_original_queries_for_indices(self, indices: Tensor):
        """For selected env indices, update original_queries to match the first atom of current_queries."""
        if indices.numel() == 0:
            return
        device = self._device
        pad = self.padding_idx

        rows = indices.to(device=device, dtype=torch.long)
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

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    # ===================== Evaluation =====================

    def set_eval_dataset(
        self,
        queries: torch.Tensor,        # [M, A, D]
        labels: torch.Tensor,         # [M]
        query_depths: torch.Tensor,   # [M]
        per_slot_lengths: Optional[torch.Tensor] = None,  # [B] (first Q > 0, rest 0)
    ) -> None:
        """
        Preloads the evaluation dataset and (optionally) a per-slot schedule.

        If per_slot_lengths is provided:
        - Data in `queries/labels/depths` MUST be laid out as a concatenation of
            per-slot blocks: slot0's E0 items, then slot1's E1 items, etc.
        - We compute per-slot (start, length) and reset per-slot pointers.
        - _reset(mode='eval') will then pull the next item for EACH SLOT independently
            (partial resets work too).

        If per_slot_lengths is None:
        - Falls back to the original global sequential pointer (self.counter).
        """
        device = self._device
        
        # Switch to eval mode (required for per-slot scheduling to work)
        self.mode = 'eval'
        self.memory_pruning = self.eval_pruning
        
        # 1) Store dataset tensors
        self._all_queries_padded = self._ensure_query_tensor(queries.to(device))
        self._all_first_atoms    = self._compute_first_atoms(self._all_queries_padded)
        self._all_labels         = self._ensure_vector(labels.to(device), self._all_queries_padded.shape[0], "labels")
        self._all_depths         = self._ensure_vector(query_depths.to(device), self._all_queries_padded.shape[0], "query_depths")
        self._num_all            = int(self._all_queries_padded.shape[0])

        # 2) Default: no per-slot schedule
        self._eval_slot_lengths = None
        self._eval_slot_starts  = None
        self._eval_slot_ptr     = None

        # 3) Optional per-slot schedule
        self._eval_init_slots(per_slot_lengths)

        # 4) Reset global pointer as well (used if no per-slot schedule)
        self.counter = 0

    def _eval_init_slots(self, per_slot_lengths: Optional[torch.Tensor]) -> None:
        """
        Initialize per-slot evaluation schedule.
        """
        if per_slot_lengths is not None:
            per_slot_lengths = per_slot_lengths.to(device=self._device, dtype=torch.long)
            if per_slot_lengths.numel() != self.batch_size_int:
                raise ValueError(f"per_slot_lengths must be [B], got {per_slot_lengths.shape}")
            total = int(per_slot_lengths.sum().item())
            if total != self._num_all:
                raise ValueError(f"Sum(per_slot_lengths)={total} must equal #items M={self._num_all}.")

            # starts[i] = sum_{j<i} lengths[j]
            starts = torch.zeros_like(per_slot_lengths)
            if starts.numel() > 1:
                starts[1:] = torch.cumsum(per_slot_lengths[:-1], dim=0)

            self._eval_slot_lengths = per_slot_lengths
            self._eval_slot_starts  = starts
            self._eval_slot_ptr     = torch.zeros_like(per_slot_lengths)  # next local offset per slot

    def _eval_sampling(self, env_idx: torch.Tensor) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """
        Sample evaluation indices for the given env indices.
        Returns updated env_idx, N, idxs.
        """
        # Per-slot schedule: pick next item for each env row independently
        # env_idx lists the rows we are resetting (ascending order)
        # Filter out slots with zero length
        starts  = self._eval_slot_starts
        lengths = self._eval_slot_lengths
        ptrs    = self._eval_slot_ptr

        # Filter env_idx to only include slots with non-zero lengths
        active_slots_mask = lengths[env_idx] > 0
        if not active_slots_mask.any():
            # No active slots to reset, return empty observation
            raise RuntimeError("No active slots to reset")

        active_env_idx = env_idx[active_slots_mask]
        N_active = active_env_idx.shape[0]

        # Pure GPU version: vectorized operations
        ofs = ptrs[active_env_idx]
        exceeded_mask = ofs >= lengths[active_env_idx]
        if exceeded_mask.any():
            exceeded_indices = active_env_idx[exceeded_mask]
            exceeded_ofs = ofs[exceeded_mask]
            exceeded_limits = lengths[active_env_idx][exceeded_mask]
            raise RuntimeError(
                f"Slots {exceeded_indices.tolist()} exceeded their scheduled eval items "
                f"(ptrs={exceeded_ofs.tolist()}, limits={exceeded_limits.tolist()})"
            )
        idxs = starts[active_env_idx] + ofs
        ptrs[active_env_idx] = ofs + 1

        # Write back updated pointers
        self._eval_slot_ptr = ptrs

        # Update env_idx and N to only include active slots
        env_idx = active_env_idx
        N = N_active

        return env_idx, N, idxs
