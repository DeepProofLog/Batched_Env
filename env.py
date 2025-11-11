
import torch
from torch import Tensor
from typing import Dict, List, Optional, Tuple
from tensordict import TensorDict
from torchrl.envs import EnvBase

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
        self.atom_stringifier = atom_stringifier
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

        # End tensor (optional)
        self.end_tensor = None
        if (self.end_pred_idx is not None) and (self.end_pred_idx >= 0):
            self.end_tensor = torch.tensor(
                [[self.end_pred_idx, self.padding_idx, self.padding_idx]],
                dtype=torch.long, device=self._device
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
            (B, self.padding_atoms, arity_plus), self.padding_idx, dtype=torch.long, device=self._device
        )  # [B, A, D] - current query states for each env
        self.current_labels = torch.zeros(B, dtype=torch.long, device=self._device)  # [B] - labels for each env
        self.current_depths = torch.zeros(B, dtype=torch.long, device=self._device)  # [B] - current depths for each env
        self.next_var_indices = torch.full(
            (B,), self.runtime_var_start_index, dtype=torch.long, device=self._device
        )  # [B] - next variable indices for each env
        self.original_queries = torch.full(
            (B, arity_plus), self.padding_idx, dtype=torch.long, device=self._device
        )  # [B, D] - original query atoms for each env

        # Derived states [B, S, A, D]
        # S: state size (max number of derived states per env)
        self.derived_states_batch = torch.full(
            (B, self.padding_states, self.padding_atoms, arity_plus),
            self.padding_idx, dtype=torch.long, device=self._device
        )  # [B, S, A, D] - derived states for each env
        self.derived_states_counts = torch.zeros(B, dtype=torch.long, device=self._device)  # [B] - number of valid derived states per env

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

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------
    def set_verbose(self, level: int) -> None:
        self.verbose = int(level)

    def _log(self, level: int, message: str) -> None:
        if self.verbose >= level:
            print(f"{self._debug_prefix} {message}")

    def _format_atoms(self, state: torch.Tensor) -> list[str]:
        """Format a state tensor into a list of atom strings or indices."""
        mask = state[:, 0] != self.padding_idx
        atoms = state[mask]
        if atoms.numel() == 0:
            return []
        if self.atom_stringifier is not None:
            return [self.atom_stringifier.atom_to_str(atom) for atom in atoms]
        return atoms.tolist()

    def _dump_states(self, label: str, rows: Optional[torch.Tensor] = None, level=2,
                     action_mask: Optional[torch.Tensor] = None,
                     done: Optional[torch.Tensor] = None,
                     rewards: Optional[torch.Tensor] = None) -> None:
        """Dump current and derived states for specified rows (or all)."""
        if rows is None:
            self._log(level, f"{label}: there are no rows to dump for {label}.")
        else:
            rows_list = rows.tolist()
            self._log(level, f"{label}: inspecting rows {rows_list}")
            for idx in rows_list:
                state = self.current_queries[idx]
                derived_count = int(self.derived_states_counts[idx].item())
                state_atoms = self._format_atoms(state)
                self._log(level, f"  Idx {idx} depth={int(self.current_depths[idx])} label={int(self.current_labels[idx])} query={state_atoms}")

                # Include action mask info if provided
                if action_mask is not None:
                    mask_for_idx = action_mask[idx].tolist()
                    valid_actions = sum(1 for m in mask_for_idx if m)
                    self._log(level, f"    Action mask valid: {valid_actions}")

                # Include done/rewards info if provided
                if done is not None:
                    done_for_idx = done[idx].item()
                    self._log(level, f"    Done: {done_for_idx}")
                if rewards is not None:
                    reward_for_idx = rewards[idx].item()
                    self._log(level, f"    Reward: {reward_for_idx}")

                derived_atoms = []
                for d in range(derived_count):
                    derived_atoms.append(self._format_atoms(self.derived_states_batch[idx, d]))
                self._log(level, f"    Derived: {derived_atoms}\n")

    def _merge_reset_obs(self, base_td: TensorDict, reset_td: TensorDict, done_mask: torch.Tensor) -> TensorDict:
        """Merge base and reset tensordicts based on done mask."""
        rows = done_mask.nonzero(as_tuple=False).view(-1)
        if rows.numel() == 0:
            return base_td
        merged = base_td.clone()
        for key in reset_td.keys():
            reset_val = reset_td.get(key)
            if isinstance(reset_val, TensorDict):
                base_val = merged.get(key) if key in merged.keys() else reset_val.clone()
                merged.set(key, self._merge_reset_obs(base_val, reset_val, done_mask))
            else:
                if key in merged.keys():
                    target = merged.get(key).clone()
                else:
                    target = reset_val.clone()
                target[rows] = reset_val[rows]
                merged.set(key, target)
        return merged

    # ---------------------------------------------------------------------
    # Specs
    # ---------------------------------------------------------------------
    def _make_specs(self):
        from torchrl.data import Composite, Bounded

        max_vocab_size = int(self.total_vocab_size)
        B = self.batch_size_int  # batch size
        A = self.padding_atoms   # atoms size
        D = self.max_arity + 1   # arity + 1
        S = self.padding_states  # state size

        self.observation_spec = Composite(
            sub_index=Bounded(
                low=-1, high=max_vocab_size,
                shape=torch.Size([B, 1, A, D]), dtype=torch.int64, device=self._device,
            ),  # [B, 1, A, D] - current query state
            derived_sub_indices=Bounded(
                low=-1, high=max_vocab_size,
                shape=torch.Size([B, S, A, D]), dtype=torch.int64, device=self._device,
            ),  # [B, S, A, D] - derived states
            action_mask=Bounded(
                low=0, high=1, shape=torch.Size([B, S]), dtype=torch.bool, device=self._device
            ),  # [B, S] - mask for valid actions
            shape=torch.Size([B]),
        )
        self.action_spec = Bounded(
            low=0, high=self.padding_states - 1, shape=torch.Size([B]), dtype=torch.int64, device=self._device
        )  # [B] - action indices
        self.reward_spec = Bounded(
            low=-float('inf'), high=float('inf'), shape=torch.Size([B, 1]), dtype=torch.float32, device=self._device
        )  # [B, 1] - rewards
        self.done_spec = Bounded(
            low=0, high=1, shape=torch.Size([B, 1]), dtype=torch.bool, device=self._device
        )  # [B, 1] - termination flags
        self.truncated_spec = Bounded(
            low=0, high=1, shape=torch.Size([B, 1]), dtype=torch.bool, device=self._device
        )  # [B, 1] - truncation flags

    # ---------------------------------------------------------------------
    # Core Env API
    # ---------------------------------------------------------------------
    def _reset(self, tensordict: Optional[TensorDict] = None) -> TensorDict:
        """
        Vectorized reset with optional negative sampling. Resets per-env Bloom
        filters and inserts the initial state into memory when memory_pruning is on.
        TD keys are: 
            "_reset": [B, 1] boolean mask for partial reset
            
        """
        device = self._device
        pad = self.padding_idx
        B = self.batch_size_int
        A = self.padding_atoms
        D = self.max_arity + 1

        if self.verbose >= 3:
            if tensordict is not None:
                self._log(3, f"[Reset]  td info:")
                for key in tensordict.keys():
                    self._log(3, f"[Reset]  Key: {key}, Shape: {tensordict[key].shape}, Dtype: {tensordict[key].dtype}, Values: {tensordict[key]}")
            else: 
                self._log(3, "[Reset]  No tensordict provided.")

        # Reset mask
        if tensordict is not None:
            if "_reset" not in tensordict.keys():
                raise ValueError("_reset key not found in tensordict for partial reset.")
            reset_mask = tensordict["_reset"].squeeze(-1) # [B] bool
            assert reset_mask.dtype == torch.bool, "_reset mask must be of boolean dtype"
            env_idx = torch.arange(reset_mask.shape[0], device=device)[reset_mask] # N Indices to reset
            assert env_idx.numel() > 0, "No environments to reset in partial reset."
        else:
            reset_mask = torch.ones(B, dtype=torch.bool, device=device)
            env_idx = torch.arange(B, device=device)

        if self.verbose >= 1:
            # if there are envs not being reset, log them
            if not reset_mask.shape[0] == B:
                self._log(2, f"[Reset] Not resetting envs: {[(i.item()) for i in torch.arange(B, device=device) if i not in env_idx]}")

        assert env_idx.shape[0] > 0, "No environments to reset."
        N = env_idx.shape[0]

        # Sample indices
        if self.mode == 'train':
            idxs = torch.randint(0, self._num_all, (N,), device=device)
        elif self.mode == 'eval':
            assert hasattr(self, "_eval_slot_starts") and self._eval_slot_starts is not None and \
                   hasattr(self, "_eval_slot_lengths") and self._eval_slot_lengths is not None and \
                   hasattr(self, "_eval_slot_ptr") and self._eval_slot_ptr is not None, \
                "Evaluation slots not initialized. Call set_eval_dataset with per_slot_lengths."
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
                return self._create_observation()
            
            active_env_idx = env_idx[active_slots_mask]
            N_active = active_env_idx.shape[0]
            
            idxs = torch.empty(N_active, dtype=torch.long, device=device)
            # local copy to avoid multiple Python <-> tensor switches
            for j in range(N_active):
                s = int(active_env_idx[j].item())    # slot id
                ofs = int(ptrs[s].item())            # next local offset for this slot
                if ofs >= int(lengths[s].item()):
                    raise RuntimeError(
                        f"Slot {s} exceeded its scheduled eval items "
                        f"(ptr={ofs}, limit={int(lengths[s].item())}, "
                        f"lengths={lengths.tolist()})"
                    )
                idxs[j] = starts[s] + ofs
                ptrs[s] = ofs + 1                    # advance local pointer
            # write back (in-place tensor already updated)
            self._eval_slot_ptr = ptrs
            
            # Update env_idx and N to only include active slots
            env_idx = active_env_idx
            N = N_active
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        # Gather batch queries/labels/depths
        batch_q = self._all_queries_padded.index_select(0, idxs)                 # [N, A, D]
        batch_labels = self._all_labels.index_select(0, idxs)                    # [N]
        batch_depths = self._all_depths.index_select(0, idxs)                    # [N]
        batch_first_atoms = self._all_first_atoms.index_select(0, idxs)          # [N, D]

        # Vectorized negative sampling
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
                batch_depths = batch_depths.clone()
                batch_depths.index_fill_(0, neg_rows, -1)

        # Write into runtime buffers
        self.current_queries.index_copy_(0, env_idx, batch_q)
        self.current_labels.index_copy_(0, env_idx, batch_labels)
        self.current_depths.index_copy_(0, env_idx, batch_depths)
        self.next_var_indices.index_fill_(0, env_idx, self.runtime_var_start_index)

        # Reset per-env memory & add the starting state
        if self.memory_pruning:
            self._bloom_reset(env_idx)
            self._bloom_add_current(env_idx)

        # original_queries = first non-padding atom of current state
        self._update_original_queries_for_indices(env_idx)

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
                false_state_full = self._create_false_state()  # [A, D] padded
                false_atom = false_state_full[0]  # Just the first atom [D]
                false_queries = torch.full((inactive_in_reset.shape[0], A, D), pad, dtype=torch.long, device=device)
                false_queries[:, 0] = false_atom
                self.current_queries.index_copy_(0, inactive_in_reset, false_queries)
                self.current_labels.index_fill_(0, inactive_in_reset, 0)
                self.current_depths.index_fill_(0, inactive_in_reset, self.max_depth + 1)  # Exceed max_depth to ensure done
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
            self._dump_states("After reset", env_idx, level=1, action_mask=obs_dict.get('action_mask'))
        return self._create_observation()

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

        # Apply
        B = self.batch_size_int
        batch_ids = torch.arange(B, device=self._device)
        if active_mask.any():
            active_rows = batch_ids[active_mask]
            chosen = self.derived_states_batch[active_rows, actions[active_rows]]
            self.current_queries.index_copy_(0, active_rows, chosen)
        self.current_depths = self.current_depths + active_mask.long()

        # Add new current state to memory
        if self.memory_pruning:
            if active_mask.any():
                self._bloom_add_current(batch_ids[active_mask])

        # Reward + done
        rewards, terminated, truncated, is_success = self._get_done_reward()
        dones = terminated | truncated
        if self.verbose >= 3:
            sample = min(4, B)
            self._log(3, f"[step] dones sample: {dones[:sample].view(-1)}")
            self._log(3, f"[step] depth sample: {self.current_depths[:sample]}, labels: {self.current_labels[:sample]}")
            self._log(3, f"[step] current query sample: {self.current_queries[:sample]}")

        # Next derived states for active envs
        active_mask = ~dones.squeeze(-1)
        if self.verbose >= 3:
            self._log(3, f"[step] active_mask={active_mask[:min(4, B)]}, sum={active_mask.sum().item()}")
            if active_mask.any():
                active_idx = torch.arange(self.batch_size_int, device=self._device)[active_mask]
                self._log(3, f"[step] current_queries[active]: {self.current_queries[active_idx[:min(2, active_idx.shape[0])]]}")
                self._log(3, f"[step] current_depths[active]: {self.current_depths[active_idx[:min(2, active_idx.shape[0])]]}")
        
        # In eval mode, don't clear inactive (done) envs - they need to keep their terminal state with valid actions
        # In train mode, clearing is OK because envs will be reset immediately
        should_clear_inactive = (self.mode == 'train')
        self._compute_derived_states(active_mask=active_mask, clear_inactive=should_clear_inactive)
        
        if self.verbose >= 3:
            self._log(3, f"[step] After compute_derived: derived_counts={self.derived_states_counts[:min(8, self.batch_size_int)]}")

        # Pack observation
        obs = self._create_observation_dict()
        obs['is_success'] = is_success.squeeze(-1)
        
        # DEBUG: Print derived counts to diagnose constrained action space
        if self.verbose >= 2:
            sample_counts = self.derived_states_counts[:min(4, B)]
            self._log(2, f"[step] derived_states_counts (first 4): {sample_counts.tolist()}")
        
        td = TensorDict(
            {**obs, "reward": rewards, "done": dones, "terminated": terminated, "truncated": truncated},
            batch_size=self.batch_size, device=self._device
        )
        if self.verbose >= 1:
            self._dump_states("After step", action_mask=obs.get('action_mask'), done=dones, rewards=rewards, level=1)
        print(osdihb)
        return td

    # ---------------------------------------------------------------------
    # Derived-state computation
    # ---------------------------------------------------------------------
    def _compute_derived_states(self, active_mask: Optional[torch.Tensor] = None, clear_inactive: bool = False):
        if active_mask is None:
            active_mask = torch.ones(self.batch_size_int, dtype=torch.bool, device=self._device)

        idx = torch.arange(active_mask.shape[0], device=self._device)[active_mask]
        batched_derived = self.derived_states_batch.clone()
        counts = self.derived_states_counts.clone()
        
        verbose = self.verbose >= 3

        if idx.numel() > 0:
            if verbose:
                self._log(3, f"[compute_derived] env_count={idx.numel()}")
                self._log(3, f"[compute_derived] sample query: {self.current_queries[idx[0]]}")
                self._log(3, f"[compute_derived] next_var_indices: {self.next_var_indices[idx[:min(3, idx.numel())]]}")
            
            all_derived, derived_counts_subset, updated_var_indices = self.unification_engine.get_derived_states(
                current_states=self.current_queries.index_select(0, idx),
                next_var_indices=self.next_var_indices.index_select(0, idx),
                excluded_queries=self.original_queries.index_select(0, idx),
                verbose=self.prover_verbose,
            )
            
            if verbose:
                self._log(3, f"[compute_derived] after UE shape={all_derived.shape}, counts={derived_counts_subset[:min(3, derived_counts_subset.shape[0])]}")
            
            self.next_var_indices.index_copy_(0, idx, updated_var_indices)

            # Optional skip-unary closure on the active subset
            if self.skip_unary_actions:
                all_derived, derived_counts_subset = self._skip_unary(idx, all_derived, derived_counts_subset)
                if verbose:
                    self._log(3, f"[compute_derived] after skip_unary counts={derived_counts_subset[:min(3, derived_counts_subset.shape[0])]}")

            # Vectorized postprocess (with memory pruning)
            all_derived, derived_counts_subset = self._postprocess(
                idx, all_derived, derived_counts_subset, stage="final"
            )
            
            if verbose:
                self._log(3, f"[compute_derived] after postprocess counts={derived_counts_subset[:min(3, derived_counts_subset.shape[0])]}")
            
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
            inactive_mask = ~active_mask
            inactive_idx = torch.arange(active_mask.shape[0], device=self._device)[inactive_mask]
            if inactive_idx.numel() > 0:
                batched_derived[inactive_idx] = self.padding_idx
                counts[inactive_idx] = 0

        # Safety: any active env with zero states -> insert FALSE
        need_false = active_mask & (counts == 0)
        dst_rows = torch.arange(active_mask.shape[0], device=self._device)[need_false]
        if dst_rows.numel() > 0:
            false_state = self._create_false_state()                  # [A, D]
            expanded = false_state.unsqueeze(0).expand(dst_rows.shape[0], -1, -1)
            batched_derived[dst_rows, 0] = expanded
            counts[dst_rows] = 1

        # Write back in-place instead of reassigning
        self.derived_states_batch.copy_(batched_derived)
        self.derived_states_counts.copy_(counts)

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
        unary_idx = torch.arange(A, device=self._device)[unary_mask]

        # Bounded iterations; each loop is a *single* batched UE call on the subset
        while unary_idx.numel() > 0 and iters < self.max_skip_unary_iters:
            uidx = unary_idx                                           # [U]
            # OPTIMIZATION: idx_subset should already be long dtype
            env_rows = idx_subset.index_select(0, uidx)
            if env_rows.dtype != torch.long:
                env_rows = env_rows.long()

            # Promote the single child to current state for those envs
            # OPTIMIZATION: derived_states should match current_queries dtype
            promoted = derived_states[uidx, 0]
            if promoted.dtype != self.current_queries.dtype:
                promoted = promoted.long()
            # Ensure promoted states respect env atom budget
            if promoted.shape[1] > self.padding_atoms:
                promoted = promoted[:, :self.padding_atoms]
            elif promoted.shape[1] < self.padding_atoms:
                pad_rows = self.padding_atoms - promoted.shape[1]
                pad_tail = torch.full(
                    (promoted.shape[0], pad_rows, promoted.shape[2]),
                    pad,
                    dtype=promoted.dtype,
                    device=self._device,
                )
                promoted = torch.cat([promoted, pad_tail], dim=1)
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
            unary_idx = torch.arange(A, device=self._device)[unary_mask]
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
        valid_atom = states[:, :, :, 0] != pad                          # [A, K, A]
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
            has_valid = bool(base_valid.any())
            if has_valid:
                cand = states.clone()  # [A, K, A, D]
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
        # Allocate full-K and fill by position; we’ll truncate later if needed
        pos = torch.cumsum(keep_mask.long(), dim=1) - 1                      # [A, K]
        pos = torch.clamp(pos, min=0, max=K - 1)
        batch_idx = torch.arange(A, device=device).view(A, 1).expand(A, K)
        col_idx   = torch.arange(K, device=device).view(1, K).expand(A, K)

        compact = torch.full((A, K, M, D), pad, dtype=states.dtype, device=device)
        if keep_mask.any():
            target_rows = batch_idx[keep_mask]
            target_pos = pos[keep_mask]
            compact[target_rows, target_pos] = states[keep_mask]
        counts_out = new_counts.clone()
        W = compact.size(1)   # width of the 2nd dim (was max_new / max_new_int / K)


        # 4) Inject FALSE where needed
        if needs_false.any():
            false_state = self._create_false_state().unsqueeze(0)     # [1, A, D]
            compact[needs_false, 0] = false_state
            counts_out[needs_false] = 1

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

            valid_state_mask = torch.arange(W, device=device).view(1, -1).expand(A, -1) < counts_out.view(A, 1)
            all_terminal = (is_term | ~valid_state_mask).all(dim=1)
            reserve_end = ~all_terminal & (counts_out > 0)

        limit = torch.full((A,), self.padding_states, dtype=torch.long, device=device)
        limit = torch.where(reserve_end, limit - 1, limit)
        limit = torch.clamp(limit, min=0)

        need_trunc = counts_out > limit
        if need_trunc.any():
            # scores / masks sized by W
            scores = (compact[:, :, :, 0] != pad).sum(dim=2)  # [A, W]
            valid_mask = torch.arange(W, device=device).view(1, -1).expand(A, -1) < counts_out.view(A, 1)
            scores_masked = torch.where(valid_mask, scores, torch.full_like(scores, -1))
            sort_idx = torch.argsort(scores_masked, dim=1, stable=True)          # [A, W]
            gather_idx = sort_idx.unsqueeze(2).unsqueeze(3).expand(A, W, M, D)
            sorted_states = torch.gather(compact, 1, gather_idx)

            take_mask = torch.arange(W, device=device).view(1, -1).expand(A, -1) < limit.view(A, 1)
            out = torch.full_like(compact, pad)                                   # keep width = W
            if take_mask.any():
                row_ids = torch.arange(A, device=device).view(A, 1).expand(A, W)
                col_ids = torch.arange(W, device=device).view(1, W).expand(A, W)
                take_rows = row_ids[take_mask]
                take_cols = col_ids[take_mask]
                pos2 = torch.cumsum(take_mask.long(), dim=1) - 1
                pos2 = torch.clamp(pos2, min=0, max=W - 1)
                out[take_rows, pos2[take_mask]] = sorted_states[take_rows, take_cols]

            compact = out
            counts_out = torch.minimum(counts_out, limit)


        # 7) Add END action if configured and room available
        if self.end_proof_action:
            can_add_end = reserve_end & (counts_out < self.padding_states)
            rows = torch.arange(A, device=device)[can_add_end]
            if rows.numel() > 0:
                end_state = self._create_end_state(target_atoms=compact.shape[2])
                pos3 = counts_out[rows]
                expanded_end = end_state.unsqueeze(0).expand(rows.shape[0], -1, -1)
                compact[rows, pos3] = expanded_end
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
        if self.verbose >= 3:
            sample = min(2, B)
            self._log(3, f"[reward] preds sample: {preds[:sample, :2]}")
            self._log(3, f"[reward] labels sample: {self.current_labels[:sample]}")
            self._log(3, f"[reward] true_idx={self.true_pred_idx} false_idx={self.false_pred_idx}")

        all_true = torch.zeros(B, dtype=torch.bool, device=device)
        any_false = torch.zeros(B, dtype=torch.bool, device=device)
        is_end = torch.zeros(B, dtype=torch.bool, device=device)

        if self.true_pred_idx is not None:
            true_mask = (preds == self.true_pred_idx) | ~non_pad
            all_true = true_mask.all(dim=1) & non_pad.any(dim=1)
            if self.verbose >= 3:
                self._log(3, f"[reward] all_true raw: {all_true[:sample]}")

        if self.false_pred_idx is not None:
            any_false = (preds == self.false_pred_idx).any(dim=1)
            if self.verbose >= 3:
                self._log(3, f"[reward] any_false raw: {any_false[:sample]}")

        if self.end_proof_action:
            single_pred = non_pad.sum(dim=1) == 1
            first_pos = non_pad.long().argmax(dim=1)
            first_pred = preds[torch.arange(B, device=device), first_pos]
            is_end = single_pred & (first_pred == self.end_pred_idx)
            all_true = all_true | is_end
            any_false = any_false & ~is_end

        depth_exceeded = self.current_depths >= self.max_depth
        natural_term = all_true | any_false
        if self.verbose >= 3:
            self._log(3, f"[reward] natural_term: {natural_term[:sample]}")
        terminated[:, 0] = natural_term
        truncated[:, 0] = depth_exceeded & ~natural_term
        done = natural_term | depth_exceeded
        if self.verbose >= 3:
            self._log(3, f"[reward] terminated: {terminated[:sample, 0]}")
            self._log(3, f"[reward] truncated: {truncated[:sample, 0]}")

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

    def step_and_maybe_reset(self, tensordict: TensorDict):
        """Leverage EnvBase implementation but keep verbose logging."""
        step_td, next_td = super().step_and_maybe_reset(tensordict)
        if self.verbose >= 2:
            done_mask = next_td.get("done").squeeze(-1).bool()
            reward = next_td.get("reward")
            action_mask = next_td.get("action_mask")
            self._log(2, f"[step_and_maybe_reset] done_mask={done_mask.tolist()}")
            self._dump_states("After step_and_maybe_reset", action_mask=action_mask, done=done_mask, rewards=reward)
        return step_td, next_td

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

    def _pad_state(self, state: Tensor, target_atoms: Optional[int] = None) -> Tensor:
        device = self._device
        pad = self.padding_idx
        A = int(target_atoms) if target_atoms is not None else self.padding_atoms
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

    def _create_end_state(self, target_atoms: Optional[int] = None) -> Tensor:
        if self.end_pred_idx is None:
            raise ValueError("End predicate index is not defined.")
        state = torch.full((1, self.max_arity + 1), self.padding_idx, dtype=torch.long, device=self._device)
        state[0, 0] = int(self.end_pred_idx)
        return self._pad_state(state, target_atoms)

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

        # 3) Optional per-slot schedule (first Q entries > 0; rest 0)
        if per_slot_lengths is not None:
            per_slot_lengths = per_slot_lengths.to(device=device, dtype=torch.long)
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

        # 4) Reset global pointer as well (used if no per-slot schedule)
        self.counter = 0
