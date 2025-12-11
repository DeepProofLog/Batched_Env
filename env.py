"""
Batched Environment for Neural-Guided Logical Reasoning.

This module provides a vectorized TorchRL-compatible environment for 
training agents on logical reasoning tasks (theorem proving, KG reasoning).

Key Features:
    - GPU-first vectorized operations across B environments.
    - Memory pruning via Bloom filters or exact memory.
    - Skip-unary optimization for efficient rollout traversal.
    - Negative sampling for contrastive learning support.
    - Logic-safe termination (TRUE/FALSE/END).

Tensor Shape Conventions:
    - B: Batch size (number of parallel environments)
    - A: Atoms per state (max_atoms/padding_atoms)
    - D: Atom dimension (max_arity + 1) -> [pred, arg0, arg1]
    - S: States per step (action space size/padding_states)
    - N: Variable number of active queries or derived states

Core Tensors:
    current_queries:        [B, A, D]
    derived_states_batch:   [B, S, A, D]
    derived_states_counts:  [B]
    action_mask:            [B, S]
    current_labels:         [B]
    current_depths:         [B]
"""
import torch
from torch import Tensor
from typing import Dict, List, Optional, Tuple
from tensordict import TensorDict
from torchrl.envs import EnvBase

from unification import UnificationEngine
from utils.memory import BloomFilter, ExactMemory, GPUExactMemory
from utils.debug_helper import DebugHelper


def _safe_device(device: Optional[torch.device]) -> torch.device:
    if device is not None:
        return device
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class BatchedEnv(EnvBase):
    """
    Vectorized, GPU-first logic environment.
    
    This environment manages `B` parallel logical proofs. It interacts with the `UnificationEngine`
    to generate derived states (successors) and supports advanced features like bloom-filter based
    memory pruning and skip-unary optimization for efficient training.
    
    Attributes:
        batch_size_int (int): Number of parallel environments (B).
        current_queries (Tensor): [B, A, D] Current state of proofs.
        derived_states_batch (Tensor): [B, S, A, D] Cached successors for the current step.
        derived_states_counts (Tensor): [B] Number of valid successors per env.
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
        end_proof_action: bool = False,
        skip_unary_actions: bool = False,
        reward_type: int = 0,
        verbose: int = 0,
        prover_verbose: int = 0,
        device: Optional[torch.device] = None,
        # Deterministic sampling mode for parity testing with SB3
        sample_deterministic_per_env: bool = False,
        # Memory pruning config (Bloom filter)
        memory_bits_pow: int = 22,     # 2**22 bits per env (512 KB) -> higher precision
        memory_hashes: int = 7,        # k hash functions -> more hashes reduce false positives
        use_exact_memory: bool = False,
        debug_config=None,  # DebugConfig instance
    ):
        # Configure device and batch size
        self.batch_size_int = int(batch_size)
        device = _safe_device(device)
        super().__init__(device=device, batch_size=torch.Size([batch_size]))
        
        # Import here to avoid circular dependency
        if debug_config is None:
            from utils.debug_config import DebugConfig
            debug_config = DebugConfig()
        self.debug_config = debug_config

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
        if self.train_neg_ratio > 0:
            self.rejection_weight = 1 / self.train_neg_ratio
        else:
            self.rejection_weight = 1
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
        # Deterministic round-robin pointer for train query sampling (parity with SB3)
        self._train_ptr = 0
        # Per-env deterministic sampling mode (for parity testing with SB3)
        self.sample_deterministic_per_env = sample_deterministic_per_env
        if self.sample_deterministic_per_env:
            # Each env has its own pointer, initialized to env_idx like SB3
            # (SB3 sets env._train_ptr = env_idx for parity testing)
            self._per_env_train_ptrs = torch.arange(self.batch_size_int, dtype=torch.long, device=self._device)

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
        self._train_neg_counters = torch.zeros(B, dtype=torch.long, device=self._device)

        # Derived states [B, S, A, D]
        self.derived_states_batch = torch.full(
            (B, self.padding_states, self.padding_atoms, arity_plus),
            self.padding_idx,
            dtype=torch.long,
            device=self._device,
        )
        self.derived_states_counts = torch.zeros(B, dtype=torch.long, device=self._device)

        # -------- Memory pruning (Bloom filter per env) --------
        # memory_pruning applies to both train and eval modes
        self.memory_pruning = bool(memory_pruning)
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
                true_pred_idx=self.true_pred_idx,
                false_pred_idx=self.false_pred_idx,
                end_pred_idx=self.end_pred_idx,
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
    # ---------------------------------------------------------------------
    # Specs
    # ---------------------------------------------------------------------
    def _make_specs(self) -> None:
        """
        Define the observation, action, and reward specifications.
        
        Observation Spec (Composite):
            - sub_index:            [B, 1, A, D] Current state (expanded dim for consistency).
            - derived_sub_indices:  [B, S, A, D] Potential successor states.
            - action_mask:          [B, S] Boolean mask of valid actions.
            - shape:                [B] Batch shape.
            
        Action Spec (Bounded):
            - shape: [B] Integer action indices in range [0, S-1].
            
        Reward Spec:
            - shape: [B, 1] Scalar reward.
        """
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
        """
        Reset environments to initial states.
        
        Supports partial resets via `tensordict['_reset']` mask.
        If `tensordict` is None, resets all environments.
        
        Logic:
        1. Identify environments to reset (all or subset).
        2. Sample new queries from the dataset (deterministic or random based on mode).
        3. Apply negative sampling (corruption) if training.
        4. Reset internal state buffers (depths, memory, etc.).
        5. Compute initial derived states for the new queries.
        
        Args:
            tensordict: Optional TensorDict containing `_reset` boolean mask [B, 1].
            
        Returns:
            TensorDict containing initial observations for the batch.
        """
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
            if self.sample_deterministic_per_env:
                # Per-env round-robin: each env has its own pointer (parity with SB3)
                # Vectorized update of pointers
                current_ptrs = self._per_env_train_ptrs[env_idx]
                idxs = current_ptrs % self._num_all
                self._per_env_train_ptrs[env_idx] = (current_ptrs + 1) % self._num_all
            else:
                # Global round-robin: queries distributed across all resets
                start = self._train_ptr
                idxs = (torch.arange(N, device=device) + start) % self._num_all
                self._train_ptr = int((start + N) % self._num_all)
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
        batch_q, batch_labels, proof_depths = self.sample_negatives(N, batch_q, batch_labels, proof_depths, batch_first_atoms, env_idx, device, pad, A, D)

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

    def sample_negatives(
        self,
        N: int,
        batch_q: Tensor,
        batch_labels: Tensor,
        proof_depths: Tensor,
        batch_first_atoms: Tensor,
        env_idx: Tensor,
        device: torch.device,
        pad: int,
        A: int,
        D: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Vectorized negative sampling (corruption) matching SB3's behavior.
        
        If enabled, corrupts the head/tail of queries to create negative examples
        for contrastive learning.
        
        Args:
            N (int):                 Number of environments being reset.
            batch_q (Tensor):        [N, A, D] Query tensors.
            batch_labels (Tensor):   [N] Label tensors.
            proof_depths (Tensor):   [N] Depth tensors.
            batch_first_atoms (Tensor): [N, D] First atom of each query (for corruption context).
            env_idx (Tensor):        [N] Indices of environments being reset.
            device (torch.device):   Target device.
            pad (int):               Padding index.
            A (int):                 Number of atoms per state.
            D (int):                 Atom dimension (arity + 1).
            
        Returns:
            Tuple[Tensor, Tensor, Tensor]: (batch_q, batch_labels, proof_depths)
                                           Shapes: [N, A, D], [N], [N]
        """
        if not ((self.mode == 'train') and self.corruption_mode and (self.train_neg_ratio > 0)):
            return batch_q, batch_labels, proof_depths

        # Validate cycle
        ratio = int(round(float(self.train_neg_ratio)))
        if ratio <= 0:
            return batch_q, batch_labels, proof_depths
            
        cycle = ratio + 1
        local_counters = self._train_neg_counters.index_select(0, env_idx)  # [N]
        
        # Clone tensors for modification
        batch_q = batch_q.clone()
        batch_labels = batch_labels.clone()
        proof_depths = proof_depths.clone()
        
        # Determine which envs need negative samples
        # Logic: ratio=4 => cycle=5. if counter % 5 != 0 (1,2,3,4) -> negative. if 0 -> positive.
        needs_negative = (local_counters % cycle) != 0  # [N] boolean tensor
        
        num_negs = needs_negative.sum().item()
        if num_negs > 0:
            # Only corrupt atoms that need it to avoid wasting RNG (parity fix)
            atoms_to_corrupt = batch_first_atoms[needs_negative]  # [M, D]
            
            if self.sample_deterministic_per_env:
                # Sequential corruption for parity with SB3
                # Loop allows RNG consumption order to match sequential env execution
                corrupted_list = []
                for i in range(num_negs):
                     atom_i = atoms_to_corrupt[i:i+1] # [1, D]
                     c = self.sampler.corrupt(atom_i, num_negatives=1, device=device)
                     if c.dim() == 3: c = c[:, 0, :]
                     corrupted_list.append(c)
                corrupted_atoms = torch.cat(corrupted_list, dim=0) # [M, D]
            else:
                # Vectorized corruption (standard)
                corrupted_atoms = self.sampler.corrupt(
                    atoms_to_corrupt, 
                    num_negatives=1,
                    device=device
                )
                if corrupted_atoms.dim() == 3:
                     corrupted_atoms = corrupted_atoms[:, 0, :]
            
            # Apply to batch_q
            batch_q[needs_negative, 0, :D] = corrupted_atoms
            batch_labels[needs_negative] = 0
            proof_depths[needs_negative] = -1

        # Update per-env counters (mod cycle) for all processed envs
        new_counters = (local_counters + 1) % cycle
        self._train_neg_counters.index_copy_(0, env_idx, new_counters)

        return batch_q, batch_labels, proof_depths

    def _step(self, tensordict: TensorDict) -> TensorDict:
        """
        Execute one step of the environment.
        
        Logic:
        1. Read action from input `tensordict`.
        2. Validate action validity against `derived_states_counts`.
        3. Update `current_queries` with the chosen successor state.
        4. Increment depth.
        5. Compute new derived states (successors) for the new query.
        6. Apply skip-unary closure (fast-forwarding unary chains) if enabled.
        7. Check termination conditions (True, False, Depth Limit).
        8. Compute rewards.
        9. Pack observations for the next step.
        
        Args:
            tensordict: TensorDict containing `action` [B].
            
        Returns:
            TensorDict containing:
            - next observation (sub_index, derived_sub_indices, action_mask, etc.)
            - reward [B, 1]
            - done, terminated, truncated [B, 1]
        """
        
        actions = tensordict["action"]  # [B] or [B, 1]
        if actions.dtype != torch.long:
            actions = actions.long()  # [B]

        counts = self.derived_states_counts  # [B]
        active_mask = counts > 0  # [B] bool

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
            active_rows = batch_ids[active_mask]  # [A] active env indices
            chosen = self.derived_states_batch[active_rows, actions[active_rows]]  # [A, M, D]
            self.current_queries.index_copy_(0, active_rows, chosen)  # Update [B, M, D]

            # Optional string-env style skip-unary: follow unary chains on the
            # newly selected current states before expanding successors.
            # For exact-memory runs (used in deterministic parity tests), we keep
            # the unary chains in the action space to mirror the SB3 env traces.
            # Skip-unary promotion is only applied in training mode where we want
            # the shorter rollouts for efficiency.


        self.current_depths = self.current_depths + active_mask.long()

        # Note: Current state will be added to memory in _postprocess during _compute_derived_states

        # Compute derived states (and apply skip_unary if enabled)
        # This must be done BEFORE checking termination because skip_unary may reach a terminal state
        active_mask_for_derived = active_mask.clone()
        
        # In eval mode, don't clear inactive (done) envs - they need to keep their terminal state with valid actions
        # In train mode, clearing is OK because envs will be reset immediately
        should_clear_inactive = (self.mode == 'train')
        
        # TRACE: Before compute
        # print(f"[TRACE] Step start Label[0]={self.current_labels[0].item()}") # Assuming captured at top
        
        self._compute_derived_states(active_mask=active_mask_for_derived, clear_inactive=should_clear_inactive)

        if self.verbose >= 2:
            self.debug_helper._log(2, f"[_step] Before _get_done_reward, current_queries[0,0]: {self.current_queries[0, 0]}")
        rewards, terminated, truncated, is_success = self._get_done_reward()  # [B,1], [B,1], [B,1], [B,1]
        dones = terminated | truncated  # [B,1] bool
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
        obs['is_success'] = is_success
        
        # DEBUG: Print derived counts to diagnose constrained action space
        if self.verbose >= 2:
            sample_counts = self.derived_states_counts[:min(4, B)]
            self.debug_helper._log(2, f"[step] derived_states_counts (first 4): {sample_counts.tolist()}")
        
        # Enhanced debug output for action space analysis
        if self.debug_config.is_enabled('env') and self.debug_config.debug_env_action_spaces:
            self._debug_action_space(obs, dones, is_success)
        
        # Clone label, query_depth, and length ONLY when episodes end to prevent mutation by subsequent reset().
        # When dones.any() is False, no reset happens so no clone needed.
        # This avoids unnecessary memory overhead while preserving episode info for callbacks.
        if dones.any():
            label_val = self.current_labels.clone()
            depth_val = self.proof_depths.clone()
            length_val = self.current_depths.clone()
        else:
            label_val = self.current_labels
            depth_val = self.proof_depths
            length_val = self.current_depths
        
        td = TensorDict(
            {
                **obs, 
                "reward": rewards, 
                "done": dones, 
                "terminated": terminated, 
                "truncated": truncated,
                "label": label_val,
                "query_depth": depth_val,
                "length": length_val
            },
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

    def step_and_maybe_reset(self, tensordict: TensorDict) -> Tuple[TensorDict, TensorDict]:
        """
        Execute a step and automatically reset environments that are done.
        
        This prevents the agent from stepping through terminal states in vectorized settings.
        
        Args:
            tensordict: Input tensordict with action [B].
            
        Returns:
            Tuple[TensorDict, TensorDict]:
                - step_result_td: The full result of the step (including done=True for finished envs).
                - next_obs_td:    Observation for the *next* step (reset observations for done envs).
        """
        # Execute the step
        step_result = self.step(tensordict)
        
        # Extract next observation from the "next" key (TorchRL convention)
        if "next" in step_result.keys():
            next_obs = step_result.get("next")
        else:
            # If no "next" key, the obs might be directly in step_result
            next_obs = step_result
        
        # Check which environments are done
        done = next_obs.get("done")
        if done is not None:
            done_mask = done.squeeze(-1).bool()
            
            # Reset done environments
            if done_mask.any():
                if self.verbose >= 1:
                    self.debug_helper._log(1, f"[step_and_maybe_reset] Resetting {done_mask.sum().item()} done environments")
                
                # Create reset tensordict with _reset mask for partial reset
                reset_td = TensorDict(
                    {"_reset": done_mask.unsqueeze(-1)},
                    batch_size=self.batch_size,
                    device=self._device
                )
                # Reset only the done environments
                reset_obs = self.reset(reset_td)
                
                if self.verbose >= 1:
                    self.debug_helper._log(1, f"[step_and_maybe_reset] Reset complete, new query[0]: {self.current_queries[0, 0] if done_mask[0] else 'not reset'}")
                
                # Merge reset observations into next_obs for done environments
                # IMPORTANT: Keep step_result unchanged (with done=True) so episode tracking works
                # But return reset observations for the environments that were reset
                next_obs_merged = next_obs.clone()
                for key in reset_obs.keys():
                    if key in next_obs_merged.keys():
                        # Replace values for done environments
                        obs_val = next_obs_merged.get(key)
                        reset_val = reset_obs.get(key)
                        if obs_val.shape[0] == done_mask.shape[0]:
                            obs_val = obs_val.clone()
                            obs_val[done_mask] = reset_val[done_mask]
                            next_obs_merged.set(key, obs_val)
                
                # Return step_result unchanged (keeps done=True for episode tracking)
                # But next_obs_merged has the reset observations
                return step_result, next_obs_merged
        
        # No resets needed, return as-is
        return step_result, next_obs

    def _compute_derived_states(self, active_mask: Optional[torch.Tensor] = None, clear_inactive: bool = False) -> None:
        """
        Compute derived states following proof-safe logic.
        
        This method interacts with the `UnificationEngine` to expand the current states
        and then applies several filtering and post-processing steps.
        
        Logic:
        1. Identify non-terminal rows (those not already TRUE/FALSE/END).
        2. Expand non-terminal rows via `UnificationEngine.get_derived_states`.
        3. (Optional) Run skip-unary loop to fast-forward deterministic chains (train mode only).
        4. Post-process:
           - Filter by memory (Bloom/Exact) to avoid cycles (prune visited).
           - Reject states that exceed atom/state limits (no partial truncation).
           - Mark as terminal if needed.
        5. (Optional) Add END action for proofs that can stop.
        6. Fit results into the fixed-size `derived_states_batch` buffer.
        7. Handle terminal rows by assigning a self-loop (or dummy derived state).
        
        Args:
            active_mask (Tensor): [B] Boolean mask for environments to process.
                                  Inactive envs are skipped to save compute.
            clear_inactive (bool): If True, zeros out the derived states for inactive envs.
                                   Useful in training reset flows; risky in evaluation.
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
                    verbose=self.prover_verbose
                )
                
                if verbose:
                    self.debug_helper._log(3, f"[compute_derived] after UE shape={all_derived.shape}, counts={derived_counts_subset[:min(3, derived_counts_subset.numel())]}")
                    # Log env 0 specifically
                    env_0_in_idx = (non_terminal_idx == 0).nonzero(as_tuple=False)
                    if env_0_in_idx.numel() > 0:
                        env_0_pos = env_0_in_idx[0].item()
                        self.debug_helper._log(3, f"[compute_derived] Env 0 has {derived_counts_subset[env_0_pos].item()} derived states")
                
                self.next_var_indices.index_copy_(0, non_terminal_idx, updated_var_indices)

                # Step 3: Skip-unary loop (runs BEFORE memory pruning on raw derived states)
                # This matches SB3 behavior exactly where skip_unary evaluates len(derived_states)==1
                # on the RAW states before any memory pruning is applied.
                if self.skip_unary_actions:
                    if verbose:
                        self.debug_helper._log(3, f"[compute_derived] BEFORE skip_unary raw_counts={derived_counts_subset[:min(3, derived_counts_subset.numel())]}")
                    all_derived, derived_counts_subset = self._skip_unary(
                        non_terminal_idx, all_derived, derived_counts_subset
                    )
                    if verbose:
                        self.debug_helper._log(3, f"[compute_derived] after skip_unary counts={derived_counts_subset}")
                
                # Step 4: Postprocess A - validity + proof-safe cut
                # This includes: memory prune non-terminals, padding limits, mark terminals, add non-terminals to memory
                # NOTE: This runs AFTER skip_unary, matching SB3 which applies memory pruning after the skip_unary loop
                if verbose:
                    self.debug_helper._log(3, f"[compute_derived] BEFORE postprocess_A counts={derived_counts_subset[:min(3, derived_counts_subset.numel())]}")
                all_derived, derived_counts_subset = self._postprocess(
                    non_terminal_idx, all_derived, derived_counts_subset
                )
                
                if verbose:
                    self.debug_helper._log(3, f"[compute_derived] AFTER postprocess_A counts={derived_counts_subset[:min(3, derived_counts_subset.numel())]}")

                # Step 5.5: Add END action if configured and room available (for non-terminal rows)
                if self.end_proof_action:
                    # Cap to padding_states - 1 BEFORE adding Endf
                    # This matches SB3 behavior: max_num_states = padding_states - 1, then append Endf
                    max_for_endf = self.padding_states - 1
                    derived_counts_subset = torch.clamp(derived_counts_subset, max=max_for_endf)
                    
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

        # Force a terminal action for depth-capped rows
        # IMPORTANT: Only apply to non-terminal rows. Terminal states (True/False) should NOT
        # be overwritten with Endf - they already have their terminal state as derived.
        if self.max_depth is not None:
            # First identify rows that hit depth cap
            depth_capped = active_mask & (self.current_depths >= self.max_depth)
            
            if depth_capped.any():
                # Exclude rows that are already terminal (True/False/Endf)
                # These have already been handled by create_terminal_derived above
                current_preds = self.current_queries[:, 0, 0]  # First predicate of current state
                already_terminal = self.unification_engine.is_terminal_pred(current_preds)
                
                # Only force terminal action on non-terminal rows that hit depth cap
                depth_mask = depth_capped & ~already_terminal
                
                if depth_mask.any():
                    rows = torch.arange(active_mask.shape[0], device=self._device)[depth_mask]
                    batched_derived[rows] = self.padding_idx
                    # Always use False() for depth-capped rows, NOT Endf()
                    # Endf is an agent-chosen action (to give up) and should never be forced
                    false_state = self.unification_engine.get_false_state()
                    expanded = self._build_single_atom_state(rows.shape[0], false_state)
                    batched_derived[rows, 0] = expanded
                    counts[rows] = 1

        # Safety check: any active env with zero states is a bug since _postprocess should inject FALSE
        need_fallback = active_mask & (counts == 0)
        if need_fallback.any():
            # Log details for debugging
            num_affected = need_fallback.sum().item()
            raise ValueError(f'Bug: {num_affected} active env(s) have zero derived states after postprocess - this should not happen')

        # If proof-end action is enabled, never prefer END over FALSE for single-option failure rows
        # end is only when the agent chooses to end the proof

        # Write back in-place instead of reassigning
        self.derived_states_batch.copy_(batched_derived)
        self.derived_states_counts.copy_(counts)

    def _fit_to_buffer(self, states: Tensor, counts: Tensor, num_envs: int) -> Tuple[Tensor, Tensor]:
        """
        Fit derived states to the environment buffer dimensions [N, S, A, D].
        
        Pads or truncates the states dimension (S) and atoms dimension (A) as needed.
        
        Args:
            states (Tensor):   [N, K_sub, M_sub, D] Input states.
            counts (Tensor):   [N] Counts.
            num_envs (int):    Number of environments N.
            
        Returns:
            Tuple[Tensor, Tensor]: (states, counts) reshaped to [N, S, A, D].
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
        Add END action to derived states for rows that don't have any terminal outcome.
        
        Matches SB3 behavior exactly:
        - If no successor is TRUE/FALSE, append the END action if space permits.
        - This allows the agent to voluntarily terminate a proof path.
        
        Args:
            states (Tensor): [A, K, M, D] Derived states.
            counts (Tensor): [A] Number of valid states per row.
        
        Returns:
            Tuple[Tensor, Tensor]: Updated (states, counts) with END actions added.
        """
        # Guard: if end_pred_idx is not set, we cannot add END actions
        if self.end_pred_idx is None:
            raise ValueError("end_pred_idx is not set; cannot add END actions.")
        
        device = self._device
        pad = self.padding_idx

        # Handle empty states input (K=0 case)
        assert states.numel() > 0, "States tensor is empty; cannot add END actions."
        
        A, K, M, D = states.shape
        
        if K == 0:
             raise RuntimeError('K should not be 0')
        
        first_preds = states[:, :, 0, 0]  # [A, K]
        valid_state_mask = torch.arange(K, device=device).view(1, -1).expand(A, -1) < counts.view(A, 1)
        
        # Check for terminal predicates (True/False) among valid states
        # Match SB3: has_terminal_outcome = any(any(atom.predicate in ('True', 'False') for atom in state) for state in derived_states)
        is_true = (first_preds == self.true_pred_idx) if self.true_pred_idx is not None else torch.zeros_like(first_preds, dtype=torch.bool)
        is_false = (first_preds == self.false_pred_idx) if self.false_pred_idx is not None else torch.zeros_like(first_preds, dtype=torch.bool)
        is_terminal = is_true | is_false
        
        # has_terminal_outcome: any valid state has True or False predicate
        has_terminal_outcome = (is_terminal & valid_state_mask).any(dim=1)
        
        # If count=0, there's no terminal outcome (empty derived states)
        # SB3 would add Endf in this case
        has_terminal_outcome = has_terminal_outcome & (counts > 0)
        
        # Reserve END for rows without terminal outcome and with room
        # Use padding_states (final buffer size) instead of K (engine working buffer)
        # because _fit_to_buffer will expand to padding_states after this
        K_final = self.padding_states
        # Reserve END for rows without terminal outcome and with room
        # Use padding_states (final buffer size) instead of K (engine working buffer)
        K_final = self.padding_states
        reserve_end = ~has_terminal_outcome & (counts < K_final)
        
        # Optimization: Avoid all CPU syncs (no .item(), no .any(), no numel())
        
        # 1. Unconditionally expand to K_final if needed to ensure safety
        # Since we filtered by counts < K_final, we know we only write within [0, K_final-1]
        if K < K_final:
            pad_extra = torch.full((A, K_final - K, M, D), pad, dtype=states.dtype, device=device)
            states = torch.cat([states, pad_extra], dim=1)
        
        # 2. Perform updates using masking/indexing without checking size on CPU
        # Boolean indexing returns a tensor; operations on it are async unless we check size
        rows = torch.arange(A, device=device)[reserve_end]
        
        # Prepare END state singleton [M, D] (broadcastable)
        end_state = torch.full((M, D), pad, dtype=torch.long, device=device)
        end_atom = self.unification_engine.get_end_state()
        end_state[0] = end_atom[0]
        
        # execute update - broadcasting handles the dimension matching for rows
        # Direct assignment with empty rows is a no-op in PyTorch, avoiding explicit size check (sync)
        pos = counts[rows]
        states[rows, pos] = end_state
        counts[rows] = pos + 1
        
        return states, counts

    # ---------------------------------------------------------------------
    # Skip-unary closure (batched; bounded loop on subset only)
    # ---------------------------------------------------------------------

    def _skip_unary(self, idx_subset: Tensor, derived_states: Tensor, derived_counts: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Skip-unary optimization matching SB3's behavior exactly.
        
        When there's only one valid action, skip it automatically since the agent
        has nothing to learn (no decision to make). This chains through multiple
        unary levels with full memory tracking.
        
        NOTE: For maximum speed, disable skip_unary_actions=False in env config.
        Unary actions have log_prob=0 so they don't affect policy learning.
        """
        if derived_states.numel() == 0:
            return derived_states, derived_counts

        pad = self.padding_idx
        A, K, M, D = derived_states.shape
        device = self._device
        verbose = self.verbose >= 3

        iters = 0
        
        # Helper: identify rows with exactly one non-terminal child
        def _unary_nonterminal_mask(ds: Tensor, dc: Tensor) -> Tensor:
            """Returns mask [A] for rows that have exactly one non-terminal child."""
            if ds.numel() == 0:
                return torch.zeros(A, dtype=torch.bool, device=device)
            
            is_single = (dc == 1)
            
            # Check the first child's first predicate
            fp = ds[:, 0, 0, 0]  # [A] first predicate of first child
            is_term = self.unification_engine.is_terminal_pred(fp)
            
            return is_single & ~is_term
        
        # Initial check on RAW derived states (before any memory pruning)
        unary_mask = _unary_nonterminal_mask(derived_states, derived_counts)
        unary_idx = torch.arange(A, device=device)[unary_mask]

        if self.verbose >= 2:
            for i in range(min(3, unary_idx.numel())):
                row = unary_idx[i]
                env_idx = idx_subset[row].item()
                current_state = self.current_queries[env_idx]
                current_str = self.debug_helper._format_atoms(current_state)
                derived_state = derived_states[row, 0]
                derived_str = self.debug_helper._format_atoms(derived_state)
                self.debug_helper._log(2, f"[skip_unary] Unary in idx {env_idx}: current={current_str}, derived={derived_str}")

        if verbose:
            self.debug_helper._log(3, f"[skip_unary] Starting with {unary_idx.numel()} unary rows")

        # Bounded iterations - matches SB3's counter > 20 check
        while unary_idx.numel() > 0 and iters < self.max_skip_unary_iters:
            uidx = unary_idx  # [U] indices into the A-dimension
            env_rows = idx_subset.index_select(0, uidx)  # [U] actual env indices
            if env_rows.dtype != torch.long:
                env_rows = env_rows.long()

            if verbose:
                self.debug_helper._log(3, f"[skip_unary] Iteration {iters}: processing {uidx.numel()} rows")

            # Step a: Promote the single child to current_state
            promoted = derived_states[uidx, 0]  # [U, M, D]
            if promoted.dtype != self.current_queries.dtype:
                promoted = promoted.long()

            # Ensure promoted states have correct dimensions
            if promoted.shape[1] > self.padding_atoms:
                promoted = promoted[:, :self.padding_atoms]
            elif promoted.shape[1] < self.padding_atoms:
                pad_rows = self.padding_atoms - promoted.shape[1]
                pad_tail = torch.full(
                    (promoted.shape[0], pad_rows, promoted.shape[2]),
                    pad, dtype=promoted.dtype, device=device
                )
                promoted = torch.cat([promoted, pad_tail], dim=1)
            
            # Check if promoted states are terminal - if so, stop processing them
            terminal_mask = self.unification_engine.is_terminal_state(promoted)
            non_terminal_rows = env_rows[~terminal_mask]
            uidx_non_terminal = uidx[~terminal_mask]
            promoted_non_terminal = promoted[~terminal_mask]
            
            if verbose:
                self.debug_helper._log(3, f"[skip_unary] After promotion: {terminal_mask.sum().item()} became terminal")

            # For terminal rows, set their derived to just themselves
            terminal_sum = terminal_mask.sum()
            if terminal_sum > 0:
                terminal_uidx = uidx[terminal_mask]
                terminal_promoted = promoted[terminal_mask]
                
                terminal_derived, terminal_counts = self.unification_engine.create_terminal_derived(
                    terminal_promoted, K, M)
                
                terminal_derived_subset = terminal_derived[:, :K, :M, :]
                derived_states.index_copy_(0, terminal_uidx, terminal_derived_subset)
                derived_counts[terminal_uidx] = terminal_counts

            # Step b: Expand non-terminal promoted states
            if non_terminal_rows.numel() > 0:
                sub_derived, sub_counts, sub_next = self.unification_engine.get_derived_states(
                    current_states=promoted_non_terminal,
                    next_var_indices=self.next_var_indices.index_select(0, non_terminal_rows),
                    excluded_queries=self.original_queries.index_select(0, non_terminal_rows).unsqueeze(1),
                    verbose=self.prover_verbose
                )
                self.next_var_indices.index_copy_(0, non_terminal_rows, sub_next)

                # Memory pruning and max_atoms filtering via _postprocess
                sub_derived, sub_counts = self._postprocess(
                    non_terminal_rows, sub_derived, sub_counts, 
                    current_states=promoted_non_terminal
                )

                if verbose:
                    self.debug_helper._log(3, f"[skip_unary] After postprocess: counts={sub_counts[:min(3, sub_counts.shape[0])]}")

                # Fit to current K, M dimensions
                U = uidx_non_terminal.numel()
                Ks = sub_derived.shape[1] if sub_derived.numel() > 0 else 0
                Ms = sub_derived.shape[2] if sub_derived.numel() > 0 else M

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
            
            # Recompute unary mask on POST-FILTERED derived states
            unary_mask = _unary_nonterminal_mask(derived_states, derived_counts)
            unary_idx = torch.arange(A, device=device)[unary_mask]
            iters += 1

            if verbose:
                self.debug_helper._log(3, f"[skip_unary] End of iteration {iters}: {unary_idx.numel()} rows remain unary")

        # If we hit the iteration cap but still have unary rows,
        # match SB3 semantics: mark those rows as False()
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

            # Set their derived states to be the terminal False() state only
            terminal_derived, terminal_counts = self.unification_engine.create_terminal_derived(
                false_batch, K, M)
            terminal_subset = terminal_derived[:, :K, :M, :]
            derived_states.index_copy_(0, unary_idx, terminal_subset)
            derived_counts.index_copy_(0, unary_idx, terminal_counts)
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
    def _postprocess(
        self, 
        env_indices: Tensor, 
        states: Tensor, 
        counts: Tensor, 
        current_states: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Proof-safe validity filtering and memory pruning.
        
        Logic:
        1. Add current state to memory (to prevent immediate cycles).
        2. Validate states (non-empty, within atom budget).
        3. Prune visited states via memory check (except terminals, which are safe).
        4. Compact derived states (remove pruned holes).
        5. Inject FALSE state if no valid successors remain.
        
        Args:
            env_indices (Tensor):    [A] Indices of active environments.
            states (Tensor):         [A, K, M, D] Derived states.
            counts (Tensor):         [A] Derived counts.
            current_states (Tensor): [A, M, D] Optional override for current state 
                                     (used during skip-unary promotion).
                                     
        Returns:
            Tuple[Tensor, Tensor]: (compact_states, new_counts)
        """
        device = self._device
        pad = self.padding_idx

        if states.numel() == 0:
            return states, counts

        # Step 0: Add current state to memory before checking derived states
        # This prevents loops (derived states that match current state will be pruned)
        if self.memory_pruning:
            state_to_add = current_states if current_states is not None else self.current_queries
            self.memory_backend.add_current(env_indices, state_to_add)

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
        keep_mask = base_valid  # Will be updated via tensor ops, no in-place mutation
        
        if self.memory_pruning:
            owners = env_indices
            if owners.dtype != torch.long:
                owners = owners.long()
            
            # OPTIMIZATION: Call membership unconditionally - it handles empty case
            visited = self.memory_backend.membership(states, owners)  # [A, K]
            visited = visited & base_valid
            
            # Only prune non-terminal visited states (terminals are protected)
            prune_mask = visited & ~is_terminal_state
            keep_mask = base_valid & ~prune_mask
            
            # Verbose logging - only runs when enabled (rare)
            if self.verbose >= 2:
                prune_sum = prune_mask.sum()
                if prune_sum > 0:
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

        pos = torch.cumsum(keep_mask.long(), dim=1) - 1  # [A, K]
        pos = torch.clamp(pos, min=0, max=K - 1)
        batch_idx = torch.arange(A, device=device).view(A, 1).expand(A, K)

        # OPTIMIZATION: Compute unconditionally - scatter handles empty case
        compact = torch.full((A, K, M, D), pad, dtype=states.dtype, device=device)
        target_rows = batch_idx[keep_mask]
        target_pos = pos[keep_mask]
        if target_rows.numel() > 0:
            compact[target_rows, target_pos] = states[keep_mask]
        
        counts_out = new_counts

        # Step 5: Inject FALSE where needed - any row with 0 states needs FALSE
        # This matches SB3 behavior where empty derived_states (after memory pruning) -> end_in_false()
        needs_false = new_counts == 0
        false_sum = needs_false.sum()
        if false_sum > 0:
            # Create FALSE state with correct M dimension to match compact
            false_state = torch.full((M, D), pad, dtype=states.dtype, device=device)
            false_state[0, 0] = self.false_pred_idx
            # Use advanced indexing to assign without .item() sync
            compact[needs_false, 0] = false_state.unsqueeze(0)
            counts_out[needs_false] = 1

        # Note: We do NOT add derived states to memory here.
        # Derived states are added to memory only when they become the current state
        # (in _step or _skip_unary when promoted). This prevents the correct tabling behavior.

        return compact, counts_out


    # ---------------------------------------------------------------------
    # Reward / Done
    # ---------------------------------------------------------------------
    def _get_done_reward(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Compute rewards and termination flags for the current state.
        
        Logic:
        - Terminated: Natural proof conclusion (TRUE/FALSE) or explicit END action.
        - Truncated: Depth limit exceeded without natural termination.
        - Success: State proves TRUE (matches label=1 in some reward modes).
        - Rewards: Computed based on `reward_type` (Binary, Sparse, Dense, etc.).
        
        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]:
            - rewards:      [B, 1] Float rewards.
            - terminated:   [B, 1] Boolean natural termination.
            - truncated:    [B, 1] Boolean depth truncation.
            - is_success:   [B, 1] Boolean proof success status.
        """
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
            rewards[tn, 0] = self.rejection_weight
        else:
            raise ValueError(f"Invalid reward_type: {self.reward_type}. Choose 0-4.")

        is_success[:, 0] = success
        return rewards, terminated, truncated, is_success

    # ---------------------------------------------------------------------
    # Observation packing
    # ---------------------------------------------------------------------
    def _create_observation_dict(self) -> Dict[str, Tensor]:
        """
        Pack internal state into an observation dictionary.
        
        Returns:
            Dict[str, Tensor]:
            - sub_index:            [B, 1, A, D] Current state
            - derived_sub_indices:  [B, S, A, D] Successors
            - action_mask:          [B, S] Valid actions mask
            - label:                [B] Ground truth label
            - query_depth:          [B] Current depth
        """
        B = self.batch_size_int
        S = self.padding_states
        device = self._device
        action_indices = torch.arange(S, device=device).view(1, S).expand(B, S)
        counts = self.derived_states_counts.view(B, 1)
        action_mask = action_indices < counts # [B, S]
        
        # Debug action mask creation
        if self.debug_config.is_enabled('env', level=2):
            self._debug_observation_creation(action_mask, counts)        
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

    def _build_single_atom_state(self, num_rows: int, atom: Tensor) -> Tensor:
        """
        Build a padded state batch with a single non-padding atom in position 0.
        
        Args:
            num_rows (int): Batch size N.
            atom (Tensor):  [1, D] or [D] Atom tensor.
            
        Returns:
            Tensor: [N, A, D] Padded state tensor with the atom at index 0.
        """
        state = torch.full(
            (num_rows, self.padding_atoms, self.max_arity + 1),
            self.padding_idx,
            dtype=torch.long,
            device=self._device,
        )
        atom = atom.to(device=self._device, dtype=torch.long)
        if atom.dim() == 2 and atom.shape[0] == 1:
            atom = atom[0]
        state[:, 0] = atom
        return state

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------
    def _ensure_query_tensor(self, queries: Tensor) -> Tensor:
        """
        Validate and pad the queries tensor to [N, padding_atoms, max_arity+1].
        
        Args:
            queries (Tensor): [N, L, D] Initial query tensor.
            
        Returns:
            Tensor: [N, A, D] Standardized query tensor.
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
            raise ValueError("Query atom count exceeds padding_atoms."
                             f" Shape of queries: {queries.shape}."
                             f" Got {cur_atoms}, expected at most {self.padding_atoms}.")

        return queries

    def _ensure_vector(self, data: Optional[Tensor], expected_len: int, name: str) -> Tensor:
        """
        Validate and standardize a 1D tensor to [expected_len].
        
        Args:
            data (Optional[Tensor]): Input data or None.
            expected_len (int):      Expected length N.
            name (str):              Debug name.
            
        Returns:
            Tensor: [N] 1D tensor on the correct device.
        """
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
        """
        Extract the first valid non-padding atom from each query.
        
        Args:
            padded_queries (Tensor): [N, A, D] Padded query states.
            
        Returns:
            Tensor: [N, D] First valid atom for each query.
        """
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

    def _update_original_queries_for_indices(self, indices: Tensor) -> None:
        """
        Update `original_queries` to match the current state for specific indices.
        
        This is used when a new episode starts (after reset) to sync the 'original'
        reference query used for loop detection and negative sampling.
        
        Args:
            indices (Tensor): [SubBatch] Indices of environments to update.
        """
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

    def _set_seed(self, seed: Optional[int]) -> None:
        """
        Set the seed for environment-local random operations.
        
        Args:
            seed (Optional[int]): Input seed (or None to generate random).
        
        Note:
            We intentionally do NOT call torch.manual_seed() here because that
            would modify the global RNG state, which can cause issues with reproducibility
            in multi-env settings. Instead, we store the seed and use local RNGs for
            operations like negative sampling.
        """
        import random as _random
        if seed is None:
            seed = _random.randint(0, 2**31 - 1)
        self.seed = int(seed)
        # Create a local Python RNG for this environment (like sb3_env.py)
        self.seed_gen = _random.Random(self.seed)

    # ===================== Evaluation =====================

    def set_eval_dataset(
        self,
        queries: torch.Tensor,        # [M, A, D]
        labels: torch.Tensor,         # [M]
        query_depths: torch.Tensor,   # [M]
        per_slot_lengths: Optional[torch.Tensor] = None,  # [B] (first Q > 0, rest 0)
    ) -> None:
        """
        Preload evaluation dataset and optional per-slot schedule.
        
        This method supports two modes:
        1. **Global Sequential**: `per_slot_lengths` is None. All environments pull sequentially
           from the dataset using a shared global pointer.
        2. **Per-Slot Partitioned**: `per_slot_lengths` is provided. The dataset is logically
           concatenated from `B` independent chunks. Each environment `i` pulls from
           its own dedicated chunk `chunk_i`.
        
        Args:
            queries (Tensor):          [M, A, D] Dataset queries (M total items).
            labels (Tensor):           [M] Labels.
            query_depths (Tensor):     [M] Target derivation depths.
            per_slot_lengths (Tensor): [B] Length of the chunk for each slot.
                                       If provided, M must equal sum(per_slot_lengths).
        """
        device = self._device
        
        # Switch to eval mode (required for per-slot scheduling to work)
        self.mode = 'eval'
        # Note: memory_pruning is set at init and applies to both train and eval modes
        
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
            # Avoid .item() sync - validate using tensor ops only
            total = per_slot_lengths.sum()
            if total != self._num_all:
                raise ValueError(f"Sum(per_slot_lengths)={total.item()} must equal #items M={self._num_all}.")

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
    
    # ---------------------------------------------------------------------
    # Debug methods for action space analysis
    # ---------------------------------------------------------------------
    def _debug_action_space(self, obs, dones, is_success):
        """Debug output to understand why action space shrinks."""
        B = self.batch_size_int
        n_show = min(self.debug_config.debug_sample_envs or 5, B)
        
        print(f"\n{self.debug_config.debug_prefix} [ENV ACTION SPACE]")
        
        action_mask = obs['action_mask']
        valid_actions = action_mask.sum(dim=-1)
        
        # Overall statistics
        print(f"  Valid actions: mean={valid_actions.float().mean():.2f}, "
              f"min={valid_actions.min():.0f}, max={valid_actions.max():.0f}")
        
        # Per-environment analysis
        for i in range(n_show):
            n_valid = valid_actions[i].item()
            depth = self.current_depths[i].item()
            is_done = dones[i].item() if dones.dim() > 1 else dones[i].squeeze().item()
            success = is_success[i].item() if is_success.dim() > 1 else is_success[i].squeeze().item()
            
            # Get current state info
            current_pred_idx = self.current_queries[i, 0, 0].item()
            
            # Check if terminal
            is_terminal = ""
            if current_pred_idx == self.true_pred_idx:
                is_terminal = " [TRUE]"
            elif current_pred_idx == self.false_pred_idx:
                is_terminal = " [FALSE]"
            elif self.end_pred_idx is not None and current_pred_idx == self.end_pred_idx:
                is_terminal = " [END]"
            elif depth >= self.max_depth:
                is_terminal = " [MAX_DEPTH]"
            
            # Get string representation if possible
            state_str = ""
            if hasattr(self, 'debug_helper') and self.debug_helper is not None:
                try:
                    state_str = self.debug_helper.state_to_str(self.current_queries[i])
                    if len(state_str) > 80:
                        state_str = state_str[:77] + "..."
                except:
                    state_str = f"pred_idx={current_pred_idx}"
            else:
                state_str = f"pred_idx={current_pred_idx}"
            
            print(f"  Env {i}: n_valid={n_valid}, depth={depth}/{self.max_depth}, "
                  f"done={is_done}, success={success}{is_terminal}")
            print(f"    State: {state_str}")
            
            # Show why actions are limited
            if n_valid == 1:
                print(f"    → Only 1 action: likely terminal or dead-end state")
            elif n_valid < 3 and not is_terminal:
                print(f"    → Few actions: memory pruning or limited successors")
    
    def _debug_observation_creation(self, action_mask, counts):
        """Debug observation creation to trace action masking."""
        B = self.batch_size_int
        n_show = min(self.debug_config.debug_sample_envs or 5, B)
        
        print(f"\n{self.debug_config.debug_prefix} [ENV OBSERVATION CREATION]")
        
        for i in range(n_show):
            n_derived = self.derived_states_counts[i].item()
            n_valid = action_mask[i].sum().item()
            
            print(f"  Env {i}: derived_states={n_derived}, valid_actions={n_valid}")
            
            # Check for memory pruning effect
            if n_derived > 0 and n_valid < n_derived:
                print(f"    → Some derived states were masked (memory pruning?)")
            
            # Show a few derived states if available
            if n_derived > 0 and hasattr(self, 'debug_helper') and self.debug_helper is not None:
                try:
                    n_to_show = min(3, n_derived)
                    for j in range(n_to_show):
                        derived_str = self.debug_helper.state_to_str(self.derived_states_batch[i, j])
                        if len(derived_str) > 60:
                            derived_str = derived_str[:57] + "..."
                        print(f"      Action {j}: {derived_str}")
                except:
                    pass
