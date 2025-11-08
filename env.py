import torch
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Set
from tensordict import TensorDict
from torchrl.envs import EnvBase

from atom_stringifier import AtomStringifier
from unification_engine import UnificationEngine


class BatchedEnv(EnvBase):
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
        corruption_scheme: List[str] = ['head', 'tail'],
        train_neg_ratio: float = 1.0,
        # Env related params
        max_depth: int = 10,
        memory_pruning: bool = True,
        end_proof_action: bool = False,
        skip_unary_actions: bool = False,
        reward_type: int = 1,
        verbose: int = 0,
        prover_verbose: int = 0,
        device: torch.device = None,
    ):
        # Store batch_size as int for internal use
        self.batch_size_int = batch_size
        
        # Initialize parent with torch.Size([batch_size])
        super().__init__(
            device=device if device is not None else torch.device('cpu'),
            batch_size=torch.Size([batch_size]),
        )
        
        self._device = torch.device('cpu') if device is None else device

        # Store parameters
        self.mode = mode
        self.counter = 0 # Counter for sampling in eval/train
        self.max_depth = max_depth
        # Disable memory pruning for eval mode (it's mainly for training)
        self.memory_pruning = memory_pruning if mode == 'train' else False
        self.padding_atoms = padding_atoms
        self.padding_states = padding_states
        self.reward_type = reward_type
        self.verbose = verbose
        self.prover_verbose = prover_verbose
        
        
        # Store queries and labels
        self.all_queries = queries
        self.all_labels = labels
        self.all_depths = query_depths
        
        # Store parameters from arguments
        self.max_arity = max_arity
        self.padding_idx = padding_idx
        self.true_pred_idx = true_pred_idx
        self.false_pred_idx = false_pred_idx
        self.end_pred_idx = end_pred_idx
        self.runtime_var_start_index = runtime_var_start_index
        self.total_vocab_size = total_vocab_size
        
        # Store unification engine
        self.unification_engine = unification_engine
        
        # Store atom stringifier for debugging
        self.atom_stringifier = atom_stringifier
        
        # Negative sampling
        self.corruption_mode = corruption_mode
        self.corruption_scheme = corruption_scheme
        self.train_neg_ratio = train_neg_ratio
        self.sampler = sampler
        self.neg_counter = 0
        self.negation_toggle = 0  # For alternating head/tail
        
        # Action modifiers
        self.end_proof_action = end_proof_action
        self.skip_unary_actions = skip_unary_actions
        self.max_skip_unary_iters = 20
                
        # Create End tensor if needed
        if self.end_pred_idx is not None and self.end_pred_idx >= 0:
            self.end_tensor = torch.tensor(
                [[self.end_pred_idx, self.padding_idx, self.padding_idx]],
                dtype=torch.long,
                device=self._device
            )
        else:
            self.end_tensor = None

        # Memory (for pruning)
        self.memories: List[Set] = [set() for _ in range(self.batch_size_int)]

        # Batch state
        self.current_queries: Tensor = torch.full(
            (self.batch_size_int, padding_atoms, self.max_arity + 1),
            self.padding_idx, dtype=torch.long, device=self._device
        )
        self.current_labels: Tensor = torch.zeros(self.batch_size_int, dtype=torch.long, device=self._device)
        self.current_depths: Tensor = torch.zeros(self.batch_size_int, dtype=torch.long, device=self._device)
        self.next_var_indices: Tensor = torch.full(
            (self.batch_size_int,), self.runtime_var_start_index, dtype=torch.long, device=self._device
        )

        self.original_queries: Tensor = torch.full(
            (self.batch_size_int, self.max_arity + 1),
            self.padding_idx, dtype=torch.long, device=self._device
        )

        # Derived states: [B, max_derived, padding_atoms, arity+1]
        self.derived_states_batch: Tensor = torch.full(
            (self.batch_size_int, padding_states, padding_atoms, self.max_arity + 1),
            self.padding_idx, dtype=torch.long, device=self._device
        )
        self.derived_states_counts: Tensor = torch.zeros(self.batch_size_int, dtype=torch.long, device=self._device)


        # Define observation and action specs
        self._make_specs()
    
    def _make_specs(self):
        """Define observation and action space specs."""
        from torchrl.data import Composite, Bounded
        
        max_vocab_size = self.total_vocab_size
        
        # Observation spec - batched version
        self.observation_spec = Composite(
            sub_index=Bounded(
                low=-1,
                high=max_vocab_size,
                shape=torch.Size([self.batch_size_int, 1, self.padding_atoms, self.max_arity + 1]),
                dtype=torch.int64,
                device=self._device,
            ),
            derived_sub_indices=Bounded(
                low=-1,
                high=max_vocab_size,
                shape=torch.Size([self.batch_size_int, self.padding_states, self.padding_atoms, self.max_arity + 1]),
                dtype=torch.int64,
                device=self._device,
            ),
            action_mask=Bounded(
                low=0,
                high=1,
                shape=torch.Size([self.batch_size_int, self.padding_states]),
                dtype=torch.bool,
                device=self._device,
            ),
            shape=torch.Size([self.batch_size_int]),
        )
        
        # Action spec - batched
        self.action_spec = Bounded(
            low=0,
            high=self.padding_states - 1,
            shape=torch.Size([self.batch_size_int]),
            dtype=torch.int64,
            device=self._device,
        )
        
        # Reward spec
        self.reward_spec = Bounded(
            low=-float('inf'),
            high=float('inf'),
            shape=torch.Size([self.batch_size_int, 1]),
            dtype=torch.float32,
            device=self._device,
        )
        
        # Done spec (terminated)
        self.done_spec = Bounded(
            low=0,
            high=1,
            shape=torch.Size([self.batch_size_int, 1]),
            dtype=torch.bool,
            device=self._device,
        )
        
        # Truncated spec (for time limits)
        self.truncated_spec = Bounded(
            low=0,
            high=1,
            shape=torch.Size([self.batch_size_int, 1]),
            dtype=torch.bool,
            device=self._device,
        )

    # ------------------- Debug helpers -------------------

    def _debug_state_to_str(self, state_idx: Tensor, oneline: bool = False) -> str:
        """Convert state indices to string for debugging."""
        if self.atom_stringifier is None:
            return f"<state with shape {state_idx.shape}>"
        if state_idx.dim() == 1:
            # Single atom [3]
            return self.atom_stringifier.atom_to_str(state_idx)
        elif state_idx.dim() == 2:
            # State [k, 3]
            return self.atom_stringifier.state_to_str(state_idx)
        else:
            return f"<state with shape {state_idx.shape}>"
    
    def _debug_states_to_str(self, states: List[Tensor]) -> str:
        """Convert list of states to string for debugging."""
        if self.atom_stringifier is None:
            return f"<{len(states)} states>"
        return "[" + ", ".join([self._debug_state_to_str(s) for s in states]) + "]"

    def _reset(self, tensordict: Optional[TensorDict] = None) -> TensorDict:
        """Reset all or some environments in the batch.
        
        Args:
            tensordict: Optional TensorDict with "_reset" key indicating which envs to reset.
                       If None or no "_reset" key, resets all environments.
        """

        # Determine which environments to reset
        if tensordict is not None and "_reset" in tensordict.keys():
            # Partial reset - only reset environments where _reset is True
            reset_mask = tensordict["_reset"].squeeze(-1).bool()  # [B]
            indices_to_reset = torch.where(reset_mask)[0].tolist()
            if len(indices_to_reset) == 0:
                raise ValueError("No environments to reset in partial reset.")
        else:
            print('\n\n-----------------------------Reset-----------------------------') if self.verbose else None
            # Full reset - reset all environments
            reset_mask = torch.ones(self.batch_size_int, dtype=torch.bool, device=self._device)
            indices_to_reset = list(range(self.batch_size_int))
            num_to_reset = self.batch_size_int
        
        # Sample queries for environments that need reset
        num_to_reset = len(indices_to_reset)
        batch_queries, batch_labels, batch_depths = [], [], []
        
        if self.mode == 'train':
            # Sample with replacement
            idxs = torch.randint(0, len(self.all_queries), (num_to_reset,))
            for i in idxs.tolist():
                batch_queries.append(self.all_queries[i])
                batch_labels.append(self.all_labels[i])
                batch_depths.append(self.all_depths[i])
            
            # Negative sampling via Bernoulli
            if self.corruption_mode and self.sampler is not None and self.train_neg_ratio > 0:
                p_neg = float(self.train_neg_ratio) / (1.0 + float(self.train_neg_ratio))
                neg_mask = (torch.rand(num_to_reset, device=self._device) < p_neg)
                for k in range(num_to_reset):
                    if neg_mask[k]:
                        q = batch_queries[k]
                        atom = q[0] if q.dim() == 2 else q  # [3]
                        corrupted = self.sampler.corrupt(atom.unsqueeze(0), num_negatives=1, device=self._device)
                        batch_queries[k] = corrupted[0, 0].unsqueeze(0)  # [1,3]
                        batch_labels[k] = 0
        
        elif self.mode == 'eval':
            # Sequential sampling
            for _ in range(num_to_reset):
                if self.counter < len(self.all_queries):
                    batch_queries.append(self.all_queries[self.counter])
                    batch_labels.append(self.all_labels[self.counter])
                    batch_depths.append(self.all_depths[self.counter])
                    self.counter += 1
                else:
                    # Wrap around if we run out
                    batch_queries.append(self.all_queries[0])
                    batch_labels.append(self.all_labels[0])
                    batch_depths.append(self.all_depths[0])
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        # Update state only for environments being reset
        for i, env_idx in enumerate(indices_to_reset):
            q = batch_queries[i]
            q = q.unsqueeze(0) if q.dim() == 1 else q  # [1,3] if needed
            self.current_queries[env_idx] = self._pad_state(q)
            self.current_labels[env_idx] = batch_labels[i]
            self.current_depths[env_idx] = 0
            self.next_var_indices[env_idx] = self.runtime_var_start_index
            self.memories[env_idx] = set()

        # Update original queries only for reset indices
        self._update_original_queries_for_indices(indices_to_reset)

        if self.verbose >=1:
            batch_indices = torch.tensor(indices_to_reset, device=self._device)
            first_atom_indices = [] 
            for i in range(len(batch_indices)):
                q = self.current_queries[batch_indices[i]]
                first_atom_indices.append(q[0] if q.dim() == 2 else q)
            print(f"\nInitial Query (Label: {self.current_labels[batch_indices]}, depth: {self.current_depths[batch_indices]}): "
                                   f"{self._debug_state_to_str(self.current_queries[batch_indices, first_atom_indices], oneline=True)}")

        # Compute derived states only for reset envs
        self._compute_derived_states(active_mask=reset_mask, clear_inactive=False)
        
        if self.verbose >= 1:
            print(f"\nReset States:")
            for i in range(self.batch_size_int):
                if reset_mask[i]:  # Only show for reset environments
                    print(f"  Env {i}: {self._debug_state_to_str(self.current_queries[i], oneline=True)}")
                    derived_count = self.derived_states_counts[i].item()
                    derived_states_list = [self.derived_states_batch[i, j] for j in range(derived_count)]
                    print(f"  Derived States ({derived_count}): {self._debug_states_to_str(derived_states_list)}")
        
        if self.verbose >= 1:
            print(f"\n[DEBUG _reset]")
            print(f"  Reset mask: {reset_mask.tolist()}")
            print(f"  Reset indices: {indices_to_reset}")
            print(f"  After _compute_derived_states_vec:")
            print(f"  derived_states_counts: {self.derived_states_counts.tolist()}")
        
        # Create batched observations
        return self._create_observation()
    
    def _step(self, tensordict: TensorDict) -> TensorDict:
        """Execute a batch of actions - FULLY VECTORIZED."""
        actions = tensordict["action"]  # [B]
        
        if self.verbose >= 1:
            print(f'\n\n-----------------------------Step-----------------------------')     
        
        if self.verbose >= 2:
            print(f'\n=== BEFORE ACTION ===')
            for i in range(self.batch_size_int):
                print(f'\nEnv {i}:')
                print(f'  Current state: {self._debug_state_to_str(self.current_queries[i], oneline=True)}')
                print(f'  Available derived states ({self.derived_states_counts[i].item()}):{self._debug_states_to_str([self.derived_states_batch[i, j] for j in range(self.derived_states_counts[i].item())])}')
                print(f'  Selected action: {actions[i].item()}')
        
        
        # if there are any invalid actions, raise an error
        invalid_mask = actions >= self.derived_states_counts  # [B]
        if invalid_mask.any():
            raise ValueError(f"Invalid actions selected: {actions[invalid_mask].tolist()} with counts {self.derived_states_counts[invalid_mask].tolist()}")

        
        # Apply actions [B, padding_atoms, arity+1]
        batch_indices = torch.arange(self.batch_size_int, device=self._device)
        selected_states = self.derived_states_batch[batch_indices, actions]
        self.current_queries = selected_states
        self.current_depths += 1
        # Reward, terminated, & truncated
        rewards, terminated, truncated = self._get_done_reward()
        
        # done is the combination of terminated and truncated
        dones = terminated | truncated
        
        if self.verbose >= 2:
            print(f'\n=== REWARD & DONE ===')
            for i in range(self.batch_size_int):
                print(f'Env {i}:   Reward: {rewards[i].item():.4f}.  Done: {dones[i].item()}. Terminated: {terminated[i].item()}. Truncated: {truncated[i].item()}')
        
        # Next derived states for non-done envs
        active_mask = ~dones.squeeze(-1)
        self._compute_derived_states(active_mask=active_mask, clear_inactive=True)
        
        # Create observation
        obs_dict = self._create_observation_dict()
        
        td = TensorDict(
            {**obs_dict, "reward": rewards, "done": dones, "terminated": terminated, "truncated": truncated},
            batch_size=self.batch_size, device=self._device
        )
        return td

    # ------------------- Unified derived-states compute -------------------

    def _compute_derived_states(self, active_mask: Optional[torch.Tensor] = None, clear_inactive: bool = False):
        """
        Compute derived states for a subset of envs (active_mask).
        - Calls the UnificationEngine ONLY on the active subset.
        - Batched skip-unary closure collapses chains of unary-only transitions.
        - Writes results back into the full batched buffers.
        - Optionally clears inactive envs' derived states.
        Optimized with vectorized operations where possible.
        """
        if active_mask is None:
            active_mask = torch.ones(self.batch_size_int, dtype=torch.bool, device=self._device)
        idx = torch.nonzero(active_mask, as_tuple=False).squeeze(1)

        # Start from previous buffers and update only active rows
        batched_derived = self.derived_states_batch.clone()
        counts = self.derived_states_counts.clone()

        if idx.numel() > 0:
            # First UE call for active subset
            all_derived, updated_var_indices = self.unification_engine.get_derived_states(
                current_states=self.current_queries[idx],
                next_var_indices=self.next_var_indices[idx],
                excluded_queries=self.original_queries[idx],
                labels=self.current_labels[idx],
                verbose=self.prover_verbose,
            )
            # Update var indices for active subset
            self.next_var_indices[idx] = updated_var_indices

            # Batched skip-unary closure (in-place mutates current_queries & memories for those envs)
            if self.skip_unary_actions:
                all_derived = self._skip_unary(idx, all_derived)

            # Process and pack per-env (vectorize padding operation)
            env_idx_list = idx.tolist()
            for k, env_idx in enumerate(env_idx_list):
                processed = self._postprocess(env_idx, all_derived[k], stage="final")
                c = min(len(processed), self.padding_states)
                counts[env_idx] = c
                
                # Clear this environment's derived states
                batched_derived[env_idx].fill_(self.padding_idx)
                
                # Batch pad states if multiple states to process
                if c > 0:
                    for j in range(c):
                        batched_derived[env_idx, j] = self._pad_state(processed[j])

        # Vectorized clear for inactive rows
        if clear_inactive:
            inactive_mask = ~active_mask
            if inactive_mask.any():
                batched_derived[inactive_mask] = self.padding_idx
                counts[inactive_mask] = 0

        self.derived_states_batch = batched_derived
        self.derived_states_counts = counts

    # -------- Batched skip-unary closure (new) --------

    def _skip_unary(self, idx_subset: torch.Tensor, derived_lists: List[List[Tensor]]) -> List[List[Tensor]]:
        """Batched skip-unary closure with vectorized checks where possible."""
        if not derived_lists:
            return derived_lists

        env_idxs = idx_subset.tolist()
        env_to_pos = {env: pos for pos, env in enumerate(env_idxs)}

        def _is_unary_nonterminal(lst: List[Tensor]) -> bool:
            """Check if list contains exactly one non-terminal state."""
            if len(lst) != 1:
                return False
            st = lst[0]
            if st is None or st.numel() == 0:
                return False
            pred = st[0, 0]
            # Vectorized comparison - no .item() needed for comparison
            is_terminal = (pred == self.true_pred_idx) | (pred == self.false_pred_idx)
            if self.end_pred_idx is not None:
                is_terminal = is_terminal | (pred == self.end_pred_idx)
            return not is_terminal.item()

        iters = 0
        while iters < self.max_skip_unary_iters:
            sel_envs = []
            states_to_check = []  # Collect states for batch atom budget check
            
            for pos, env in enumerate(env_idxs):
                lst = derived_lists[pos]
                if not _is_unary_nonterminal(lst):
                    continue
                st = lst[0]
                states_to_check.append((pos, env, st))
            
            if not states_to_check:
                break
            
            # Vectorized atom budget check
            for pos, env, st in states_to_check:
                if st.shape[0] > self.padding_atoms:
                    # Force FALSE and stop skipping
                    derived_lists[pos] = [self._create_false_state()]
                    continue
                # Promote single successor to current state
                self.current_queries[env] = self._pad_state(st)
                if self.memory_pruning:
                    self.memories[env].add(self._state_signature(self.current_queries[env]))
                sel_envs.append(env)

            if not sel_envs:
                break

            sel_tensor = torch.tensor(sel_envs, dtype=torch.long, device=self._device)
            sub_derived, sub_var_idx = self.unification_engine.get_derived_states(
                current_states=self.current_queries[sel_tensor],
                next_var_indices=self.next_var_indices[sel_tensor],
                excluded_queries=self.original_queries[sel_tensor],
                labels=self.current_labels[sel_tensor],
                verbose=self.prover_verbose,
            )
            self.next_var_indices[sel_tensor] = sub_var_idx

            # Reuse the exact same logic (intermediate stage: no END, no #states truncation)
            for j, env in enumerate(sel_envs):
                pos = env_to_pos[env]
                derived_lists[pos] = self._postprocess(env, sub_derived[j], stage="intermediate")

            iters += 1

        # If we hit the cap, force termination for any env still stuck in unary-only
        if iters >= self.max_skip_unary_iters:
            for pos, env in enumerate(env_idxs):
                if _is_unary_nonterminal(derived_lists[pos]):
                    derived_lists[pos] = [self._create_false_state()]

        return derived_lists


    def _postprocess(self, env_idx: int, states: List[Tensor], stage: str = "final") -> List[Tensor]:
        """
        Unified post-processing for derived states.

        stage='intermediate' : used inside skip-unary iterations
            - filter None
            - memory pruning (if enabled)
            - truncate by atom budget
            - (no truncation by number of states, no END injection)

        stage='final'        : used before exposing actions to the agent
            - all of the above
            - truncate by number of states (reserving an END slot if needed)
            - maybe add END action
        """
        # ---- ensure non-empty list ----
        if not states:
            return [self._create_false_state()]

        # ---- filter None (vectorized check) ----
        states = [s for s in states if s is not None and s.numel() > 0]
        if not states:
            return [self._create_false_state()]

        # ---- memory pruning (vectorized signature computation) ----
        if self.memory_pruning:
            curr_sig = self._state_signature(self.current_queries[env_idx])
            self.memories[env_idx].add(curr_sig)
            mem = self.memories[env_idx]
            
            # Vectorize signature computation where possible
            # Compute signatures in batch for states with same number of atoms
            kept_states = []
            for s in states:
                if self._state_signature(s) not in mem:
                    kept_states.append(s)
            states = kept_states
            
            if not states:
                return [self._create_false_state()]

        # ---- truncate by atom budget (vectorized) ----
        # Create a boolean mask for valid states
        states = [s for s in states if s.shape[0] <= self.padding_atoms]
        if not states:
            return [self._create_false_state()]

        # If this is only an intermediate step (skip-unary), stop here.
        if stage != "final":
            return states

        # ---- final stage: truncate by number of states (reserve END if needed) ----
        reserve_end = False
        if self.end_proof_action and states:
            # Vectorized check for non-terminal states
            # Stack first predicates and check in batch
            first_preds = torch.stack([s[0, 0] for s in states if s.numel() > 0])
            terminal_mask = (first_preds == self.true_pred_idx) | (first_preds == self.false_pred_idx)
            reserve_end = not terminal_mask.all().item()

        limit = self.padding_states - (1 if reserve_end else 0)
        if len(states) > limit:
            # Prefer simpler states (fewer atoms)
            states.sort(key=lambda t: t.shape[0])
            states = states[:limit]

        # ---- optionally add END action ----
        if self.end_proof_action and reserve_end and len(states) < self.padding_states:
            states = list(states)
            states.append(self._create_end_state())

        return states

        return states

    # ------------------- Reward/Done -------------------

    def _get_done_reward(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute rewards, terminated flags, and truncated flags for all environments.
        Vectorized implementation where possible.
        
        Returns:
            rewards: [B, 1] reward tensor
            terminated: [B, 1] bool tensor for natural episode termination (success/failure)
            truncated: [B, 1] bool tensor for artificial termination (time limit)
        """
        # Initialize outputs
        rewards = torch.zeros(self.batch_size_int, 1, dtype=torch.float32, device=self._device)
        terminated = torch.zeros(self.batch_size_int, 1, dtype=torch.bool, device=self._device)
        truncated = torch.zeros(self.batch_size_int, 1, dtype=torch.bool, device=self._device)
        
        # Get all states [B, padding_atoms, arity+1]
        states = self.current_queries
        
        # Vectorized: Check for non-padding predicates [B, padding_atoms]
        non_padding_mask = states[:, :, 0] != self.padding_idx  # [B, A]
        
        # Vectorized terminal checks per batch
        # Extract predicates for each environment [B, padding_atoms]
        predicates = states[:, :, 0]  # [B, padding_atoms]
        
        # For vectorization, we need to handle variable-length states
        # Create masks for all_true and any_false per batch element
        all_true_batch = torch.zeros(self.batch_size_int, dtype=torch.bool, device=self._device)
        any_false_batch = torch.zeros(self.batch_size_int, dtype=torch.bool, device=self._device)
        is_end_batch = torch.zeros(self.batch_size_int, dtype=torch.bool, device=self._device)
        
        if self.true_pred_idx is not None:
            # all_true: all non-padding predicates are true_pred_idx
            true_mask = (predicates == self.true_pred_idx) | ~non_padding_mask
            all_true_batch = true_mask.all(dim=1) & non_padding_mask.any(dim=1)
        
        if self.false_pred_idx is not None:
            # any_false: at least one predicate is false_pred_idx
            any_false_batch = (predicates == self.false_pred_idx).any(dim=1)
        
        if self.end_proof_action:
            # Check if single predicate is end_pred_idx
            is_single_pred = non_padding_mask.sum(dim=1) == 1
            first_pred = predicates[torch.arange(self.batch_size_int, device=self._device), 
                                   non_padding_mask.long().argmax(dim=1)]
            is_end_batch = is_single_pred & (first_pred == self.end_pred_idx)
            # End counts as success
            all_true_batch = all_true_batch | is_end_batch
            any_false_batch = any_false_batch & ~is_end_batch
        
        # Vectorized depth check
        depth_exceeded_batch = self.current_depths >= self.max_depth
        
        # Natural termination vs truncation
        natural_termination = all_true_batch | any_false_batch
        terminated[:, 0] = natural_termination
        truncated[:, 0] = depth_exceeded_batch & ~natural_termination
        
        # done is combination
        done_batch = natural_termination | depth_exceeded_batch
        
        # Vectorized reward computation based on reward_type
        successful_batch = all_true_batch
        labels_batch = self.current_labels  # [B]
        
        if self.reward_type == 0:
            # Simple: +1 only for successful proof with positive label
            reward_mask = done_batch & successful_batch & (labels_batch == 1)
            rewards[reward_mask, 0] = 1.0
            
        elif self.reward_type == 1:
            # +1 for true positive, -1 for false positive
            tp_mask = done_batch & successful_batch & (labels_batch == 1)
            fp_mask = done_batch & successful_batch & (labels_batch == 0)
            rewards[tp_mask, 0] = 1.0
            rewards[fp_mask, 0] = -1.0
            
        elif self.reward_type == 2:
            # +1 for correct classification (TP or TN)
            tp_mask = done_batch & successful_batch & (labels_batch == 1)
            tn_mask = done_batch & ~successful_batch & (labels_batch == 0)
            rewards[tp_mask, 0] = 1.0
            rewards[tn_mask, 0] = 1.0
            
        elif self.reward_type == 3:
            # Asymmetric: penalize false positives more
            pos_label = labels_batch == 1
            neg_label = labels_batch == 0
            
            # Positive label cases
            tp_mask = done_batch & successful_batch & pos_label
            fn_mask = done_batch & ~successful_batch & pos_label
            rewards[tp_mask, 0] = 1.0
            rewards[fn_mask, 0] = -0.5
            
            # Negative label cases
            fp_mask = done_batch & successful_batch & neg_label
            tn_mask = done_batch & ~successful_batch & neg_label
            rewards[fp_mask, 0] = -1.5
            rewards[tn_mask, 0] = 1.0
            
        elif self.reward_type == 4:
            pos_label = labels_batch == 1
            neg_label = labels_batch == 0
            
            # Positive label: +1 if done&successful, -1 if done&not successful, 0 otherwise
            tp_mask = done_batch & successful_batch & pos_label
            fp_done_mask = done_batch & ~successful_batch & pos_label
            rewards[tp_mask, 0] = 1.0
            rewards[fp_done_mask, 0] = -1.0
            
            # Negative label: -1 if done&successful, +1 if done&not successful, 0 otherwise
            fp_mask = done_batch & successful_batch & neg_label
            tn_mask = done_batch & ~successful_batch & neg_label
            rewards[fp_mask, 0] = -1.0
            rewards[tn_mask, 0] = 1.0
            
        else:
            raise ValueError(f"Invalid reward_type: {self.reward_type}. Choose 0-4.")
    
        return rewards, terminated, truncated

    # ------------------- Observation packing -------------------

    def _create_observation_dict(self) -> Dict:
        action_indices = torch.arange(self.padding_states, device=self._device)  # [S]
        action_indices = action_indices.unsqueeze(0).expand(self.batch_size_int, -1)  # [B, S]
        counts = self.derived_states_counts.unsqueeze(-1)  # [B, 1]
        action_mask = action_indices < counts

        return {
            'sub_index': self.current_queries.unsqueeze(1),     # [B,1,A,D]
            'derived_sub_indices': self.derived_states_batch,   # [B,S,A,D]
            'action_mask': action_mask,                         # [B,S]
        }

    def _create_observation(self) -> TensorDict:
        obs = self._create_observation_dict()
        obs['done'] = torch.zeros(self.batch_size_int, 1, dtype=torch.bool, device=self._device)
        obs['terminated'] = torch.zeros(self.batch_size_int, 1, dtype=torch.bool, device=self._device)
        obs['truncated'] = torch.zeros(self.batch_size_int, 1, dtype=torch.bool, device=self._device)
        return TensorDict(obs, batch_size=self.batch_size, device=self._device)

    # ------------------- Utilities -------------------

    def _pad_state(self, state: Tensor) -> Tensor:
        """Pad a state tensor to padding_atoms."""
        # Ensure state is on the correct device
        if state.device != self._device:
            state = state.to(self._device)
        
        if state.shape[0] >= self.padding_atoms:
            return state[:self.padding_atoms]
        
        padding_needed = self.padding_atoms - state.shape[0]
        padding = torch.full(
            (padding_needed, self.max_arity + 1),
            self.padding_idx,
            dtype=torch.long,
            device=self._device
        )
        return torch.cat([state, padding], dim=0)
    
    def _create_false_state(self) -> Tensor:
        """Create a False state for terminal condition."""        
        false_state = torch.full(
            (1, self.max_arity + 1),
            self.padding_idx,
            dtype=torch.long,
            device=self._device
        )
        false_state[0, 0] = self.false_pred_idx
        return self._pad_state(false_state)
    
    def _create_end_state(self) -> Tensor:
        """Create an End state for early stopping."""
        end_state = torch.full(
            (1, self.max_arity + 1),
            self.padding_idx,
            dtype=torch.long,
            device=self._device
        )
        end_state[0, 0] = self.end_pred_idx
        return self._pad_state(end_state)

    def _update_original_queries_for_indices(self, indices: List[int]):
        for env_idx in indices:
            state = self.current_queries[env_idx]  # [A,D]
            mask = state[:, 0] != self.padding_idx
            if mask.any():
                first_idx = mask.long().argmax().item()
                self.original_queries[env_idx] = state[first_idx]
            else:
                self.original_queries[env_idx] = torch.full(
                    (self.max_arity + 1,), self.padding_idx, dtype=torch.long, device=self._device
                )

    def _state_signature(self, state: Tensor) -> tuple:
        """Compute a signature for a state for deduplication.
        Optimized version with cached state tuples for faster comparison.
        """
        # Remove padding
        valid_mask = state[:, 0] != self.padding_idx
        if not valid_mask.any():
            return (0, 0, 0)
        
        valid_state = state[valid_mask]
        n = valid_state.shape[0]
        first_pred = int(valid_state[0, 0].item())

        # Convert to tuple for hashing - this is the fastest approach for small tensors
        # Python's built-in hash on tuples is highly optimized
        state_tuple = tuple(valid_state.flatten().tolist())
        h = hash(state_tuple) & 0x7FFFFFFFFFFFFFFF  # Ensure positive 63-bit int
        
        return (int(n), int(first_pred), int(h))
    
    def _set_seed(self, seed: int):
        """Set random seed."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
