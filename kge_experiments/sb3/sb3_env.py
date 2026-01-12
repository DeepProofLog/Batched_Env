from typing import Any, Dict, List, Optional, Set, Tuple

import math
import random
import warnings

import gymnasium as gym
import numpy as np
import torch
from tensordict import NonTensorData, TensorDict

from pykeen.constants import TARGET_TO_INDEX
try:
    # Try relative import first (when sb3/ is in sys.path)
    from sb3_dataset import DataHandler
    from sb3_index_manager import IndexManager
    from sb3_unification import get_next_unification_python
    from sb3_utils import Term, print_state_transition
except ImportError:
    # Fallback to package import (when imported as sb3.sb3_env)
    from sb3.sb3_dataset import DataHandler
    from sb3.sb3_index_manager import IndexManager
    from sb3.sb3_unification import get_next_unification_python
    from sb3.sb3_utils import Term, print_state_transition

State = List[Term]

def _state_to_hashable(state: State) -> frozenset:
    """
    Hashable, order-independent key for a state using Term objects directly.
    Term is an immutable, hashable dataclass, so this is safe and cheap.
    """
    return frozenset(state) if state else frozenset()

class LogicEnv_gym(gym.Env):
    batch_locked = False
    
    def __init__(self,
                index_manager: Optional[IndexManager] = None,
                data_handler: Optional[DataHandler] = None,
                queries: Optional[List[Term]] = None,
                labels: Optional[List[int]] = None,
                query_depths: Optional[List[Optional[int]]] = None,
                facts: Optional[Set[Term]] = None,
                mode: str = 'train',
                corruption_mode: Optional[str] = None,
                corruption_scheme: Optional[List[str]] = None,
                train_neg_ratio: int = 1,
                seed: Optional[int] = None,
                max_depth: int = 10,
                memory_pruning: bool = True,
                endt_action: bool = False,
                endf_action: bool = False,
                skip_unary_actions: bool = False,
                canonical_action_order: bool = False,
                padding_atoms: int = 10,
                padding_states: int = 20,
                verbose: int = 0,
                prover_verbose: int = 1,
                device: torch.device = torch.device("cpu"),
                engine: str = 'python',
                engine_strategy: str = 'cmp', # 'cmp' (complete) or 'rtf' (rules_then_facts)
                kge_action: bool = False,
                reward_type: int = 2,
                shaping_beta: float = 0.0,
                shaping_gamma: Optional[float] = None,
                kge_inference_engine: Optional[Any] = None,
                sample_deterministic: bool = False,
                ):

        '''Initialize the environment'''
        super().__init__()

        self.engine = engine
        self.engine_strategy = 'rules_then_facts' if engine_strategy=='rtf' else 'complete'
        self.reward_type = reward_type
        self.verbose = verbose
        self.prover_verbose = prover_verbose
        self.device = device
        self.pt_idx_dtype = torch.int32
        self.np_idx_dtype = np.int32
        self.reward_dtype = torch.float32
        self.max_arity = data_handler.max_arity # Maximum arity of the predicates
        self.padding_atoms = padding_atoms  # Maximum number of atoms in a state
        self.padding_states = padding_states # Maximum number of possible next states
        self.max_depth = max_depth # Maximum depth of the proof tree
        self.index_manager = index_manager
        self.predicates_arity = data_handler.predicates_arity

        self._set_seed(seed)
        self._make_spec()

        self.dataset_name = data_handler.dataset_name
        self.facts = facts

        self.corruption_mode = corruption_mode
        self.kge_action = kge_action

        if self.corruption_mode:
            self.counter = 0  # Determine whether to sample from positive or negative queries in KGE settings
            
        if self.corruption_mode:
            self.sampler = data_handler.sampler
            self.corruption_scheme = corruption_scheme
        
        self.janus_file = data_handler.janus_path
        self.janus_facts = data_handler.janus_facts

        self.memory = set() # Store grounded predicates, avoid loop
        self.memory_pruning = memory_pruning # two ways to avoid loop: limit action space, stop when a state has been visited
        self.endt_action = endt_action
        self.endf_action = endf_action
        self.skip_unary_actions = skip_unary_actions # Skip unary actions in the action space
        self.canonical_action_order = bool(canonical_action_order)
        self.predicate_false_idx = index_manager.predicate_str2idx['False'] 

        self.special_predicates = set(getattr(self.index_manager, "special_preds", []))
        end_predicates = {p for p in self.special_predicates if p.lower().startswith('end')}
        # Retain legacy 'End' sentinel if present in data
        end_predicates.add('End')
        self.end_predicates = end_predicates
        self.terminal_predicates = {'True', 'False'} | self.end_predicates

        self.kge_inference_engine = kge_inference_engine
        self.shaping_beta = float(shaping_beta)
        self.shaping_gamma = float(shaping_gamma) if shaping_gamma is not None else 1.0
        self.shaping_eps = 1e-9
        self._potential_cache: Dict[str, float] = {}
        self._last_potential = torch.zeros((), dtype=self.reward_dtype, device=self.device)

        self.current_query = None
        self.current_label = None

        self.mode = mode
        self.queries = queries
        self.labels = labels
        if query_depths is not None:
            self.query_depths = list(query_depths)
        elif queries is not None:
            self.query_depths = [None] * len(queries)
        else:
            self.query_depths = []
        self.n_episodes = len(queries) if queries is not None else 0
        self.eval_idx = 0
        
        # Deterministic round-robin sampling (for parity with tensor env)
        self.sample_deterministic = sample_deterministic
        self._train_ptr = 0  # Round-robin pointer for deterministic sampling
        self.consult_janus_eval = False
        self.next_var_index = self.index_manager.variable_start_index

        self.current_query_depth_value = None

        self.train_neg_ratio = train_neg_ratio
        if self.train_neg_ratio > 0:
            self.rejection_weight = 1.0 / self.train_neg_ratio
        else:
            self.rejection_weight = 1.0

        self._one  = torch.tensor(1.0, device=self.device, dtype=self.reward_dtype)
        self._zero = torch.zeros((), device=self.device, dtype=self.reward_dtype)
        self._n15  = torch.tensor(-1.5, device=self.device, dtype=self.reward_dtype)
        self._n05  = torch.tensor(-0.5, device=self.device, dtype=self.reward_dtype)

        self._n1 = torch.tensor(-1.0, device=self.device, dtype=self.reward_dtype)

    def set_shaping_beta(self, beta: float) -> None:
        """Update the shaping weight used for potential-based reward shaping."""
        self.shaping_beta = float(beta)
        self._potential_cache.clear()

    def _set_seed(self, seed:int):
        '''Set the seed for the environment. If no seed is provided, generate a random one.
        
        Note: We intentionally do NOT call torch.manual_seed() here because that would
        modify the global RNG state, which can cause issues with reproducibility in
        eval_corruptions when multiple batches are evaluated. Instead, we use Python's
        random.Random for local sampling needs.
        '''
        if seed is None:
            # Use Python's random to avoid consuming torch's global RNG
            seed = random.randint(0, 2**31 - 1)
        self.seed = int(seed)
        self.seed_gen = random.Random(self.seed)


    def _make_spec(self):
        '''Create the observation and action specs'''
        obs_spaces = {
            'sub_index': gym.spaces.Box(
                low=float('-inf'),
                high=float('inf'),
                shape = torch.Size([1])+ torch.Size([self.padding_atoms])+ torch.Size([self.max_arity+1]),
                dtype=self.np_idx_dtype,
            ),

            'derived_sub_indices': gym.spaces.Box(
                low=float('-inf'),
                high=float('inf'),
                shape=torch.Size([self.padding_states])+torch.Size([self.padding_atoms])+torch.Size([self.max_arity+1]),
                dtype=self.np_idx_dtype,
            ),  
            'action_mask': gym.spaces.MultiBinary(self.padding_states),
        }
        self.observation_space = gym.spaces.Dict(obs_spaces)
        self.action_space = gym.spaces.Discrete(self.padding_states)

    def _pad_sub_indices(self, sub_indices: List[torch.Tensor]) -> torch.Tensor:
        """Stack sub-index tensors and pad them to the action-space size."""
        if not sub_indices:
            return torch.zeros(
                self.padding_states,
                self.padding_atoms,
                self.max_arity + 1,
                device=self.device,
                dtype=self.pt_idx_dtype,
            )

        stacked = torch.stack(
            [tensor.to(device=self.device, dtype=self.pt_idx_dtype) for tensor in sub_indices]
        )
        num_states = stacked.shape[0]
        if num_states < self.padding_states:
            padding = torch.zeros(
                self.padding_states - num_states,
                self.padding_atoms,
                self.max_arity + 1,
                device=self.device,
                dtype=self.pt_idx_dtype,
            )
            stacked = torch.cat([stacked, padding], dim=0)
        return stacked

    def _pad_single_state(self, state: State) -> torch.Tensor:
        """Helper to stack and pad a single state's sub-index """
        state_tensor = self.index_manager.get_atom_sub_index(state)
        return self._pad_sub_indices([state_tensor])


    def _normalize_state_obj(self, state: Any) -> List[Term]:
        if isinstance(state, NonTensorData):
            state = getattr(state, "data", getattr(state, "_data", state))
        if isinstance(state, Term):
            return [state]
        return state if isinstance(state, list) else list(state)

    def _compute_state_potential(self, state: Any) -> float:
        if self.kge_inference_engine is None or self.shaping_beta == 0.0:
            return 0.0
        normalized = self._normalize_state_obj(state)
        if not normalized:
            return 0.0

        for atom in normalized:
            predicate = getattr(atom, "predicate", None)
            if predicate is None:
                continue
            if predicate in self.terminal_predicates or predicate.endswith('_kge'):
                continue
            args = getattr(atom, "args", ())
            atom_str = f"{predicate}({','.join(map(str, args))})"
            cached = self._potential_cache.get(atom_str)
            if cached is not None:
                return cached

            predictor_batch = getattr(self.kge_inference_engine, "predict_batch", None)
            if predictor_batch is not None:
                try:
                    preds = predictor_batch([atom_str])
                    score = preds[0] if preds else None
                except Exception:
                    score = None

            if score is None:
                return 0.0

            score = max(float(score), self.shaping_eps)
            potential = self.shaping_beta * math.log(score)
            self._potential_cache[atom_str] = potential
            return potential

        return 0.0

    def _select_query_for_reset(self) -> Tuple[Term, int, Optional[int]]:
        """Pick the next query/label/depth triple based on the current mode."""
        if self.mode == "eval":
            if self.eval_idx < self.n_episodes:
                idx = self.eval_idx
                state = self.queries[idx]
                label = self.labels[idx]
                depth = self.query_depths[idx] if idx < len(self.query_depths) else None
            else:
                state = Term(predicate="False", args=())
                label = 0
                depth = None
            self.eval_idx += 1
            return state, label, depth

        if self.mode == "eval_with_restart":
            if self.eval_idx == self.n_episodes:
                self.eval_idx = 0
            assert (
                self.eval_idx < self.n_episodes
            ), f"Eval index: {self.eval_idx}, n_episodes: {self.n_episodes}. Adjust the number of episodes to evaluate in the callback."
            idx = self.eval_idx
            state = self.queries[idx]
            label = self.labels[idx]
            depth = self.query_depths[idx] if idx < len(self.query_depths) else None
            self.eval_idx += 1
            return state, label, depth

        if self.mode == "train":
            return self._sample_train_query()

        raise ValueError(f"Invalid mode: {self.mode}. Choose from 'train', 'eval', or 'eval_with_restart'.")

    def _sample_train_query(self) -> Tuple[Term, int, Optional[int]]:
        """Sample a training query, optionally performing corruption sampling."""
        if self.sample_deterministic:
            # Deterministic round-robin sampling to match tensor env behavior
            idx = self._train_ptr % len(self.queries)
            self._train_ptr += 1
            state = self.queries[idx]
            depth = self.query_depths[idx] if idx < len(self.query_depths) else None
        else:
            state, _, depth = self.get_random_queries(
                self.queries,
                n=1,
                labels=self.labels,
                depths=self.query_depths,
                return_depth=True,
            )
        label = 1

        if not self.corruption_mode:
            return state, label, depth

        if self.counter % (int(self.train_neg_ratio) + 1) != 0:
            num_to_generate = (
                len(self.corruption_scheme) if len(self.corruption_scheme) > 1 else 1
            )
            
            # Retry sampling if negative generation fails
            max_retries = 5
            for attempt in range(max_retries):
                # Force sampler usage of our specific corruption scheme
                original_scheme = self.sampler.corruption_scheme
                original_indices = self.sampler._corruption_indices
                
                try:
                    self.sampler.corruption_scheme = self.corruption_scheme
                    self.sampler._corruption_indices = [TARGET_TO_INDEX[c] for c in self.corruption_scheme]
                    
                    negative_samples = self.sampler.get_negatives_from_states(
                        state, self.device, num_negs=num_to_generate
                    )
                finally:
                    # Restore original scheme
                    self.sampler.corruption_scheme = original_scheme
                    self.sampler._corruption_indices = original_indices
                selected = negative_samples
                if not isinstance(selected, list):
                    selected = [selected]
                if len(self.corruption_scheme) > 1:
                    if not hasattr(self, "negation_toggle"):
                        self.negation_toggle = 0
                    if self.negation_toggle < len(negative_samples):
                        selected = [negative_samples[self.negation_toggle]]
                    else:
                        selected = [] # Signal failure to pick
                    self.negation_toggle = 1 - self.negation_toggle

                # Check if we successfully generated a negative sample
                if len(selected) > 0 and (len(selected) != 1 or selected[0]):
                    break
                    
                # If this is not the last attempt, sample a new positive query to try again
                if attempt < max_retries - 1:
                    state, _, depth = self.get_random_queries(
                        self.queries,
                        n=1,
                        labels=self.labels,
                        depths=self.query_depths,
                        return_depth=True,
                    )
            
            # After retries, check if we have a valid negative
            if len(selected) == 0 or (len(selected) == 1 and not selected[0]):
                # All retries failed, fall back to positive sample
                pass  # Keep original state and label=1
                warnings.warn("Negative sampling produced fewer candidates than requested: got 0, expected 1", RuntimeWarning)
            else:
                assert len(selected) == 1, f"Length of negatives should be 1, but is {len(selected)}"
                state = selected[0]
                label = 0
                depth = None
        self.counter += 1
        return state, label, depth


    def reset(self, seed: Optional[int] = None, options=None):
        """Reset the environment and sample the next query according to the mode."""
        if self.verbose or self.prover_verbose:
            print("\n\nReset-----------------------------")
        if seed is not None:
            self._set_seed(seed)

        state, label, depth = self._select_query_for_reset()

        # if self.engine == 'prolog' and (self.mode == 'train' or self.consult_janus_eval == True) and label == 1:
        #     if state in self.facts:
        #         janus.query_once(f"retract({state.prolog_str()}).")
        #         # janus.query_once(f"retract({state}).")

        self.current_query = state
        self.current_label = label
        self.current_query_depth_value = depth if label == 1 else None
        return self._reset([state], label)


    def _reset(self, query: State, label: int):
        '''Reset the environment to the initial state'''    
        print('Initial query:', query, label) if self.verbose else None
        self.current_depth = torch.tensor(0, device=self.device)
        self.next_var_index = self.index_manager.variable_start_index

        self.memory = set()
        filtered_query = [q for q in query if q.predicate not in self.terminal_predicates]
        self.memory.add(_state_to_hashable(filtered_query))

        sub_index = self.index_manager.get_atom_sub_index(query).to(
            device=self.device, dtype=self.pt_idx_dtype
        )
        derived_states, derived_sub_indices, truncated_flag = self.get_next_states(query)
        valid = len(derived_states)
        action_mask = torch.zeros(self.padding_states, dtype=torch.uint8)
        action_mask[:valid] = 1
        if truncated_flag:  # end in false
            derived_states, derived_sub_indices = self.end_in_false()
            valid = len(derived_states)
            action_mask.zero_()
            action_mask[:valid] = 1
        if self.verbose:
            mask_list = action_mask.tolist()
            derived_preview = derived_sub_indices[:valid].cpu().numpy() if hasattr(derived_sub_indices, "cpu") else derived_sub_indices[:valid]
            print(f"[SB3Env reset] action_mask valid={int(action_mask.sum().item())} mask={mask_list}")
            print(f"[SB3Env reset] derived_sub_indices (first {valid}): {derived_preview}")

        self.tensordict = TensorDict(
            {
                "sub_index": sub_index.unsqueeze(0), # to match the shape of derived_sub_indices
                "state": NonTensorData(data=query),
                "label": torch.tensor(label, device=self.device),
                "done": torch.tensor(0, dtype=torch.bool, device=self.device),
                "reward": torch.zeros((), dtype=self.reward_dtype, device=self.device),
                "derived_states": NonTensorData(data=derived_states),
                "derived_sub_indices": derived_sub_indices,
            },
        )
        initial_potential_value = self._compute_state_potential(self.tensordict["state"])
        self._last_potential = torch.tensor(initial_potential_value, dtype=self.reward_dtype, device=self.device)
        obs = {'sub_index': self.tensordict['sub_index'].cpu().numpy(),
               'derived_sub_indices': self.tensordict['derived_sub_indices'].cpu().numpy(),
               'action_mask': action_mask.cpu().numpy(),}
        if self.verbose:
            print_state_transition(self.tensordict['state'], self.tensordict['derived_states'],self.tensordict['reward'], self.tensordict['done'],label=label)
        return obs, {}

    def step(self, action):
        '''
        Given the current state, possible next states, an action, and return the next state.
        (It should be: given the current state, and an action, return the next state, but we need to modify it for our case)
        '''
        derived_states = self.tensordict["derived_states"]
        derived_sub_indices = self.tensordict["derived_sub_indices"]

        if action >= self.padding_states or action > len(derived_states):
            raise ValueError(
                f"Invalid action ({action}). Derived states: {derived_states}.")
        
        next_state = derived_states[action]
        next_sub_index = derived_sub_indices[action]

        done_next, reward_next, successful = self.get_done_reward(next_state, self.tensordict['label'].item())
        if self.kge_inference_engine is not None and self.shaping_beta != 0.0:
            phi_s = self._last_potential
            phi_sp_value = 0.0 if bool(done_next) else self._compute_state_potential(next_state)
            phi_sp = torch.tensor(phi_sp_value, dtype=self.reward_dtype, device=self.device)
            reward_next = reward_next + self.shaping_gamma * phi_sp - phi_s
            self._last_potential = phi_sp if not bool(done_next) else torch.zeros_like(phi_sp)
        else:
            self._last_potential = torch.zeros((), dtype=self.reward_dtype, device=self.device)

        derived_states_next, derived_sub_indices_next, truncate_flag = self.get_next_states(next_state)
        valid = len(derived_states_next)
        action_mask = torch.zeros(self.padding_states, dtype=torch.uint8)
        action_mask[:valid] = 1
        self.current_depth += 1
        exceeded_max_depth = (self.current_depth >= self.max_depth)
        if exceeded_max_depth: print('\nMax depth reached', self.current_depth.item()) if self.verbose else None

        if self.verbose:
            mask_list = action_mask.tolist()
            derived_preview = derived_sub_indices_next[:valid].cpu().numpy() if hasattr(derived_sub_indices_next, "cpu") else derived_sub_indices_next[:valid]
            print(f"[SB3Env step] action_mask valid={int(action_mask.sum().item())} mask={mask_list}")
            print(f"[SB3Env step] derived_sub_indices (first {valid}): {derived_preview}")

        done_next = done_next | exceeded_max_depth | truncate_flag
        truncated = bool(exceeded_max_depth) or bool(truncate_flag)
        
        label_value = int(self.current_label) 
        info = {
            "label": label_value,
            "query_type": "positive" if label_value == 1 else "negative",
        }
        info["query_depth"] = self.current_query_depth_value
        info["max_depth_reached"] = bool(exceeded_max_depth)
        if done_next:
            info["is_success"] = successful
            # if self.engine == 'prolog' and self.current_label == 1 and self.current_query in self.facts:
            #     janus.query_once(f"asserta({self.current_query.prolog_str()}).")
            #     # janus.query_once(f"asserta({self.current_query}).")
            self.current_query, self.current_label = None, None

        self.tensordict.update(TensorDict({
                "sub_index": next_sub_index.unsqueeze(0), # to match the shape of derived_sub_indices
                "state": NonTensorData(data=next_state),
                "label": self.tensordict['label'],
                "done": done_next,
                "reward": reward_next,
                "derived_states": NonTensorData(data=derived_states_next),
                "derived_sub_indices": derived_sub_indices_next,
        }))

        obs = {'sub_index': self.tensordict['sub_index'].cpu().numpy(),
               'derived_sub_indices': self.tensordict['derived_sub_indices'].cpu().numpy(),
               'action_mask': action_mask.cpu().numpy(),}
        reward = self.tensordict['reward'].cpu().numpy()
        done = self.tensordict['done'].cpu().numpy()
        if self.verbose:
            print_state_transition(self.tensordict['state'], self.tensordict['derived_states'],self.tensordict['reward'], self.tensordict['done'], action=action,truncated=truncated)
        return obs, reward, done, truncated, info


    
    def get_next_states(self, state: State) -> Tuple[List[State], torch.Tensor, bool]:
        """Compute candidate next states from a given state.

        Returns
        -------
        derived_states : list[list[Term]]
            Next symbolic states (unpadded list).
        derived_sub_indices : torch.Tensor
            Int32 tensor of shape ``(padding_states, padding_atoms, max_arity+1)``.
        truncated_flag : bool
            True if truncation enforced (e.g., explicit end).
        """
        state = list(state)
        truncated_flag = False

        terminal_predicates = self.terminal_predicates
        if len(state) > 1 and all(atom.predicate in terminal_predicates for atom in state):
            raise ValueError(f"Invalid state: A state should not contain multiple only terminal atoms. Received: {state}")

        # If the state is a single terminal atom, return it as the only option.
        if state and len(state) == 1 and state[0].predicate in terminal_predicates:
            # Return the terminal state as the only option; termination is handled by the caller.
            return [state], self._pad_single_state(state), False

        # END ACTION MODULE
        if self.endt_action or self.endf_action:
            state = [atom for atom in state if atom.predicate not in self.end_predicates]

        if self.kge_action and state:
            filtered_state = [term for term in state if not term.predicate.endswith('_kge')]
            state = [Term(predicate='True', args=())] if not filtered_state else filtered_state
            if len(state) == 1 and state[0].predicate in terminal_predicates:
                return [state], self._pad_single_state(state), False
            # CAREFUL WHEN THE FIRST ATOM HAS A VAR AND THERE ARE MORE STATES, NEED TO UNIFY

        # if self.engine == 'prolog':
        #     derived_states, self.next_var_index = get_next_unification_prolog(state,
        #                                                  next_var_index=self.index_manager.next_var_index, 
        #                                                  verbose=self.prover_verbose)
        if self.engine == 'python':
            derived_states, self.next_var_index = get_next_unification_python(state,
                                                            facts_set=self.facts,
                                                            facts_indexed=self.index_manager.fact_index,
                                                            rules=self.index_manager.rules_by_pred,
                                                            excluded_fact = self.current_query if self.current_label == 1 else None,
                                                            verbose=self.prover_verbose,
                                                            next_var_index=self.next_var_index,
                                                            strategy= self.engine_strategy,
                                                            max_derived_states=self.padding_states,
                                                            canonical_order=self.canonical_action_order,
                                                            index_manager=self.index_manager
                                                            )
        if self.skip_unary_actions:
            current_state = state.copy() if isinstance(state, list) else [state]
            counter = 0
            while (len(derived_states) == 1 and 
                derived_states[0] and
                derived_states[0][0].predicate not in terminal_predicates):
                print('\n*********') if self.verbose else None
                print(f"Skipping unary action: current state: {current_state} -> derived states: {derived_states}") if self.verbose else None
                print('\n') if self.verbose else None
                counter += 1
                current_state = derived_states[0].copy()    

                if self.kge_action and current_state:
                    filtered_state = [term for term in current_state if not term.predicate.endswith('_kge')]
                    current_state = [Term(predicate='True', args=())] if not filtered_state else filtered_state
                    # CAREFUL WHEN THE FIRST ATOM HAS A VAR AND THERE ARE MORE STATES, NEED TO UNIFY

                if self.engine == 'python':
                    derived_states, self.next_var_index = get_next_unification_python(
                        current_state,
                        facts_set=self.facts,
                        facts_indexed=self.index_manager.fact_index,
                        rules=self.index_manager.rules_by_pred,
                        excluded_fact = self.current_query if self.current_label == 1 else None,
                        verbose=self.prover_verbose,
                        next_var_index=self.next_var_index,
                        strategy= self.engine_strategy,
                        max_derived_states=self.padding_states,
                        canonical_order=self.canonical_action_order,
                        index_manager=self.index_manager
                    )
                # MEMORY
                if self.memory_pruning:
                    self.memory.add(_state_to_hashable([s for s in current_state if s.predicate not in self.terminal_predicates]))
                    visited_mask = [_state_to_hashable(state) in self.memory for state in derived_states]
                    if any(visited_mask):
                        print(f"Memory: {self.memory}") if self.verbose else None
                        print(f"Visited mask: {visited_mask}. Current state: {current_state} -> Derived states: {derived_states}") if self.verbose else None
                        derived_states = [state for state, is_visited in zip(derived_states, visited_mask) if not is_visited]
                    
                # TRUNCATE MAX ATOMS
                mask_exceeded_max_atoms = [len(state) > self.padding_atoms for state in derived_states]
                print(f" Exceeded max atoms: {[len(state) for state in derived_states]}") if self.verbose and any(mask_exceeded_max_atoms) else None
                derived_states = [state for state, is_exceeded in zip(derived_states, mask_exceeded_max_atoms) if not is_exceeded]
                
                print('\n') if self.verbose else None
                print(f"Updated Next States: Current state: {current_state} -> Derived states: {derived_states}") if self.verbose else None
                print('*********\n') if self.verbose else None

                if counter > 20:
                    print('Max iterations reached') if self.verbose else None
                    derived_states = [[Term(predicate='False', args=())]]
                    truncated_flag = True
                    break

        final_states: List[State] = []
        final_sub_indices: List[torch.Tensor] = []
        # MEMORY MODULE
        if self.memory_pruning:
            self.memory.add(_state_to_hashable([s for s in state if s.predicate not in self.terminal_predicates]))

        for d_state in derived_states:
            # 1. Memory Pruning Check
            if self.memory_pruning and _state_to_hashable(d_state) in self.memory:
                continue

            # 2. Max Atoms Check
            if len(d_state) > self.padding_atoms:
                if self.verbose:
                    print(f"Exceeded max atoms in next states: {len(d_state)}")
                continue

            # If checks pass, convert to tensor and add to final lists
            final_states.append(d_state)
            final_sub_indices.append(self.index_manager.get_atom_sub_index(d_state))

        derived_states = final_states # Use the filtered list from now on

        # If no derived states remain after memory pruning, end in False (proof is stuck)
        if not derived_states:
            derived_states, derived_sub_indices = self.end_in_false()
            return derived_states, derived_sub_indices, truncated_flag

        # KGE ACTION MODULE
        # It uses the original `state` variable to ensure alignment.
        if self.kge_action and state and state[0].predicate not in terminal_predicates:
            kge_pred_name = f"{state[0].predicate}_kge"
            kge_action_term = Term(predicate=kge_pred_name, args=state[0].args)
            
            # Avoid adding a duplicate KGE action
            if not any(kge_action_term in derived_state for derived_state in derived_states):
                derived_states.append([kge_action_term])
                final_sub_indices.append(self.index_manager.get_atom_sub_index([kge_action_term]))

        # TRUNCATE MAX STATES
        max_num_states = self.padding_states
        if self.endt_action:
            max_num_states -= 1
        if self.endf_action:
            max_num_states -= 1

        if len(derived_states) > max_num_states:
            print(f"Exceeded max next states: {len(derived_states)}") if self.verbose else None
            
            # Preserve natural order when truncating (matches tensor engine behavior)
            # The natural order is: rules first, then facts, in the order they appear in data
            indices = list(range(max_num_states))
            
            derived_states = [derived_states[i] for i in indices]
            final_sub_indices = [final_sub_indices[i] for i in indices]

        # END ACTION MODULE
        # Only add Endf/Endt when there are non-terminal derived states available
        # (giving the agent the option to give up instead of continuing)
        if self.endt_action or self.endf_action:
            has_terminal_outcome = any(
                any(atom.predicate in ('True', 'False') for atom in state)
                for state in derived_states
            )

        if self.endf_action and not has_terminal_outcome:
            # Add Endf (End proof as False)
            endf_state = [Term(predicate='Endf', args=())]
            derived_states.append(endf_state)
            final_sub_indices.append(self.index_manager.get_atom_sub_index(endf_state))

        if self.endt_action and not has_terminal_outcome:
            # Add Endt (End proof as True)
            endt_state = [Term(predicate='Endt', args=())]
            derived_states.append(endt_state)
            final_sub_indices.append(self.index_manager.get_atom_sub_index(endt_state))

        derived_sub_indices = self._pad_sub_indices(final_sub_indices)

        return derived_states, derived_sub_indices, truncated_flag
    
    def get_done_reward(self,state: List[Term], label: int) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """Compute termination and reward signals for a proposed next state.

        Returns
        -------
        done : torch.Tensor
            Boolean scalar.
        reward : torch.Tensor
            Scalar reward.
        successful : bool
            Whether the transition solved the query.
        """
        assert label is not None, f"Label is None"

        any_atom_false = any(atom.predicate == 'False' for atom in state)
        all_atoms_true = all(atom.predicate == 'True' for atom in state)

        done = any_atom_false or all_atoms_true
        successful = all_atoms_true

        if self.endf_action and any(atom.predicate == 'Endf' for atom in state):
            assert len(state) == 1, f"State with Endf action should have only one atom, but has {len(state)}: {state}"
            done, successful = True, False
        if self.endt_action and any(atom.predicate == 'Endt' for atom in state):
            assert len(state) == 1, f"State with Endt action should have only one atom, but has {len(state)}: {state}"
            done, successful = True, True

        done = torch.tensor(done, device=self.device)

        if self.reward_type == 0:
            reward = self._one if (done and successful and label == 1) else self._zero
        elif self.reward_type == 1:
            if done and successful and label == 1:
                reward = self._one
            elif done and successful and label == 0:
                reward = self._n1
                # print(f"Done with success but label is 0: {state}")
            else:
                reward = self._zero
        elif self.reward_type == 2:
            if done and successful and label == 1:
                reward = self._one
            elif done and not successful and label == 0:
                reward = self._one
            else:
                reward = self._zero
        elif self.reward_type == 3:
            if label == 1:
                if done and successful:
                    reward = self._one
                elif done and not successful:
                    reward = self._n05
                else:
                    reward = self._zero
            if label == 0:
                if done and successful:
                    # The asymmetric −1.5 on false positives makes the policy aggressively lower the log‑probabilities of spurious negative proofs
                    reward = self._n15
                elif done and not successful:
                    reward = self._one
                else:
                    reward = self._zero
        elif self.reward_type == 4:
            if label == 1:
                if done and successful:
                    reward = self._one
                elif done and not successful:
                    reward = self._n1
                else:
                    reward = self._zero
            if label == 0:
                if done and successful:
                    reward = self._n1
                elif done and not successful:
                    reward = self._one * self.rejection_weight
                else:
                    reward = self._zero
        else: 
            raise ValueError(f"Invalid reward type: {self.reward_type}. Choose from 0-4.")

        return done, reward, successful
    
    def get_random_queries(self,
                           queries: List[Term], 
                           n: int = 1, 
                           labels: List[int] = None,
                           depths: Optional[List[Optional[int]]] = None,
                           return_depth: bool = False):
        """Sample queries (and labels) from a pool.

        Parameters
        ----------
        queries : list[Term]
            Source pool to sample from.
        n : int
            Number to sample.
        return_tensor : bool
            If True, also return tensor versions.
        """
        assert n <= len(queries), f"Number of queries ({n}) is greater than the number of queries ({len(queries)})"
        sampled_indices = self.seed_gen.sample(range(len(queries)), n)
        sampled_queries = [queries[i] for i in sampled_indices]
        sampled_labels = [labels[i] for i in sampled_indices] if labels else [None] * n
        if depths is not None:
            sampled_depths = [depths[i] if i < len(depths) else None for i in sampled_indices]
        else:
            sampled_depths = [None] * n

        if return_depth:
            if n == 1:
                return sampled_queries[0], sampled_labels[0], sampled_depths[0]
            return sampled_queries, sampled_labels, sampled_depths

        if n == 1:
            return sampled_queries[0], sampled_labels[0]
        return sampled_queries, sampled_labels


    def end_in_false(self) -> Tuple[List[State], torch.Tensor]:
        """Return a single False state as the only available transition."""
        false_term = Term(predicate="False", args=())
        derived_states_next: List[State] = [[false_term]]

        derived_sub_indices_next = torch.zeros(
            self.padding_states,
            self.padding_atoms,
            self.max_arity + 1,
            device=self.device,
            dtype=self.pt_idx_dtype,
        )
        false_row = torch.zeros(
            self.max_arity + 1,
            device=self.device,
            dtype=self.pt_idx_dtype,
        )
        false_row[0] = self.predicate_false_idx
        derived_sub_indices_next[0, 0] = false_row
        return derived_states_next, derived_sub_indices_next
