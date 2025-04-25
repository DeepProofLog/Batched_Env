from typing import List, Optional, Tuple, Dict, Union, Set
import random
from math import prod
from optimizaztion_subindices.utils_v1 import Term, print_state_transition
from index_manager_v2 import IndexManager
from python_unification_subidx import get_next_unification_ids, debug_print_state_ids, debug_print_states_list
import torch
from tensordict import TensorDict, TensorDictBase, NonTensorData
import gymnasium as gym
import numpy as np
from dataset import DataHandler
import janus_swi as janus

class LogicEnv_gym(gym.Env):
    batch_locked = False  # Allow dynamic batch sizes

    def __init__(self,
                 index_manager: IndexManager, # Now mandatory and holds all mappings/rules
                 data_handler: Optional[DataHandler] = None, # Still needed for facts, queries, arity etc.
                 queries: Optional[List[Term]] = None,
                 labels: Optional[List[int]] = None,
                 mode: str = 'train',
                 corruption_mode: Optional[str] = None,
                 corruption_scheme: Optional[List[str]] = None,
                 train_neg_pos_ratio: int = 1,
                 seed: Optional[int] = None,
                 dynamic_consult: bool = True, # Keep for Prolog compatibility if needed
                 max_depth: int = 10,
                 memory_pruning: bool = True,
                 end_proof_action: bool = False,
                 skip_unary_actions: bool = False,
                 truncate_atoms: bool = False, # Max number of atoms in a state
                 truncate_states: bool = False, # Max number of derived next states
                 padding_atoms: int = 10,    # Max atoms per state for padding
                 padding_states: int = 20,   # Max derived states for padding
                 max_atom_id_length: Optional[int] = None, # Predicate ID + Max Arity
                 verbose: int = 0,
                 prover_verbose: int = 0,
                 device: torch.device = torch.device("cpu"),
                 engine: str = 'python', # Should primarily be 'python' now
                 ):

        '''Initialize the environment'''
        super().__init__()

        assert engine == 'python', "This version is optimized for the python ID-based engine."
        self.engine = engine

        self.verbose = verbose
        self.prover_verbose = prover_verbose
        self.device = device

        self.index_manager = index_manager
        # Ensure padding ID is 0 for implicit masking to work
        assert self.index_manager.padding_id == 0, "Implicit masking requires padding ID to be 0."

        self.max_arity = data_handler.max_arity if data_handler else index_manager.max_arity # Get from one source
        self.predicates_arity = data_handler.predicates_arity if data_handler else {} # Need if using Janus

        self.max_atom_id_length = max_atom_id_length if max_atom_id_length is not None else self.max_arity + 1

        self.padding_atoms = padding_atoms
        self.padding_states = padding_states
        self.max_depth = max_depth

        # Pre-compute facts IDs
        self.facts_terms = data_handler.facts if data_handler else []
        self.facts_as_ids_set: Set[Tuple[int, ...]] = set()
        if self.facts_terms:
             facts_var_map = self.index_manager.variable_str2idx.copy()
             facts_next_temp_id = self.index_manager.temp_variable_start_idx
             for fact in self.facts_terms:
                 try:
                     fact_ids, _, facts_next_temp_id = self.index_manager.term_to_ids(
                         fact, facts_var_map, facts_next_temp_id, assign_new_vars=False
                     )
                     if any(self.index_manager.is_variable_id(arg) for arg in fact_ids[1:]):
                          print(f"Warning: Fact '{fact}' appears to contain variables. Treating as ground.")
                     self.facts_as_ids_set.add(fact_ids)
                 except ValueError as e:
                     print(f"Warning: Skipping fact '{fact}' during initial conversion: {e}")
        if self.prover_verbose >= 3: print(f"Pre-computed Facts as IDs: {self.facts_as_ids_set}")


        self._set_seed(seed)
        self._make_spec() # Define Observation/Action space (now without masks)

        self.dataset_name = data_handler.dataset_name if data_handler else "Unknown"

        self.corruption_mode = corruption_mode
        # ... (rest of corruption setup) ...
        if 'static' in self.corruption_mode or 'dynamic' in self.corruption_mode:
             self.counter = 0
        if self.corruption_mode == "dynamic":
             assert data_handler is not None, "DataHandler needed for dynamic corruption"
             self.sampler = data_handler.sampler
             self.triples_factory = data_handler.triples_factory
             self.corruption_scheme = corruption_scheme


        # Janus related attributes
        self.janus_file = data_handler.janus_path if data_handler else None
        self.janus_facts = data_handler.janus_facts if data_handler else []

        self.memory = set()
        self.memory_pruning = memory_pruning
        self.end_proof_action = end_proof_action
        self.skip_unary_actions = skip_unary_actions
        self.truncate_atoms = truncate_atoms
        self.truncate_states = truncate_states
        self.predicate_false_offset = self.index_manager.false_pred_id
        self.predicate_true_offset = self.index_manager.true_pred_id
        # self.predicate_end_offset = ... # If using End action

        self.dynamic_consult = dynamic_consult
        self.current_query_term: Optional[Term] = None
        self.current_query_ids: Optional[List[Tuple[int,...]]] = None
        self.current_label: Optional[int] = None
        self.last_query_term: Optional[Term] = None

        assert mode in ['train', 'eval', 'eval_corr'], f"Invalid mode: {mode}"
        self.mode = mode
        self.queries_terms = queries if queries else []
        self.labels = labels if labels else []
        self.n_episodes = len(self.queries_terms)
        self.eval_idx = 0
        self.mask_eval = []
        self.consult_janus_eval = False

        if self.mode == 'train':
            self.train_neg_pos_ratio = train_neg_pos_ratio
            if self.corruption_mode == "static":
                 assert data_handler is not None, "DataHandler needed for static corruption"
                 self.neg_queries_terms = data_handler.neg_train_queries
                 self.neg_labels = [0]*len(self.neg_queries_terms)
                 self.pos_queries_terms = self.queries_terms
                 self.pos_labels = self.labels


    def _set_seed(self, seed:int):
        '''Set the seed for the environment.'''
        self.seed = seed if seed is not None else torch.empty((), dtype=torch.int64).random_().item()
        rng = torch.manual_seed(self.seed)
        self.rng = rng
        self.seed_gen = random.Random(self.seed)


    def _make_spec(self):
        '''Create the observation and action specs (No explicit masks).'''
        obs_spaces = {
            # Padded ID representation of the current state
            'state_ids': gym.spaces.Box(
                low=0, # Assuming IDs are non-negative, 0 is padding
                high=np.iinfo(np.int64).max,
                shape=(self.padding_atoms, self.max_atom_id_length),
                dtype=np.int64,
            ),
            # Padded ID representation of derived next states
            'derived_state_ids': gym.spaces.Box(
                low=0,
                high=np.iinfo(np.int64).max,
                shape=(self.padding_states, self.padding_atoms, self.max_atom_id_length),
                dtype=np.int64,
            ),
        }
        self.observation_space = gym.spaces.Dict(obs_spaces)
        self.action_space = gym.spaces.Discrete(self.padding_states)
        self.num_valid_actions = 0


    def _pad_state_ids(self, state_ids: List[Tuple[int, ...]]) -> torch.Tensor:
        """Pads a single state (list of ID tuples) to fixed dimensions. Returns only padded IDs."""
        padded_ids = torch.full(
            (self.padding_atoms, self.max_atom_id_length),
            self.index_manager.padding_id, # Should be 0
            dtype=torch.int64,
            device=self.device
        )
        num_atoms = min(len(state_ids), self.padding_atoms)

        for i in range(num_atoms):
            atom_tuple = state_ids[i]
            atom_len = min(len(atom_tuple), self.max_atom_id_length)
            padded_ids[i, :atom_len] = torch.tensor(atom_tuple[:atom_len], dtype=torch.int64, device=self.device)

        return padded_ids


    def _pad_derived_states_ids(self, derived_states_ids: List[List[Tuple[int, ...]]]) -> torch.Tensor:
        """Pads a list of derived states. Returns only padded IDs."""
        padded_derived_ids = torch.full(
            (self.padding_states, self.padding_atoms, self.max_atom_id_length),
            self.index_manager.padding_id, # Should be 0
            dtype=torch.int64,
            device=self.device
        )

        num_derived = min(len(derived_states_ids), self.padding_states)
        for i in range(num_derived):
            state_ids = derived_states_ids[i]
            num_atoms = min(len(state_ids), self.padding_atoms)
            for j in range(num_atoms):
                atom_tuple = state_ids[j]
                atom_len = min(len(atom_tuple), self.max_atom_id_length)
                padded_derived_ids[i, j, :atom_len] = torch.tensor(atom_tuple[:atom_len], dtype=torch.int64, device=self.device)

        return padded_derived_ids # Only return IDs


    # --- Janus functions ---
    def new_consult_janus(self, query_term: Term):
        '''Consult janus with new facts (if using Prolog engine)'''
        if not janus or not self.janus_file: return
        query_str = str(query_term).replace(' ', '')
        facts = [line for line in self.janus_facts if query_str not in line]
        # ... (rest of the function remains the same)
        for predicate, arity in self.predicates_arity.items():
             try: janus.query_once(f"abolish({predicate}/{arity}).")
             except: pass
        try: janus.query_once("abolish_all_tables.")
        except: pass
        tmp_file = self.janus_file.replace('.pl', '_tmp.pl')
        with open(tmp_file, "w") as f:
            for line in facts: f.write(line)
        try: janus.consult(tmp_file)
        except Exception as e: print(f"Janus consult error: {e}")


    def dynamic_consult_janus(self, query_term: Term):
        '''Dynamically manage facts in Janus (if using Prolog engine)'''
        if not janus: return
        if self.last_query_term and not self.last_query_term == query_term:
             try: janus.query_once(f"asserta({str(self.last_query_term)}).")
             except: pass
        if query_term in self.facts_terms:
             self.last_query_term = query_term
             try: janus.query_once(f"retract({str(self.last_query_term)}).")
             except: pass
        else:
             self.last_query_term = None
    # --- End Janus functions ---

    def reset(self, seed: Optional[int]= None, options=None):
        print('\n\nReset-----------------------------') if self.verbose else None
        if seed is not None: self._set_seed(seed)

        query_term: Optional[Term] = None
        label: Optional[int] = None

        # --- Select Query Term based on mode ---
        # ... (Query selection logic remains the same) ...
        if self.mode == 'eval':
            if self.eval_idx == self.n_episodes: self.eval_idx = 0
            if not self.queries_terms: return self._reset([], 0) # Handle empty queries
            query_term = self.queries_terms[self.eval_idx]
            label = self.labels[self.eval_idx] if self.labels else 0
            self.eval_idx += 1
            self.consult_janus_eval = True

        elif self.mode == 'eval_parallel':
             if self.eval_idx < self.n_episodes:
                 if not self.queries_terms: return self._reset([], 0) # Handle empty queries
                 query_term = self.queries_terms[self.eval_idx]
                 label = self.labels[self.eval_idx] if self.labels else 0
                 self.mask_eval.append(True)
             else:
                 query_term = Term(predicate='False', args=[])
                 label = 0
                 self.mask_eval.append(False)
             self.eval_idx += 1

        else:  # Training mode
            if self.corruption_mode == "dynamic":
                if not self.queries_terms: return self._reset([], 0) # Handle empty queries
                query_term, _ = self.get_random_queries(self.queries_terms, n=1)
                label = 1
                if self.counter % (self.train_neg_pos_ratio + 1) != 0:
                    neg_terms = self.get_negatives(query_term)
                    if neg_terms:
                        if len(self.corruption_scheme) > 1:
                             if not hasattr(self, 'negation_toggle'): self.negation_toggle = 0
                             idx = self.negation_toggle % len(neg_terms)
                             query_term = neg_terms[idx]
                             self.negation_toggle = 1 - self.negation_toggle
                        else:
                             query_term = neg_terms[0]
                        label = 0
                    else: # Handle failed negative generation
                        print("Warning: Failed to generate negative sample, keeping positive.")
                self.counter += 1
            elif self.corruption_mode == "static":
                if self.counter % (self.train_neg_pos_ratio + 1) == 0:
                    if not self.pos_queries_terms: return self._reset([], 0)
                    query_term, _ = self.get_random_queries(self.pos_queries_terms, n=1)
                    label = 1
                else:
                    if not self.neg_queries_terms: return self._reset([], 0)
                    query_term, _ = self.get_random_queries(self.neg_queries_terms, n=1)
                    label = 0
                self.counter += 1
            else: # No corruption
                if not self.queries_terms: return self._reset([], 0)
                query_term, _ = self.get_random_queries(self.queries_terms, n=1)
                label = 1


        # --- Handle Janus Consultation ---
        if self.engine == 'prolog' and query_term and label == 1 and (self.mode == 'train' or self.consult_janus_eval):
             # ... (Janus logic remains the same) ...
             if query_term in self.facts_terms:
                 if not self.dynamic_consult:
                     self.new_consult_janus(query_term)
                 elif self.dynamic_consult:
                      try: janus.query_once(f"retract({str(query_term)}).")
                      except: pass
                 self.last_query_term = query_term
             else:
                 self.last_query_term = None


        self.current_query_term = query_term
        self.current_label = label

        return self._reset([query_term] if query_term else [], label if label is not None else 0)


    def reset_from_query(self, query_term: Term, label: int, consult_janus=False):
        ''' Reset the environment from a given query Term and label.'''
        print('\n\nReset from query-----------------------------') if self.verbose else None
        # Handle Janus
        if self.engine == 'prolog' and consult_janus and label == 1:
             # ... (Janus logic remains the same) ...
             if query_term in self.facts_terms:
                 if not self.dynamic_consult:
                     self.new_consult_janus(query_term)
                 elif self.dynamic_consult:
                     try: janus.query_once(f"retract({str(query_term)}).")
                     except: pass
                 self.last_query_term = query_term
             else:
                 self.last_query_term = None


        self.current_query_term = query_term
        self.current_label = label
        return self._reset([query_term], label)


    def _reset(self, query_terms: List[Term], label: int):
        '''Internal reset logic using ID representation'''
        print('Initial query:', query_terms, label) if self.verbose else None
        self.current_depth = torch.tensor(0, device=self.device)

        initial_temp_id = self.index_manager.temp_variable_start_idx
        state_ids, _, next_temp_id = self.index_manager.state_to_ids(
            query_terms, None, initial_temp_id
        )
        self.current_query_ids = state_ids

        self.memory = set()
        state_repr = self._get_state_repr_for_memory(state_ids)
        if state_repr: self.memory.add(state_repr)

        derived_states_ids, next_temp_id, truncated_flag = self.get_next_states(state_ids, next_temp_id)

        if self.truncate_states and truncated_flag:
             if self.verbose: print("State truncated at reset, ending in False.")
             derived_states_ids, next_temp_id = self._end_in_false_ids(next_temp_id)

        # Padding for Observation
        padded_state_ids = self._pad_state_ids(state_ids)
        padded_derived_ids = self._pad_derived_states_ids(derived_states_ids)
        self.num_valid_actions = len(derived_states_ids)

        # Create TensorDict
        self.tensordict = TensorDict(
            {
                "state_ids_pad": padded_state_ids, # Padded current state
                "state_ids": NonTensorData(data=state_ids), # Actual current state IDs
                "next_temp_id": torch.tensor(next_temp_id, device=self.device),
                "label": torch.tensor(label, device=self.device),
                "done": torch.tensor(0, dtype=torch.bool, device=self.device),
                "reward": torch.tensor(0, dtype=torch.float32, device=self.device),
                "derived_states_ids_pad": padded_derived_ids, # Padded derived states
                "derived_states_ids": NonTensorData(data=derived_states_ids), # Actual derived states IDs
            },
            batch_size=[]
        )

        # Create Observation for Gym
        obs = {
            'state_ids': padded_state_ids.cpu().numpy(),
            'derived_state_ids': padded_derived_ids.cpu().numpy(),
        }

        if self.verbose:
             current_terms = self.index_manager.ids_to_state(state_ids)
             derived_terms = [self.index_manager.ids_to_state(s_ids) for s_ids in derived_states_ids]
             print_state_transition(current_terms, derived_terms, self.tensordict['reward'], self.tensordict['done'], label=label)
        return obs, {"label": label}


    def step(self, action: int):
        '''Perform a step using the chosen action (index into derived states).'''

        derived_states_ids = self.tensordict["derived_states_ids"]
        current_next_temp_id = self.tensordict["next_temp_id"].item()
        # print('actions:', action)
        if action < 0 or action >= self.num_valid_actions:
             raise ValueError(f"Warning: Invalid action ({action}) selected. Num valid actions: {self.num_valid_actions}. Forcing False state.")
        
        next_state_ids = derived_states_ids[action]
        done_next, reward_next = self.get_done_reward(next_state_ids, self.tensordict['label'].item())
        derived_states_next_ids, next_next_temp_id, truncated_next_flag = self.get_next_states(
            next_state_ids, current_next_temp_id)

        self.current_depth += 1
        exceeded_max_depth = (self.current_depth >= self.max_depth)
        if exceeded_max_depth: print(f'\nMax depth reached ({self.current_depth.item()}/{self.max_depth})') if self.verbose else None

        final_done = done_next or exceeded_max_depth or truncated_next_flag
        final_truncated = exceeded_max_depth or truncated_next_flag

        info = {}
        if final_done:
            if self.engine == 'prolog' and self.current_label == 1 and self.current_query_term in self.facts_terms:
                if self.dynamic_consult:
                    try: janus.query_once(f"asserta({str(self.current_query_term)}).")
                    except: pass

            self.current_query_term, self.current_label, self.current_query_ids = None, None, None
            self.last_query_term = None
            info["episode"] = {
                "r": float(reward_next.item()),
                "l": int(self.current_depth.item()),
                "is_success": bool(reward_next.item() > 0)
            }

        # Padding for Observation
        padded_next_state_ids = self._pad_state_ids(next_state_ids)
        padded_derived_next_ids = self._pad_derived_states_ids(derived_states_next_ids)
        self.num_valid_actions = len(derived_states_next_ids)

        # Update TensorDict
        self.tensordict.update(TensorDict(
            {
                "state_ids_pad": padded_next_state_ids,
                "state_ids": NonTensorData(data=next_state_ids),
                "next_temp_id": torch.tensor(next_next_temp_id, device=self.device),
                "done": final_done,
                "reward": reward_next,
                "derived_states_ids_pad": padded_derived_next_ids,
                "derived_states_ids": NonTensorData(data=derived_states_next_ids),
            }, batch_size=[]
        ))

        # Create Observation for Gym
        obs = {
            'state_ids': padded_next_state_ids.cpu().numpy(),
            'derived_state_ids': padded_derived_next_ids.cpu().numpy(),
        }

        reward_val = reward_next.cpu().numpy()
        terminated = bool(done_next and not final_truncated)
        truncated = bool(final_truncated)

        if self.verbose:
            current_terms = self.index_manager.ids_to_state(next_state_ids)
            derived_terms = [self.index_manager.ids_to_state(s_ids) for s_ids in derived_states_next_ids]
            print_state_transition(current_terms, derived_terms, reward_next, final_done, action=action, truncated=truncated)
        return obs, reward_val, terminated, truncated, info


    def _get_state_repr_for_memory(self, state_ids: List[Tuple[int, ...]]) -> Optional[str]:
        """Creates a canonical, hashable representation of a state ID list for memory."""
        # Filter out True/False atoms
        filtered_ids = [
            ids for ids in state_ids
            if ids and ids[0] not in (self.index_manager.true_pred_id, self.index_manager.false_pred_id)
            # Add End ID check if using end_proof_action
        ]
        if not filtered_ids: return None
        try:
            canonical_tuple = tuple(sorted(filtered_ids))
            return str(canonical_tuple)
        except TypeError:
             print(f"Warning: Could not create hashable representation for state IDs: {state_ids}")
             return None


    def get_next_states(self, state_ids: List[Tuple[int,...]], current_next_temp_id: int) -> Tuple[List[List[Tuple[int,...]]], int, bool]:
        """
        Get next possible states using ID-based unification.
        Manages temporary variables and applies pruning/truncation.
        Includes atom truncation *within* the skip_unary_actions loop.
        Returns: (list_of_next_states_ids, updated_next_temp_id, truncated_flag)
        """
        local_next_temp_id = current_next_temp_id
        truncated_flag = False
        false_state_ids = [(self.index_manager.false_pred_id,)]
        true_state_ids = [(self.index_manager.true_pred_id,)]

        # Initial unification call
        current_derived_states_ids, local_next_temp_id = get_next_unification_ids(
            state_ids=state_ids,
            facts_ids_set=self.facts_as_ids_set,
            rules_as_ids=self.index_manager.rules_as_ids,
            index_manager=self.index_manager,
            next_temp_var_id=local_next_temp_id,
            verbose=self.prover_verbose
        )

        # Skip Unary Actions Loop
        if self.skip_unary_actions:
            initial_state_ids_for_skip = state_ids # Keep track for debugging if needed
            counter = 0
            while (len(current_derived_states_ids) == 1 and
                   current_derived_states_ids[0] != true_state_ids and
                   current_derived_states_ids[0] != false_state_ids):

                current_terms = self.index_manager.ids_to_state(state_ids)
                derived_terms = [self.index_manager.ids_to_state(s_ids) for s_ids in current_derived_states_ids]
                print('\n*********') if self.verbose else None
                print(f"Skipping unary action:") if self.verbose else None
                print('State', current_terms) if self.verbose else None
                print('Next State:',derived_terms) if self.verbose else None
                print('\n') if self.verbose else None

                intermediate_state_ids = current_derived_states_ids[0]
                current_state_repr = self._get_state_repr_for_memory(intermediate_state_ids)

                # --- Memory Pruning Check within Loop ---
                if self.memory_pruning and current_state_repr and current_state_repr in self.memory:
                    if self.verbose >= 1: print(f"Unary skip loop detected (state {current_state_repr} in memory). Ending in False.")
                    current_derived_states_ids = [false_state_ids]
                    truncated_flag = True
                    break # Exit while loop

                if self.memory_pruning and current_state_repr:
                    self.memory.add(current_state_repr)

                # --- Unify from the intermediate state ---
                current_derived_states_ids, local_next_temp_id = get_next_unification_ids(
                    state_ids=intermediate_state_ids,
                    facts_ids_set=self.facts_as_ids_set,
                    rules_as_ids=self.index_manager.rules_as_ids,
                    index_manager=self.index_manager,
                    next_temp_var_id=local_next_temp_id,
                    verbose=self.prover_verbose
                )

                # --- Truncate Max Atoms per State *within* Loop ---
                if self.truncate_atoms:
                    original_count = len(current_derived_states_ids)
                    filtered_derived_states_in_loop = []
                    exceeded_mask_debug_in_loop = []
                    for next_s_ids_in_loop in current_derived_states_ids:
                        is_exceeded_in_loop = len(next_s_ids_in_loop) >= self.padding_atoms
                        exceeded_mask_debug_in_loop.append(is_exceeded_in_loop)
                        if not is_exceeded_in_loop:
                            filtered_derived_states_in_loop.append(next_s_ids_in_loop)

                    if self.verbose >= 1 and any(exceeded_mask_debug_in_loop):
                        print(f"Exceeded max atoms: {[len(s_ids) for s_ids in current_derived_states_ids]}")

                    current_derived_states_ids = filtered_derived_states_in_loop
                    # If truncation removed all states, end in False
                    if not current_derived_states_ids and original_count > 0:
                         if self.verbose >=1: print("Unary skip atom truncation removed all states. Ending in False.")
                         current_derived_states_ids = [false_state_ids]
                         truncated_flag = True
                         # No need to break here, the while condition will handle it or loop finishes

                print('\n') if self.verbose else None
                # print('Updated State', current_terms) if self.verbose else None
                derived_terms = [self.index_manager.ids_to_state(s_ids) for s_ids in current_derived_states_ids] 
                print('Updated Next State:',derived_terms) if self.verbose else None
                print('*********\n') if self.verbose else None

                # --- Loop counter / Max iterations check ---
                counter += 1
                if counter > 20: # Increased limit slightly
                    print('Max iterations reached') if self.verbose else None
                    current_derived_states_ids = [false_state_ids]
                    truncated_flag = True
                    break # Exit while loop

        # --- Processing *after* the skip_unary_actions loop ---

        # Memory Pruning (on the final set of states from the loop)
        if self.memory_pruning:
            # Add the original state passed to the function to memory if not already done
            original_state_repr = self._get_state_repr_for_memory(state_ids)
            if original_state_repr and original_state_repr not in self.memory:
                 self.memory.add(original_state_repr)

            filtered_derived_states = []
            visited_mask_debug = []
            for next_s_ids in current_derived_states_ids:
                # Don't add terminal states (True/False) to memory checks here usually,
                # but representer handles filtering them out anyway.
                next_s_repr = self._get_state_repr_for_memory(next_s_ids)
                is_visited = bool(next_s_repr and next_s_repr in self.memory)
                visited_mask_debug.append(is_visited)
                if not is_visited:
                    filtered_derived_states.append(next_s_ids)
                # else:
                    # Optionally add already visited states back to memory if needed for some logic? Usually not.
                    # if next_s_repr: self.memory.add(next_s_repr) # Add visited states' reps too?

            if self.verbose >= 1 and any(visited_mask_debug):
                current_terms = self.index_manager.ids_to_state(state_ids)
                derived_terms = [self.index_manager.ids_to_state(s_ids) for s_ids in current_derived_states_ids]
                print('\n*********') if self.verbose else None
                print(f"current state: {current_terms}") if self.verbose else None
                print(f"next states: {current_terms}") if self.verbose else None
                print(f"Visited mask: {visited_mask_debug}") if self.verbose else None
                # sort the memory to see the visited states
                # print(f"Memory sorted: {sorted(self.memory)}") if self.verbose else None
                print('*******\n') if self.verbose else None

            current_derived_states_ids = filtered_derived_states

        # Truncate Max Atoms per State (Final Check - Redundant if done perfectly in loop, but safer)
        # This section is now potentially redundant if the check inside the loop is comprehensive,
        # but keeping it acts as a safeguard ensures the final output meets the criteria.
        # If you are *certain* the loop handles all cases, you could potentially remove this block.
        if self.truncate_atoms:
            final_filtered_derived_states = []
            final_exceeded_mask_debug = []
            for next_s_ids in current_derived_states_ids:
                is_exceeded = len(next_s_ids) >= self.padding_atoms
                final_exceeded_mask_debug.append(is_exceeded)
                if not is_exceeded:
                    final_filtered_derived_states.append(next_s_ids)

            if self.verbose >= 1 and any(final_exceeded_mask_debug):
                print(f"Exceeded max atoms: {[len(s_ids) for s_ids in current_derived_states_ids]}")

            current_derived_states_ids = final_filtered_derived_states

        # Truncate Max Derived States
        if self.truncate_states:
            max_num_states = self.padding_states
            if len(current_derived_states_ids) > max_num_states:
                if self.verbose >= 1: print(f"Truncating {len(current_derived_states_ids)} derived states to {max_num_states} (post-skip)")
                # Sort by length might be good heuristic to keep smaller states
                current_derived_states_ids = sorted(current_derived_states_ids, key=len)
                current_derived_states_ids = current_derived_states_ids[:max_num_states]
                # truncated_flag = True # Indicate states were dropped, might affect termination logic?

        # Add End Action (if used) - check logic carefully based on `end_pred_id`
        # if self.end_proof_action: ...

        # Handle No Derived States -> False
        if not current_derived_states_ids:
            # Check if it was already False state, avoid printing misleading message
            if not (len(state_ids) == 1 and state_ids[0] == false_state_ids):
                if self.verbose >= 1: print("No valid next states after processing/filtering. Ending in False.")
            current_derived_states_ids = [false_state_ids]
            # Setting truncated_flag might be important if this case should end the episode
            truncated_flag = True # Indicate forced False state

        return current_derived_states_ids, local_next_temp_id, truncated_flag

    # --- get_done_reward remains the same ---
    def get_done_reward(self, state_ids: List[Tuple[int, ...]], label: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get done and reward based on the ID representation of the state."""
        assert label is not None, "Label is None"
        is_false_state = state_ids == [(self.index_manager.false_pred_id,)]
        is_true_state = state_ids == [(self.index_manager.true_pred_id,)]
        # is_end_state = self.end_proof_action and state_ids == [(self.index_manager.end_pred_id,)]

        done = is_true_state or is_false_state # or is_end_state
        successful = is_true_state

        done_tensor = torch.tensor(done, dtype=torch.bool, device=self.device)
        successful_tensor = torch.tensor(successful, dtype=torch.bool, device=self.device)
        label_tensor = torch.tensor(label == 1, dtype=torch.bool, device=self.device)

        reward = torch.tensor(1.0, dtype=torch.float32, device=self.device) if (successful_tensor and label_tensor) else torch.tensor(0.0, dtype=torch.float32, device=self.device)
        # Optional penalties...

        return done_tensor, reward


    # --- get_random_queries remains the same ---
    def get_random_queries(self,
                           queries_terms: List[Term], # Takes Term list
                           n: int = 1,
                           labels: Optional[List[int]] = None
                           ) -> Tuple[Union[Term, List[Term]], Union[Optional[int], List[Optional[int]]]]:
        """Get random queries (as Terms) from a list."""
        # ... (implementation is unchanged) ...
        if not queries_terms: return (Term('False',[]), 0) if n==1 else ([],[]) # Handle empty
        actual_n = min(n, len(queries_terms)) # Adjust n if list is smaller
        sampled_indices = self.seed_gen.sample(range(len(queries_terms)), actual_n)
        sampled_queries = [queries_terms[i] for i in sampled_indices]
        sampled_labels = [labels[i] for i in sampled_indices] if labels and len(labels) == len(queries_terms) else [None] * actual_n

        if n == 1: # Return single element even if actual_n was 0
            return sampled_queries[0] if sampled_queries else Term('False',[]), sampled_labels[0] if sampled_labels else 0
        else:
            return sampled_queries, sampled_labels


    # --- _end_in_false_ids remains the same ---
    def _end_in_false_ids(self, current_next_temp_id: int) -> Tuple[List[List[Tuple[int,...]]], int]:
        """ Return a state list containing only the False state ID tuple. """
        false_state_ids = [[(self.index_manager.false_pred_id,)]]
        return false_state_ids, current_next_temp_id


    # --- get_negatives remains the same ---
    def get_negatives(self, state_term: Term, all_negatives=False) -> List[Term]:
        """Generate negative examples (Terms) by corrupting a positive example Term."""
        # ... (implementation is unchanged) ...
        if not self.sampler or not self.triples_factory:
             print("Warning: Sampler or TriplesFactory not available for get_negatives.")
             return []
        if len(state_term.args) != 2:
             print(f"Warning: Cannot generate KGE negatives for non-binary predicate: {state_term}")
             return []

        query = [(state_term.args[0], state_term.predicate, state_term.args[1])]
        try:
             positive_batch = self.triples_factory.map_triples(np.array(query))
        except Exception as e:
             print(f"Error mapping triple {query}: {e}")
             return []

        if all_negatives:
             negative_batch = self.sampler.corrupt_batch_all(positive_batch)
        else:
             negative_batch = self.sampler.corrupt_batch(positive_batch)
             padding_const_id = 0 # Assuming 0 is padding
             for batch in negative_batch:
                 for n in batch:
                     if n[0].item() == padding_const_id: n[0] = positive_batch[0][0]
                     if n[2].item() == padding_const_id: n[2] = positive_batch[0][2]


        negative_terms: List[Term] = []
        # Add try-except around index access
        try:
            for batch in negative_batch:
                 for n in batch:
                     try:
                         h_idx, r_idx, t_idx = n[0].item(), n[1].item(), n[2].item()
                         # Ensure indices are valid before accessing dictionaries
                         if h_idx not in self.index_manager.constant_idx2str or \
                            r_idx not in self.index_manager.predicate_idx2str or \
                            t_idx not in self.index_manager.constant_idx2str:
                             # print(f"Warning: Invalid index in negative sample {n.tolist()}. Skipping.")
                             continue

                         h_str = self.index_manager.constant_idx2str[h_idx]
                         r_str = self.index_manager.predicate_idx2str[r_idx]
                         t_str = self.index_manager.constant_idx2str[t_idx]

                         # Check against padding ID 0 if used explicitly
                         if h_idx == 0 or t_idx == 0:
                              # print(f"Warning: Negative sample contains padding ID 0: ({h_str}, {r_str}, {t_str}). Skipping.")
                              continue
                         negative_terms.append(Term(predicate=r_str, args=[h_str, t_str]))
                     except KeyError as e:
                         print(f"Error converting negative sample IDs {n.tolist()} to strings: Missing key {e}")
                     except Exception as e:
                          print(f"Unexpected error converting negative sample {n.tolist()}: {e}")
        except IndexError as e:
             print(f"Error accessing negative batch elements: {e}") # Handle cases where negative_batch might be malformed


        return negative_terms