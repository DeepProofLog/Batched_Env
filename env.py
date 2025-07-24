from typing import List, Optional, Tuple, Set
import random
from tensordict import TensorDict, NonTensorData
import torch
import gymnasium as gym
import numpy as np
# import janus_swi as janus

from dataset import DataHandler
from index_manager import IndexManager
from utils import Rule, Term, print_state_transition
from python_unification import get_next_unification_python
# from python_unification import get_next_unification_python_old as get_next_unification_python
# from prolog_unification import get_next_unification_prolog

def _state_to_hashable(state: List[Term]) -> frozenset:
    """
    Converts a list of Term objects to a hashable frozenset representation.
    A frozenset is inherently order-independent and efficient for hashing.
    """
    if not state:
        return frozenset()
    return frozenset((term.predicate, tuple(term.args)) for term in state)

class LogicEnv_gym(gym.Env):
    batch_locked = False
    
    def __init__(self,
                index_manager: Optional[IndexManager] = None,
                data_handler: Optional[DataHandler] = None,
                queries: Optional[List[Term]] = None,
                labels: Optional[List[int]] = None,
                facts: Optional[Set[Term]] = None,
                mode: str = 'train',
                corruption_mode: Optional[str] = None,
                corruption_scheme: Optional[List[str]] = None,
                train_neg_ratio: int = 1,
                seed: Optional[int] = None,
                max_depth: int = 10,
                memory_pruning: bool = True,
                end_proof_action: bool = False,
                skip_unary_actions: bool = False,
                padding_atoms: int = 10,
                padding_states: int = 20,
                verbose: int = 0,
                prover_verbose: int = 0,
                device: torch.device = torch.device("cpu"),
                engine: str = 'python',
                use_kge_action: bool = False,
                ):

        '''Initialize the environment'''
        super().__init__()

        self.engine = engine

        self.verbose = verbose
        self.prover_verbose = prover_verbose
        self.device = device

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
        self.use_kge_action = use_kge_action

        if self.corruption_mode:
            self.counter = 0  # Determine whether to sample from positive or negative queries in KGE settings
            
        if self.corruption_mode:
            self.sampler = data_handler.sampler
            self.corruption_scheme = corruption_scheme
        
        self.janus_file = data_handler.janus_path
        self.janus_facts = data_handler.janus_facts

        self.memory = set() # Store grounded predicates, avoid loop
        self.memory_pruning = memory_pruning # two ways to avoid loop: limit action space, stop when a state has been visited
        self.end_proof_action = end_proof_action # Add the action 'end of the proof' to the action space
        self.skip_unary_actions = skip_unary_actions # Skip unary actions in the action space
        self.predicate_false_idx = index_manager.predicate_str2idx['False'] 

        self.current_query = None
        self.current_label = None

        self.mode = mode
        self.queries = queries
        self.labels = labels
        self.n_episodes = len(queries) if queries is not None else 0
        self.eval_idx = 0
        self.consult_janus_eval = False
        self.next_var_index = self.index_manager.variable_start_index

        if self.mode == 'train':
            self.train_neg_ratio = train_neg_ratio
            assert self.sampler.num_negs_per_pos <3, f"Sampler num_negs_per_pos should <3, but is {self.sampler.num_negs_per_pos}"

    def _set_seed(self, seed:int):
        '''Set the seed for the environment. If no seed is provided, generate a random one'''
        self.seed = seed if seed is not None else torch.empty((), dtype=torch.int64).random_().item()
        rng = torch.manual_seed(seed)
        self.rng = rng
        self.seed_gen = random.Random(seed)


    def _make_spec(self):
        '''Create the observation and action specs'''
        obs_spaces = {
            'sub_index': gym.spaces.Box(
                low=float('-inf'),
                high=float('inf'),
                shape = torch.Size([1])+ torch.Size([self.padding_atoms])+ torch.Size([self.max_arity+1]),
                dtype=np.int64,
            ),

            'derived_sub_indices': gym.spaces.Box(
                low=float('-inf'),
                high=float('inf'),
                shape=torch.Size([self.padding_states])+torch.Size([self.padding_atoms])+torch.Size([self.max_arity+1]),
                dtype=np.int64,
            ),  
        }
        self.observation_space = gym.spaces.Dict(obs_spaces)
        self.action_space = gym.spaces.Discrete(self.padding_states)


    def reset(self, seed: Optional[int]= None, options=None):
        print('\n\nReset-----------------------------') if self.verbose or self.prover_verbose else None
        '''Reset the environment and get a new query based on the environment configuration'''
        if self.mode == 'eval':
            '''When the number of episodes is reached, we set the eval mask to false
            so that the other envs can finish the eval. After n_episodes, reset with a False state'''
            if self.eval_idx < self.n_episodes:
                state = self.queries[self.eval_idx]
                label = self.labels[self.eval_idx]
            else:
                state = Term(predicate='False', args=())
                label = 0
            self.eval_idx += 1

        elif self.mode == 'eval_with_restart':
            if self.eval_idx == self.n_episodes:
                self.eval_idx = 0 

            assert self.eval_idx < self.n_episodes, f"Eval index: {self.eval_idx}, n_episodes: {self.n_episodes}. "\
                    f"Adjust the number of episodes to evaluate in the callback."
            
            state = self.queries[self.eval_idx]
            label = self.labels[self.eval_idx]
            self.eval_idx += 1

        elif self.mode == 'train':
            if self.corruption_mode:
                state, _ = self.get_random_queries(self.queries, n=1)
                label = 1
                if self.counter % (int(self.train_neg_ratio) + 1) != 0:
                    # Determine the number of negatives to request from the sampler
                    num_to_generate = len(self.corruption_scheme) if len(self.corruption_scheme) > 1 else 1
                    
                    negative_samples = self.sampler.get_negatives_from_states(state, self.device, num_negs=num_to_generate)
                    if len(self.corruption_scheme) > 1:
                        if not hasattr(self, 'negation_toggle'):
                            self.negation_toggle = 0  # Initialize
                        
                        # Select head or tail corruption based on the toggle
                        state = [negative_samples[self.negation_toggle]]
                        self.negation_toggle = 1 - self.negation_toggle  # Flip for the next time
                    else:
                        state = negative_samples
                    assert len(state) == 1, f"Length of negatives should be 1, but is {len(state)}"
                    state = state[0] # In train there should be only one negative
                    label = 0
                self.counter += 1

            else: # Default case
                state, _ = self.get_random_queries(self.queries, n=1)
                label = 1
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Choose from 'train', 'eval', or 'eval_with_restart'.")
                
        # if self.engine == 'prolog' and (self.mode == 'train' or self.consult_janus_eval == True) and label == 1:
        #     if state in self.facts:
        #         janus.query_once(f"retract({state.prolog_str()}).")
        #         # janus.query_once(f"retract({state}).")

        self.current_query = state
        self.current_label = label
        return self._reset([state], label)


    def _reset(self, query, label):
        '''Reset the environment to the initial state'''    
        print('Initial query:', query, label) if self.verbose else None
        self.current_depth = torch.tensor(0, device=self.device)
        self.next_var_index = self.index_manager.variable_start_index

        self.memory = set()
        filtered_query = [q for q in query if q.predicate not in ['False', 'True', 'End']]
        self.memory.add(_state_to_hashable(filtered_query))

        sub_index = self.index_manager.get_atom_sub_index(query)
        derived_states, derived_sub_indices, truncated_flag = self.get_next_states(query)
        if truncated_flag: # end in false
            size_sub_index = torch.Size([self.padding_states]) + sub_index.size()
            derived_states, derived_sub_indices = self.end_in_false(size_sub_index)

        self.tensordict = TensorDict(
            {
                "sub_index": sub_index.unsqueeze(0), # to match the shape of derived_sub_indices
                "state": NonTensorData(data=query),
                "label": torch.tensor(label, device=self.device),
                "done": torch.tensor(0, dtype=torch.bool, device=self.device),
                "reward": torch.tensor(0, dtype=torch.float32, device=self.device),
                "derived_states": NonTensorData(data=derived_states),
                "derived_sub_indices": derived_sub_indices,
            },
        )
        obs = {'sub_index': self.tensordict['sub_index'].cpu().numpy(),
               'derived_sub_indices': self.tensordict['derived_sub_indices'].cpu().numpy()}
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
        derived_states_next, derived_sub_indices_next, truncate_flag = self.get_next_states(next_state)
        self.current_depth += 1
        exceeded_max_depth = (self.current_depth >= self.max_depth)
        if exceeded_max_depth: print('\nMax depth reached', self.current_depth.item()) if self.verbose else None

        done_next = done_next | exceeded_max_depth | truncate_flag
        truncated = bool(exceeded_max_depth) or bool(truncate_flag)
        
        info = {}
        if done_next:
            info["is_success"] = successful and not truncated
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
               'derived_sub_indices': self.tensordict['derived_sub_indices'].cpu().numpy()}
        reward = self.tensordict['reward'].cpu().numpy()
        done = self.tensordict['done'].cpu().numpy()
        if self.verbose:
            print_state_transition(self.tensordict['state'], self.tensordict['derived_states'],self.tensordict['reward'], self.tensordict['done'], action=action,truncated=truncated)
        return obs, reward, done, truncated, info


    
    def get_next_states(self,state: List[Term]) -> Tuple[List[List[Term]], torch.Tensor, torch.Tensor]:
        """Get next possible states and their indices for all states in the batch"""
        truncated_flag = False

        # END ACTION MODULE
        if self.end_proof_action: # filter the end of the proof action to get the next states
            if len(state) > 1:
                state = [atom for atom in state if atom.predicate != 'End']
            else:
                if state[0].predicate == 'End':
                    state = [Term(predicate='False', args=())]

        if self.use_kge_action and state:
            filtered_state = [term for term in state if not term.predicate.endswith('_kge')]
            if not filtered_state: state = [Term(predicate='True', args=())]
            else: state = filtered_state
            # CAREFUL WHEN THE FIRST ATOM HAS A VAR AND THERE ARE MORE STATES, NEED TO UNIFY

        # if self.engine == 'prolog':
        #     derived_states, self.next_var_idx = get_next_unification_prolog(state,
        #                                                  next_var_index=self.index_manager.next_var_index, 
        #                                                  verbose=self.prover_verbose)
        if self.engine == 'python':
            derived_states, self.next_var_index = get_next_unification_python(state,
                                                            facts_set=self.facts,
                                                            facts_indexed=self.index_manager.fact_index,
                                                            rules=self.index_manager.rules,
                                                            excluded_fact = self.current_query if self.current_label == 1 else {},
                                                            verbose=self.prover_verbose,
                                                            next_var_index=self.next_var_index,
                                                            )
        # print(f"\nCurrent state after unification: {state} -> derived states: {derived_states}") if self.verbose else None


        if self.skip_unary_actions:
            current_state = state.copy() if isinstance(state, list) else [state]
            counter = 0
            while (len(derived_states) == 1 and 
                derived_states[0] and
                derived_states[0][0].predicate not in ['End', 'False', 'True']):
                print('\n*********') if self.verbose else None
                print(f"Skipping unary action: current state: {current_state} -> derived states: {derived_states}") if self.verbose else None
                print('\n') if self.verbose else None
                counter += 1
                current_state = derived_states[0].copy()    

                if self.use_kge_action and current_state:
                    filtered_state = [term for term in state if not term.predicate.endswith('_kge')]
                    if not filtered_state: state = [Term(predicate='True', args=())]
                    else: state = filtered_state
                    # CAREFUL WHEN THE FIRST ATOM HAS A VAR AND THERE ARE MORE STATES, NEED TO UNIFY

                # if self.engine == 'prolog':
                #     derived_states, self.next_var_idx = get_next_unification_prolog(state,
                #                                 next_var_index=self.next_var_index, 
                #                                 verbose=self.prover_verbose)
                if self.engine == 'python':
                    derived_states, self.next_var_idx = get_next_unification_python(
                        current_state,
                        facts_set=self.facts,
                        facts_indexed=self.index_manager.fact_index,
                        rules=self.index_manager.rules,
                        excluded_fact = self.current_query if self.current_label == 1 else None,
                        verbose=self.prover_verbose,
                        next_var_index=self.next_var_index,
                    )
                # print(f"\nCurrent state after unification: {current_state} -> derived states: {derived_states}") if self.verbose else None


                # MEMORY
                if self.memory_pruning:
                    self.memory.add(",".join(str(s) for s in current_state if s.predicate not in ['False', 'True', 'End']))
                    visited_mask = [",".join(str(s) for s in state) in self.memory for state in derived_states]
                    if any(visited_mask):
                        print(f"Memory: {self.memory}") if self.verbose else None
                        print(f"Visited mask: {visited_mask}. Current state: {current_state} -> Derived states: {derived_states}") if self.verbose else None
                        derived_states = [state for state, is_visited in zip(derived_states, visited_mask) if not is_visited]
                    
                # TRUNCATE MAX ATOMS
                mask_exceeded_max_atoms = [len(state) >= self.padding_atoms for state in derived_states]
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

        final_states = []
        final_sub_indices = []
        # MEMORY MODULE
        if self.memory_pruning:
            self.memory.add(_state_to_hashable([s for s in state if s.predicate not in ['False', 'True', 'End']]))

        for d_state in derived_states:
            # 1. Memory Pruning Check
            if self.memory_pruning and _state_to_hashable(d_state) in self.memory:
                continue

            # 2. Max Atoms Check
            if len(d_state) >= self.padding_atoms:
                if self.verbose:
                    print(f"Exceeded max atoms in next states: {len(d_state)}")
                continue

            # If checks pass, convert to tensor and add to final lists
            final_states.append(d_state)
            final_sub_indices.append(self.index_manager.get_atom_sub_index(d_state))

        derived_states = final_states # Use the filtered list from now on

        # KGE ACTION MODULE
        # It uses the original `state` variable to ensure alignment.
        if self.use_kge_action and state and state[0].predicate not in ['True', 'False', 'End']:
            kge_pred_name = f"{state[0].predicate}_kge"
            kge_action_term = Term(predicate=kge_pred_name, args=state[0].args)
            
            # Avoid adding a duplicate KGE action
            if not any(kge_action_term in derived_state for derived_state in derived_states):
                derived_states.append([kge_action_term])
                final_sub_indices.append(self.index_manager.get_atom_sub_index([kge_action_term]))

        # TRUNCATE MAX STATES
        max_num_states = self.padding_states if not self.end_proof_action else self.padding_states - 1
        if len(derived_states) > max_num_states:
            print(f"Exceeded max next states: {len(derived_states)}") if self.verbose else None
            indices = sorted(range(len(derived_states)), key=lambda k: len(derived_states[k]))
            derived_states = [derived_states[i] for i in indices[:max_num_states]]
            final_sub_indices = [final_sub_indices[i] for i in indices[:max_num_states]]

        # END ACTION MODULE
        if self.end_proof_action:
            if any(atom.predicate not in ('True', 'False') for next_state in derived_states for atom in next_state):
                end_term_state = [Term(predicate='End', args=())]
                derived_states.append(end_term_state)
                final_sub_indices.append(self.index_manager.get_atom_sub_index(end_term_state))
        if not derived_states:
            derived_states = [[Term(predicate='False', args=())]]
            derived_sub_indices = torch.stack([self.index_manager.get_atom_sub_index(s) for s in derived_states])
        else:
            derived_sub_indices = torch.stack(final_sub_indices)

        # Pad derived_sub_indices
        num_derived = len(derived_states)
        if num_derived < self.padding_states:
            padding = torch.zeros(self.padding_states - num_derived, self.padding_atoms, self.max_arity + 1, device=self.device, dtype=torch.int64)
            derived_sub_indices = torch.cat([derived_sub_indices, padding])

        return derived_states, derived_sub_indices, truncated_flag
    
    def get_done_reward(self,state: List[Term], label: int) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """Get done, reward, and success flag for a state."""
        assert label is not None, f"Label is None"

        any_atom_false = any(atom.predicate == 'False' for atom in state)
        all_atoms_true = all(atom.predicate == 'True' for atom in state)

        done = any_atom_false or all_atoms_true
        successful = all_atoms_true

        if self.end_proof_action and any(atom.predicate == 'End' for atom in state):
            done, successful = True, False

        done = torch.tensor(done, device=self.device)
        reward = torch.tensor(1.0, device=self.device) if (done and successful and label == 1) else torch.tensor(0.0, device=self.device)
        
        # if done and successful and label == 1:
        #     reward = torch.tensor(1.0, device=self.device)
        # elif done and not successful and label == 1:
        #     reward = torch.tensor(0.0, device=self.device)
        # elif done and successful and label == 0:
        #     reward = torch.tensor(-1.0, device=self.device)
        # elif done and not successful and label == 0:
        #     reward = torch.tensor(1.0, device=self.device)

        return done, reward, successful
    
    def get_random_queries(self,
                           queries: List[Rule], 
                           n: int = 1, 
                           labels: List[int] = None):
        """Get random queries from a list of queries"""
        assert n <= len(queries), f"Number of queries ({n}) is greater than the number of queries ({len(queries)})"
        sampled_indices = self.seed_gen.sample(range(len(queries)), n)
        sampled_queries = [queries[i] for i in sampled_indices]
        sampled_labels = [labels[i] for i in sampled_indices] if labels else [None] * n

        if n == 1:
            return sampled_queries[0], sampled_labels[0]
        else:
            return sampled_queries, sampled_labels


    def end_in_false(self, sub_indices_shape: torch.Size) -> Tuple[List[List[Term]], torch.Tensor, torch.Tensor]:
        " Return a state that ends in False"
        false_state = Term(predicate="False", args=())

        derived_states_next = [[false_state]]
        derived_sub_indices_next = torch.zeros(sub_indices_shape, device=self.device, dtype=torch.int64)

        false_sub_id = torch.tensor([self.predicate_false_idx, 0, 0], device=self.device, dtype=torch.int64)
        derived_sub_indices_next[0, 0] = false_sub_id
        return derived_states_next, derived_sub_indices_next