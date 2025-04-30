from typing import List, Optional, Tuple, Dict, Union
import random
from math import prod
from utils import Term, print_state_transition
from python_unification import get_next_unification_python
# from python_unification import get_next_unification_python_old as get_next_unification_python

import torch
from tensordict import TensorDict, TensorDictBase, NonTensorData

import torch
from tensordict import TensorDict, TensorDictBase, NonTensorData
from typing import Optional
import gymnasium as gym
import numpy as np

from dataset import Rule
import janus_swi as janus
from dataset import DataHandler
from index_manager import IndexManager

class LogicEnv_gym(gym.Env):
    batch_locked = False  # Allow dynamic batch sizes
    
    def __init__(self,
                index_manager: Optional[IndexManager] = None,
                data_handler: Optional[DataHandler] = None,
                queries: Optional[List[Term]] = None,
                labels: Optional[List[int]] = None,
                mode: str = 'train',
                corruption_mode: Optional[str] = None,
                corruption_scheme: Optional[List[str]] = None,
                train_neg_pos_ratio: int = 1,
                seed: Optional[int] = None,
                dynamic_consult: bool = True,
                max_depth: int = 10,
                memory_pruning: bool = True,
                end_proof_action: bool = False,
                skip_unary_actions: bool = False,
                truncate_atoms: bool = False,
                truncate_states: bool = False,
                padding_atoms: int = 10,
                padding_states: int = 20,
                verbose: int = 0,
                prover_verbose: int = 0,
                device: torch.device = torch.device("cpu"),
                engine: str = 'python',
                ):
        
        '''Initialize the environment'''
        super().__init__()

        # self.engine = 'prolog'
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
        # self.facts = data_handler.facts
        self.facts = set(data_handler.facts)

        self.corruption_mode = corruption_mode

        if 'static' in self.corruption_mode or 'dynamic' in self.corruption_mode:
            self.counter = 0  # Determine whether to sample from positive or negative queries in KGE settings
            
        if self.corruption_mode == "dynamic":
            self.sampler = data_handler.sampler
            self.triples_factory = data_handler.triples_factory
            self.corruption_scheme = corruption_scheme
        
        self.janus_file = data_handler.janus_path
        self.janus_facts = data_handler.janus_facts

        self.memory = set() # Store grounded predicates, avoid loop
        self.memory_pruning = memory_pruning # two ways to avoid loop: limit action space, stop when a state has been visited
        self.end_proof_action = end_proof_action # Add the action 'end of the proof' to the action space
        self.skip_unary_actions = skip_unary_actions # Skip unary actions in the action space
        self.truncate_atoms = truncate_atoms # Truncate atoms to a fixed size
        self.truncate_states = truncate_states
        self.predicate_false_idx = index_manager.predicate_str2idx['False'] 

        self.dynamic_consult = dynamic_consult
        self.current_query = None
        self.current_label = None
        self.last_query = None  # Track last query for dynamic consulting

        assert mode in ['train', 'eval', 'eval_corr'], f"Invalid mode: {mode}"
        self.mode = mode
        self.queries = queries
        self.labels = labels
        self.n_episodes = len(queries)
        self.eval_idx = 0
        self.consult_janus_eval = False

        if self.mode == 'train':
            self.train_neg_pos_ratio = train_neg_pos_ratio
            if self.corruption_mode == "static":
                self.neg_queries = data_handler.neg_train_queries
                self.neg_labels = [0]*len(self.neg_queries)
                self.pos_queries = self.queries
                self.pos_labels = self.labels
        self.counter_q = 0        

    def _set_seed(self, seed:int):
        '''Set the seed for the environment. If no seed is provided, generate a random one'''
        self.seed = seed if seed is not None else torch.empty((), dtype=torch.int64).random_().item()
        rng = torch.manual_seed(seed)
        self.rng = rng
        # create a seed generator for the environment
        self.seed_gen = random.Random(seed)


    def _make_spec(self):
        '''Create the observation and action specs'''
        obs_spaces = {
            'sub_index': gym.spaces.Box(
                low=float('-inf'),
                high=float('inf'),
                # shape=torch.Size([self.padding_atoms])+torch.Size([self.max_arity+1]),
                shape = torch.Size([1])+ torch.Size([self.padding_atoms])+ torch.Size([self.max_arity+1]),
                dtype=np.int64,
            ),
            # 'atom_index': gym.spaces.Box(
            #     low=float('-inf'),
            #     high=float('inf'),
            #     # shape=torch.Size([self.padding_atoms]),
            #     shape = torch.Size([1])+ torch.Size([self.padding_atoms]),
            #     dtype=np.int64,
            # ),
            # 'derived_atom_indices': gym.spaces.Box(
            #     low=float('-inf'),
            #     high=float('inf'),
            #     shape=torch.Size([self.padding_states])+torch.Size([self.padding_atoms]),
            #     dtype=np.int64,
            # ),
            'derived_sub_indices': gym.spaces.Box(
                low=float('-inf'),
                high=float('inf'),
                shape=torch.Size([self.padding_states])+torch.Size([self.padding_atoms])+torch.Size([self.max_arity+1]),
                dtype=np.int64,
            ),
            # 'valid_actions_mask': gym.spaces.Box(
            #     low=0,
            #     high=1,
            #     shape=torch.Size([self.padding_states]),
            #     dtype=np.bool_,
            # ),            
        }
        self.observation_space = gym.spaces.Dict(obs_spaces)
        # Use a fixed action space with the size of padding_states (maximum possible actions)
        self.action_space = gym.spaces.Discrete(self.padding_states)


    def update_action_space(self, derived_states: List[List[Term]]):
        '''Update the action space based on the possible next states 
        To be called every time the possible next states are updated'''
        # No longer updating action space size - it's fixed to padding_states
        # Just tracking the number of valid actions for masking
        self.num_valid_actions = len(derived_states)


    def new_consult_janus(self, query: Term):
        '''Consult janus with new facts
        1. load the original facts
        2. save a file with the new facts
        3. consult janus with the new file

        fact_removed = janus.query(f"retract(({query_str})).")
        janus.query(f"assertz({fact}).")  # Adds new facts directly to the knowledge base.
        '''
        # to compare the query, convert it to str by removing the spaces
        query_str = str(query).replace(' ', '')

        facts = [line for line in self.janus_facts if query_str not in line]
        if self.dataset_name != "mnist_addition":
            assert len(facts) == len(self.janus_facts) - 1, f"Length of facts: {len(facts)}, Length of janus_facts: {len(self.janus_facts)}"

        # 2. save a _tmp file with the new facts
        tmp_file = self.janus_file.replace('.pl', '_tmp.pl')
        with open(tmp_file, "w") as f:
            for line in facts:
                f.write(line)

        # 3. abolish all the facts and tables in janus        
        for predicate, arity in self.predicates_arity.items():
            janus.query_once(f"abolish({predicate}/{arity}).")
        janus.query_once("abolish_all_tables.")
        
        # 4. consult janus with the new file
        janus.consult(tmp_file)

    def dynamic_consult_janus(self, query: Term):
        '''Dynamically manage facts in Janus by retracting and re-asserting as needed'''
        if self.last_query and not self.last_query == query:
            janus.query_once(f"asserta({str(self.last_query)}).")

        if query in self.facts:
            self.last_query = query
            janus.query_once(f"retract({str(self.last_query)}).")
        else:
            self.last_query = None


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

        else:  # Training mode
            if self.corruption_mode == "dynamic":
                state, _ = self.get_random_queries(self.queries, n=1)
                label = 1
                if self.counter % (int(self.train_neg_pos_ratio) + 1) != 0:
                    state = self.get_negatives(state)

                    # Alternate between head and tail
                    if not hasattr(self, 'negation_toggle'):
                        self.negation_toggle = 0  # Initialize if it doesn't exist
                    if len(self.corruption_scheme) > 1:
                        state = [state[self.negation_toggle]]  
                        self.negation_toggle = 1 - self.negation_toggle  # Flip for next time get head or tail
                    
                    assert len(state) == 1, f"Length of negatives: {len(state)}"
                    state = state[0] # In train there should be only one negative
                    label = 0
                self.counter += 1

            elif self.corruption_mode == "static":
                if self.counter % (int(self.train_neg_pos_ratio) + 1) == 0:
                    state, _ = self.get_random_queries(self.pos_queries, n=1) 
                    label = 1
                else:
                    state, _ = self.get_random_queries(self.neg_queries, n=1) 
                    label = 0
                self.counter += 1

            else: # Default case
                state, _ = self.get_random_queries(self.queries, n=1)
                label = 1
                
        if self.engine == 'prolog' and (self.mode == 'train' or self.consult_janus_eval == True) and label == 1:
            if not self.dynamic_consult:
                self.new_consult_janus(state)
            elif self.dynamic_consult and state in self.facts:
                janus.query_once(f"retract({str(state)}).")

        # brother(144, 2032) start with this as test
        # state = Term(predicate='brother', args=('144', '2032'))
        # label = 1
        # if state == Term(predicate='brother', args=('144', '2032')): print(kszhbdv)
        self.current_query = state
        self.current_label = label

        return self._reset([state], label)


    def _reset(self, query, label):
        '''Reset the environment to the initial state'''    
        print('Initial query:', query, label) if self.verbose else None
        self.current_depth = torch.tensor(0, device=self.device)
        self.index_manager.reset_atom()

        self.memory = set()
        self.memory.add(",".join(str(q) for q in query if q.predicate not in ['False', 'True', 'End']))
        
        sub_index = self.index_manager.get_atom_sub_index(query)
        # print('input state0',query)
        derived_states, derived_sub_indices, truncated_flag = self.get_next_states(query)
        # print('ouput next states0:', derived_states)

        if self.truncate_states and truncated_flag: # end in false
            size_sub_index = torch.Size([self.padding_states]) + sub_index.size()
            derived_states, derived_sub_indices = self.end_in_false(size_sub_index)
        # print('ouput next states0:', derived_states)

        self.tensordict = TensorDict(
            {
                "sub_index": sub_index.unsqueeze(0), # to match the shape of derived_sub_indices
                "state": NonTensorData(data=query),
                "label": torch.tensor(label, device=self.device),
                "done": torch.tensor(0, dtype=torch.bool, device=self.device),
                "reward": torch.tensor(0, dtype=torch.float32, device=self.device)*(-1),
                "derived_states": NonTensorData(data=derived_states),
                "derived_sub_indices": derived_sub_indices,
                # "valid_actions_mask": valid_actions_mask,
            },
        )
        sub_index = self.tensordict['sub_index'].cpu().numpy()
        derived_sub_indices = self.tensordict['derived_sub_indices'].cpu().numpy()
        obs = {'sub_index':sub_index,
               'derived_sub_indices':derived_sub_indices,
            #    'valid_actions_mask':valid_actions_mask
            }
        if self.verbose:
            print_state_transition(self.tensordict['state'], self.tensordict['derived_states'],self.tensordict['reward'], self.tensordict['done'],label=label)
            # print('Number of next states:', len(derived_states)) if self.verbose else None
            # print('N atoms in next states:', [len(state) for state in derived_states],) if self.verbose else None
            # print(f"Next states: {derived_states}\n") if self.verbose else None
            # print('idx derived sub:', list(self.tensordict['derived_sub_indices'][:3,0,:].cpu().numpy()),'\n')
        # print('ouput next states1 (reset) :', self.tensordict['derived_states']) if self.verbose else None
        return obs, {}

    def step(self, action):
        '''
        Given the current state, possible next states, an action, and return the next state.
        (It should be: given the current state, and an action, return the next state, but we need to modify it for our case)
        '''
        derived_states = self.tensordict["derived_states"]
        derived_sub_indices = self.tensordict["derived_sub_indices"]
        # print('ouput next states2 (step):', derived_states) if self.verbose else None
        # Check if the action is valid using the mask
        if action >= self.padding_states or action > len(derived_states):
            raise ValueError(
                # f"Invalid action ({action}). Valid actions are indicated by the mask. {valid_actions_mask}. Derived states: {derived_states}. Derived atom indices: {derived_atom_indices}.")
                f"Invalid action ({action}). Derived states: {derived_states}.")
        # print('action:', action) if self.verbose else None
        next_state = derived_states[action]
        next_sub_index = derived_sub_indices[action]

        done_next, reward_next = self.get_done_reward(next_state,self.tensordict['label'].item())
        derived_states_next, derived_sub_indices_next, truncate_flag = self.get_next_states(next_state)
        self.current_depth += 1
        
        exceeded_max_depth = (self.current_depth >= self.max_depth)
        if exceeded_max_depth: print('\nMax depth reached', self.current_depth.item()) if self.verbose else None

        done_next = done_next | exceeded_max_depth | truncate_flag
        
        # Track episode info for SB3
        info = {}
        # Handle episode completion
        if done_next:
            # # Add episode info for Stable Baselines
            # reward_value = float(reward_next.item())
            # episode_length = int(self.current_depth)
            # info["episode"] = {
            #     "r": reward_value,
            #     "l": episode_length
            # }
            # Restore facts in knowledge base if this was a positive query
            if self.engine == 'prolog' and self.current_label == 1 and self.current_query in self.facts:
                if self.dynamic_consult:
                    # Re-add the current query to Janus knowledge base
                    janus.query_once(f"asserta({str(self.current_query)}).")
            # Clear current query reference and label
            self.current_query, self.current_label = None, None

        tensordict = TensorDict(
            {
                "sub_index": next_sub_index.unsqueeze(0), # to match the shape of derived_sub_indices
                "state": NonTensorData(data=next_state),
                "label": self.tensordict['label'],
                "done": done_next,
                "reward": reward_next,
                "derived_states": NonTensorData(data=derived_states_next),
                "derived_sub_indices": derived_sub_indices_next,
            },
            )

        self.tensordict = tensordict
        sub_index = self.tensordict['sub_index'].cpu().numpy()
        derived_sub_indices = self.tensordict['derived_sub_indices'].cpu().numpy()
        obs = {'sub_index':sub_index, 
               'derived_sub_indices':derived_sub_indices,
               }

        reward = self.tensordict['reward'].cpu().numpy()
        done = self.tensordict['done'].cpu().numpy()
        # print('exceeded max depth:',exceeded_max_depth, 'truncate_flag:',truncate_flag) if self.verbose else None
        truncated = bool(exceeded_max_depth) or bool(truncate_flag)
        if self.verbose:
            print_state_transition(self.tensordict['state'], self.tensordict['derived_states'],self.tensordict['reward'], self.tensordict['done'], action=action,truncated=truncated)
            # print('Number of next states:', len(derived_states)) if self.verbose else None
            # print('N atoms in next states:', [len(state) for state in derived_states]) if self.verbose else None
            # print(f"Next states: {derived_states}\n") if self.verbose else None
            # print('idx derived sub:', list(self.tensordict['derived_sub_indices'][:3,0,:].cpu().numpy()),'\n')
        # print('ouput next states3 (step):', derived_states) if self.verbose else None
        return obs, reward, done, truncated, info


    
    def get_next_states(self,state: List[Term]) -> Tuple[List[List[Term]], torch.Tensor, torch.Tensor]:
        self.counter_q += 1
        # print('Counter:', self.counter_q) 
        """Get next possible states and their indices for all states in the batch"""
        truncated_flag = False

        # END ACTION MODULE
        if self.end_proof_action: # filter the end of the proof action to get the next states
            assert len(state) > 0, f"State is empty"
            if len(state) > 1:
                state = [atom for atom in state if atom.predicate != 'End']
            else:
                if state[0].predicate == 'End':
                    state = [Term(predicate='False', args=())]


        assert self.engine == 'python', "Only python unification is supported"
        
        # print('input state:', state)
        derived_states = get_next_unification_python(state,
                                                        facts_set=self.facts,
                                                        facts_indexed=self.index_manager.fact_index,
                                                        rules=self.index_manager.rules,
                                                        excluded_fact = self.current_query if self.current_label == 1 else {},
                                                        verbose=self.prover_verbose
                                                        )
        # print('ouput next states:', derived_states)

        if self.skip_unary_actions:
            current_state = state.copy() if isinstance(state, list) else [state]
            counter = 0
            while (len(derived_states) == 1 and 
                derived_states[0] and  # Ensure not empty
                derived_states[0][0].predicate not in ['End', 'False', 'True']):
                print('\n*********') if self.verbose else None
                print(f"Skipping unary action:") if self.verbose else None
                print('State', current_state) if self.verbose else None
                print('Next State:',derived_states) if self.verbose else None
                print('\n') if self.verbose else None
                counter += 1
                current_state = derived_states[0].copy()    

                derived_states = get_next_unification_python(
                    current_state,
                    facts_set=self.facts,
                    facts_indexed=self.index_manager.fact_index,
                    rules=self.index_manager.rules,
                    excluded_fact = self.current_query if self.current_label == 1 else None,
                    verbose=self.prover_verbose
                )
                # print('Derived states:', derived_states)

                # MEMORY
                if self.memory_pruning:
                    self.memory.add(",".join(str(s) for s in current_state if s.predicate not in ['False', 'True', 'End']))
                    visited_mask = [",".join(str(s) for s in state) in self.memory for state in derived_states]
                    if any(visited_mask):
                        print(f"Current state: {current_state}") if self.verbose else None
                        print(f"Next states: {derived_states}") if self.verbose else None
                        print(f"Memory: {self.memory}") if self.verbose else None
                        print(f"Visited mask: {visited_mask}") if self.verbose else None
                        derived_states = [state for state, is_visited in zip(derived_states, visited_mask) if not is_visited]
                    
                # TRUNCATE MAX ATOMS
                if self.truncate_atoms:
                    mask_exceeded_max_atoms = [len(state) >= self.padding_atoms for state in derived_states]
                    print(f" Exceeded max atoms: {[len(state) for state in derived_states]}") if self.verbose and any(mask_exceeded_max_atoms) else None
                    derived_states = [state for state, is_exceeded in zip(derived_states, mask_exceeded_max_atoms) if not is_exceeded]
                
                print('\n') if self.verbose else None
                print(f"Updated Next State: {derived_states}") if self.verbose else None
                print('*********\n') if self.verbose else None

                if counter > 20:
                    print('Max iterations reached') if self.verbose else None
                    derived_states = [[Term(predicate='False', args=())]]
                    # print(f'Setting truncated flag to True. Counter: {counter}') if self.verbose else None
                    truncated_flag = True
                    break


        # MEMORY MODULE
        if self.memory_pruning:
            # Please do this only if the state is not false or true or end
            self.memory.add(",".join(str(s) for s in state if s.predicate not in ['False', 'True', 'End']))
            visited_mask = [",".join(str(s) for s in state) in self.memory for state in derived_states]

            if any(visited_mask):
                print('\n-----------') if self.verbose else None
                print(f"Current state: {state}") if self.verbose else None
                print(f"Next states: {derived_states}") if self.verbose else None
                print(f"Memory: {self.memory}") if self.verbose else None
                print(f"Visited mask: {visited_mask}") if self.verbose else None
                print('-----------\n') if self.verbose else None
            # if len([state for state, is_visited in zip(derived_states, visited_mask) if not is_visited]) > 0 and self.verbose:
            #     print(f"Removing visited states: {[state for state, is_visited in zip(derived_states, visited_mask) if is_visited]}")
                derived_states = [state for state, is_visited in zip(derived_states, visited_mask) if not is_visited]

        # TRUNCATE MAX ATOMS
        if self.truncate_atoms:
            mask_exceeded_max_atoms = [len(state) >= self.padding_atoms for state in derived_states]
            if any(mask_exceeded_max_atoms):
                print(f"State {state}. Next states: {derived_states}. Exceeded max atoms in next states: {[len(state) for state in derived_states]}") #if self.verbose else None
            # if len([state for state, is_exceeded in zip(derived_states, mask_exceeded_max_atoms) if is_exceeded]) > 0 and self.verbose:
            #     print(f" Exceeded max atoms: {[len(state) for state in derived_states]}") if self.verbose else None
            derived_states = [state for state, is_exceeded in zip(derived_states, mask_exceeded_max_atoms) if not is_exceeded]

        # TRUNCATE MAX STATES
        if self.truncate_states:
            max_num_states = self.padding_states if not self.end_proof_action else self.padding_states + -1
            if len(derived_states) > max_num_states:
                print(f"State {state}. Next states: {derived_states}. Exceeded max next states: {len(derived_states)}") if self.verbose else None
                # print(f"Exceeded max next states: {len(derived_states)}")
                derived_states = sorted(derived_states, key=lambda x: len(x))
                derived_states = derived_states[:max_num_states]     

        # END ACTION MODULE
        if self.end_proof_action:
            if any(atom.predicate not in ('True', 'False') for next_state in derived_states for atom in next_state):
                derived_states.append([Term(predicate='End', args=())])
            # print(f"Possible next states with end action: {derived_states}") if self.verbose else None       

        if len(derived_states) == 0:
            # print(f'Setting truncated flag to True. Len of derived states is 0. {derived_states}') if self.verbose else None
            derived_states = [[Term(predicate='False', args=())]]
            # truncated_flag = True

        # CREATE INDICES 
        derived_sub_indices = []
        for s in derived_states:
            sub_idx = self.index_manager.get_atom_sub_index(s)
            derived_sub_indices.append(sub_idx)

        derived_sub_indices = torch.stack(derived_sub_indices)
        derived_sub_indices = torch.cat([derived_sub_indices, torch.zeros(self.padding_states - len(derived_sub_indices), self.padding_atoms, self.max_arity+1, device=self.device, dtype=torch.int64)])
        # if self.counter_q == 20:
        #     print (pzjsnbv)
        return derived_states, derived_sub_indices, truncated_flag
    
    def get_done_reward(self,state: List[Term], label: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get done and reward keys for all states in the batch. To be called in the step() method"""
        assert label is not None, f"Label is None"

        any_atom_false = any([atom.predicate == 'False' for atom in state])
        all_atoms_true = all([atom.predicate == 'True' for atom in state])

        done = any_atom_false or all_atoms_true
        successful = all_atoms_true

        if self.end_proof_action and any([atom.predicate == 'End' for atom in state]):
            assert len(state) == 1, f"Length of state: {len(state)} should be 1 when the action is 'End'"
            done = True
            successful = False

        done = torch.tensor(done, device=self.device)
        successful = torch.tensor(successful, device=self.device)
        label_tensor = torch.tensor(label, device=self.device)

        reward = torch.tensor(1, device=self.device) if (done and successful and label_tensor == 1) else torch.tensor(0,device=self.device)
        return done, reward
    
    def get_random_queries(self,
                           queries: List[Rule], 
                           n: int = 1, 
                           labels: List[int] = None):
        """Get random queries from a list of queries"""
        assert n <= len(queries), f"Number of queries ({n}) is greater than the number of queries ({len(queries)})"
        sampled_indices = self.seed_gen.sample(range(len(queries)), n)
        sampled_queries = [queries[i] for i in sampled_indices]
        sampled_labels = [labels[i] for i in sampled_indices] if labels else [None]

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


    def get_negatives(self, state, all_negatives=False):
        """Generate negative examples by corrupting a positive example"""
        query = [(state.args[0], state.predicate, state.args[1])] # convert query to (cte, pred, cte) format
        positive_batch = self.triples_factory.map_triples(np.array(query))
        
        if all_negatives:
            negative_batch = self.sampler.corrupt_batch_all(positive_batch)
        else:
            negative_batch = self.sampler.corrupt_batch(positive_batch)
            # if there is a 0, replace it with another constant from the same batch
            if any([n[0].item()==0 for batch in negative_batch for n in batch]):
                for batch in negative_batch:
                    for n in batch:
                        if n[0].item() == 0:
                            n[0] = torch.tensor(random.choice(list(n[0].item() for n in batch if n[0].item() != 0)), device=self.device)
            if any([n[2].item()==0 for batch in negative_batch for n in batch]):
                for batch in negative_batch:
                    for n in batch:
                        if n[2].item() == 0:
                            n[2] = torch.tensor(random.choice(list(n[2].item() for n in batch if n[2].item() != 0)), device=self.device)
            

        negative_batch_str = []
        for batch in negative_batch:
            for n in batch:
                assert self.index_manager.constant_idx2str[n[0].item()] != 0, f"Negative batch contains 0s, used for padding,{n}"
                assert self.index_manager.constant_idx2str[n[2].item()] != 0, f"Negative batch contains 0s, used for padding,{n}"
                negative_batch_str.append((self.index_manager.constant_idx2str[n[0].item()],self.index_manager.predicate_idx2str[n[1].item()],
                                        self.index_manager.constant_idx2str[n[2].item()]))
        state = [Term(predicate=n[1].strip(), args=(n[0].strip(), n[2].strip())) for n in negative_batch_str] # convert each negative back to Term format
        return state