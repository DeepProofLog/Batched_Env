from typing import List, Optional, Tuple, Dict, Union
import random
from math import prod
from utils import Term, is_variable, extract_var, print_state_transition, get_rule_from_string
from unification.python_unification import get_next_unification_python
# from unification.prolog_unification import get_next_unification_prolog
from unification.prolog_unification_v2 import get_next_unification_prolog

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
from environments.index_manager import IndexManager

class LogicEnv_gym(gym.Env):
    batch_locked = False  # Allow dynamic batch sizes
    
    def __init__(self, 
                max_depth: int = 10,
                seed: Optional[int] = None,
                device: torch.device = torch.device("cpu"),
                index_manager: Optional[IndexManager] = None,
                data_handler: Optional[DataHandler] = None,
                corruption_mode: Optional[str] = None,
                corruption_scheme: Optional[List[str]] = None,
                train_neg_pos_ratio: int = 1,
                limit_space: bool = True,
                dynamic_consult: bool = True,
                eval=False,
                valid_negatives=None,
                end_proof_action: bool = False,
                padding_atoms: int = 10,
                padding_states: int = 20,
                verbose: int = 0,
                ):
        
        '''Initialize the environment'''
        super().__init__()

        self.engine = 'prolog'
        # self.engine = 'python'

        self.verbose = verbose
        self.device = device

        self.corruption_mode = corruption_mode

        self.max_arity = data_handler.max_arity # Maximum arity of the predicates
        self.padding_atoms = padding_atoms  # Maximum number of atoms in a state
        self.padding_states = padding_states # Maximum number of possible next states
        self.max_depth = max_depth # Maximum depth of the proof tree
        self.index_manager = index_manager
        self.predicates_arity = data_handler.predicates_arity

        self._set_seed(seed)
        self._make_spec()

        self.dataset_name = data_handler.dataset_name
        self.facts=data_handler.facts

        if 'static' in self.corruption_mode or 'dynamic' in self.corruption_mode:
            self.counter = 0  # Determine whether to sample from positive or negative queries in KGE settings

        if self.corruption_mode == "dynamic":
            self.sampler = data_handler.sampler
            self.triples_factory = data_handler.triples_factory
            self.corruption_scheme = corruption_scheme

        self.train_neg_pos_ratio = train_neg_pos_ratio
        self.train_queries = data_handler.train_queries
        self.neg_train_queries = data_handler.neg_train_queries

        self.valid_queries = data_handler.valid_queries
        self.test_queries = data_handler.test_queries
        
        self.janus_file = data_handler.janus_path
        self.janus_facts = data_handler.janus_facts

        self.memory = set() # Store grounded predicates, avoid loop
        self.limit_space = limit_space # two ways to avoid loop: limit action space, stop when a state has been visited
        self.end_proof_action = end_proof_action # Add the action 'end of the proof' to the action space
        self.predicate_false_offset = index_manager.predicate_false_offset

        self.dynamic_consult = dynamic_consult
        self.current_query = None
        self.current_label = None
        self.last_query = None  # Track last query for dynamic consulting

        self.eval = eval
        self.eval_dataset = 'validation' # by default, evaluate on the validation set. It can be changed to 'test' or 'train'
        self.eval_idx = 0 # Index to go through all the eval queries
        self.eval_len = len(self.valid_queries) # Number of eval queries (to reset the index)
        self.valid_negatives = valid_negatives
        # generate a random sequence of indices to go through the eval queries (in case we dont want to evaluate on the whole dataset)
        self.eval_seq = list(range(self.eval_len))
        # random.Random(0).shuffle(self.eval_seq) # to get the same sequence every time


    def _set_seed(self, seed:int):
        '''Set the seed for the environment'''
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
                shape=torch.Size([self.padding_atoms])+torch.Size([self.max_arity+1]),
                dtype=np.int64,
            ),
            'atom_index': gym.spaces.Box(
                low=float('-inf'),
                high=float('inf'),
                shape=torch.Size([self.padding_atoms]),
                dtype=np.int64,
            ),
            'derived_atom_indices': gym.spaces.Box(
                low=float('-inf'),
                high=float('inf'),
                shape=torch.Size([self.padding_states])+torch.Size([self.padding_atoms]),
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
        self.action_space = gym.spaces.Discrete(100)

    def update_action_space(self, derived_states: List[List[Term]]):
        '''Update the action space based on the possible next states 
        To be called every time the possible next states are updated'''
        self.action_space = gym.spaces.Discrete(len(derived_states))


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
        '''Reset the environment and get a new query based on the environment configuration'''
        if self.eval: 
            '''Only use this during training with the callback EvalCallback and the env initialised with eval=True. 
            We just test in this case positive queries. Can be adapted to test negative queries as well by using the counter
            and passing neg_{train/val/test}_queries. For final eval, we use reset from query'''
            if self.eval_dataset == 'validation':
                eval_dataset = self.valid_queries
            elif self.eval_dataset == 'test':
                eval_dataset = self.test_queries
            elif self.eval_dataset == 'train':
                eval_dataset = self.train_queries

            if self.eval_idx == self.valid_negatives: # reset the index
                self.eval_idx = 0

            state, label = eval_dataset[self.eval_seq[self.eval_idx]], 1
            self.eval_idx += 1

        else:
            
            if self.corruption_mode == "dynamic":
                state, _ = self.get_random_queries(self.train_queries, n=1)
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
                    state = state[0] # In train there shuold be only one negative
                    label = 0
                self.counter += 1

            elif 'static' in self.corruption_mode:
                if self.counter % (int(self.train_neg_pos_ratio) + 1) == 0:
                    state, _ = self.get_random_queries(self.train_queries, n=1) 
                    label = 1
                else:
                    state, _ = self.get_random_queries(self.neg_train_queries, n=1) 
                    label = 0
                self.counter += 1

            else:  # Default case
                state, _ = self.get_random_queries(self.train_queries, n=1)
                label = 1
                
            if label == 1 and not self.dynamic_consult:
                self.new_consult_janus(state)
            elif label == 1 and self.dynamic_consult and state in self.facts:
                janus.query_once(f"retract({str(state)}).")

        self.current_query = state
        self.current_label = label

        return self._reset([state], label)


    def reset_from_query(self, query, label, consult_janus=False):
        ''' Reset the environment from a given query and label. Consult janus is needed when 
        doing eval of train queries that are in the facts'''
        if consult_janus and label == 1 and self.engine == 'prolog':
            if not self.dynamic_consult:
                self.new_consult_janus(query)
            elif self.dynamic_consult and query in self.facts: 
                    janus.query_once(f"retract({str(query)}).") 

        self.current_query = query
        self.current_label = label

        return self._reset([query], label)

    def _reset(self, query, label):
        '''Reset the environment to the initial state'''    
        self.current_depth = torch.tensor(0, device=self.device)
        self.index_manager.reset_atom()
        self.memory = set()

        self.memory.add(",".join(str(q) for q in query))
        atom_index, sub_index = self.index_manager.get_atom_sub_index(query)

        derived_states, derived_atom_indices, derived_sub_indices = self.get_next_states(query)
        # if self.limit_space: RODRIGO: CAN THIS BE REMOVED?
        #     derived_states, derived_atom_indices, derived_sub_indices = self.limit_action_space(derived_states, derived_atom_indices, derived_sub_indices)
        self.update_action_space(derived_states)

        self.tensordict = TensorDict(
            {
                "atom_index": atom_index.unsqueeze(0),
                "sub_index": sub_index.unsqueeze(0),
                "state": NonTensorData(data=query),
                "label": torch.tensor(label, device=self.device),
                "done": torch.tensor(0, dtype=torch.bool, device=self.device),
                "reward": torch.tensor(0, dtype=torch.float32, device=self.device)*(-1),
                "derived_states": NonTensorData(data=derived_states),
                "derived_atom_indices": derived_atom_indices,
                "derived_sub_indices": derived_sub_indices,
            },
        )
        sub_index = self.tensordict['sub_index'].numpy()
        atom_index = self.tensordict['atom_index'].numpy()
        derived_atom_indices = self.tensordict['derived_atom_indices'].numpy()
        derived_sub_indices = self.tensordict['derived_sub_indices'].numpy()
        obs = {'sub_index':sub_index, 
               'atom_index':atom_index, 
               'derived_atom_indices':derived_atom_indices, 
               'derived_sub_indices':derived_sub_indices}
        if self.verbose:
            print_state_transition(self.tensordict['state'], self.tensordict['derived_states'],self.tensordict['reward'], self.tensordict['done'])
            # print('idx derived sub:', list(self.tensordict['derived_sub_indices'][:3,0,:].numpy()),'\n')
        return obs, {}

    def step(self, action):
        '''
        Given the current state, possible next states, an action, and return the next state.
        (It should be: given the current state, and an action, return the next state, but we need to modify it for our case)
        '''
        derived_atom_indices = self.tensordict["derived_atom_indices"]
        derived_states = self.tensordict["derived_states"]
        derived_sub_indices = self.tensordict["derived_sub_indices"]

        if action >= len(derived_states):
            raise ValueError(
                f"Invalid action ({action}). Max action: {len(derived_states)}, derived_states: {derived_states}")
        next_state = derived_states[action]
        next_atom_index = derived_atom_indices[action]
        next_sub_index = derived_sub_indices[action]

        done_next, reward_next = self.get_done_reward(next_state,self.tensordict['label'].item())

        if not self.limit_space and ",".join(str(s) for s in next_state) in self.memory:
            derived_states_next, derived_atom_indices_next, derived_sub_indices_next = self.end_in_false(derived_atom_indices.size(), derived_sub_indices.size())
        else:
            derived_states_next, derived_atom_indices_next, derived_sub_indices_next = self.get_next_states(next_state)
            if not extract_var(",".join(str(s) for s in next_state)):
                    self.memory.add(",".join(str(s) for s in next_state))
            if self.limit_space:
                derived_states_next, derived_atom_indices_next, derived_sub_indices_next = self.limit_action_space(derived_states_next, derived_atom_indices_next, derived_sub_indices_next)
                if len(derived_states_next) == 1 and derived_states_next[0][0].predicate == 'End':
                    derived_states_next, derived_atom_indices_next, derived_sub_indices_next = self.end_in_false(derived_atom_indices.size(), derived_sub_indices.size())
        if all(len(elem)==0 for elem in derived_states_next):
            derived_states_next, derived_atom_indices_next, derived_sub_indices_next = self.end_in_false(derived_atom_indices.size(), derived_sub_indices.size())
        self.update_action_space(derived_states_next)

        self.current_depth += 1
        self.exceeded_max_depth = (self.current_depth >= self.max_depth)
        done_next = done_next | self.exceeded_max_depth
        
        # Track episode info for SB3
        info = {}
        
        # Handle episode completion
        if done_next:
            # Add episode info for Stable Baselines
            reward_value = float(reward_next.item())
            episode_length = int(self.current_depth)
            info["episode"] = {
                "r": reward_value,
                "l": episode_length
            }
            
            # Restore facts in knowledge base if this was a positive query
            if self.current_label == 1 and self.current_query in self.facts:
                if self.dynamic_consult:
                    # Re-add the current query to Janus knowledge base
                    janus.query_once(f"asserta({str(self.current_query)}).")
            # Clear current query reference and label
            self.current_query, self.current_label = None, None

        tensordict = TensorDict(
            {
                "atom_index": next_atom_index.unsqueeze(0),
                "sub_index": next_sub_index.unsqueeze(0),
                "state": NonTensorData(data=next_state),
                "label": self.tensordict['label'],
                "done": done_next,
                "reward": reward_next,
                "derived_states": NonTensorData(data=derived_states_next),
                "derived_atom_indices": derived_atom_indices_next,
                "derived_sub_indices": derived_sub_indices_next,
            },
            )

        self.tensordict = tensordict
        self.tensordict["action"] = torch.tensor(action, device=self.device)

        sub_index = self.tensordict['sub_index'].numpy()
        atom_index = self.tensordict['atom_index'].numpy()
        derived_atom_indices = self.tensordict['derived_atom_indices'].numpy()
        derived_sub_indices = self.tensordict['derived_sub_indices'].numpy()
        obs = {'sub_index':sub_index, 
               'atom_index':atom_index, 
               'derived_atom_indices':derived_atom_indices, 
               'derived_sub_indices':derived_sub_indices}

        reward = self.tensordict['reward'].numpy()
        done = self.tensordict['done'].numpy()
        truncated = bool(self.exceeded_max_depth)
        if self.verbose:
            print_state_transition(self.tensordict['state'], self.tensordict['derived_states'],self.tensordict['reward'], self.tensordict['done'], action=self.tensordict['action'],truncated=truncated)
            # print('idx derived sub:', list(self.tensordict['derived_sub_indices'][:3,0,:].numpy()),'\n')
        return obs, reward, done, truncated, info


    
    def get_next_states(self,state: List[Term]) -> Tuple[List[List[Term]], torch.Tensor, torch.Tensor]:
        """Get next possible states and their indices for all states in the batch"""

        if self.end_proof_action: # filter the end of the proof action
            assert len(state) > 0, f"State is empty"
            if len(state) > 1:
                state = [atom for atom in state if atom.predicate != 'End']
            else:
                if state[0].predicate == 'End':
                    state = [Term(predicate='False', args=[])]

        if self.engine == 'prolog':
            possible_states_next = get_next_unification_prolog(state, verbose=0)
        elif self.engine == 'python':
            possible_states_next = get_next_unification_python(state,
                                                        # pass the self.facts - the self.current_query
                                                        facts=[fact for fact in self.facts if fact != self.current_query] if self.current_label == 1 else self.facts,
                                                        rules=self.index_manager.rules,
                                                        verbose=0
                                                        )

        if self.end_proof_action:
            append = False
            for next_state in possible_states_next:
                if any([atom.predicate != 'True' and atom.predicate != 'False' for atom in next_state]):
                    append = True
                    break
            if append:
                possible_states_next.append([Term(predicate='End', args=[])])

        possible_atom_indices = []
        possible_sub_indices = []

        for s in possible_states_next:
            atom_idx, sub_idx = self.index_manager.get_atom_sub_index(s)
            possible_atom_indices.append(atom_idx)
            possible_sub_indices.append(sub_idx)

        possible_atom_indices = torch.stack(possible_atom_indices)
        possible_sub_indices = torch.stack(possible_sub_indices)

        # Do padding_states with 0s
        if len(possible_atom_indices) > self.padding_states:
            raise ValueError(f"Padding_states is too small. number of next states: {len(possible_atom_indices)}, padding_states: {self.padding_states}")
        
        possible_atom_indices = torch.cat([possible_atom_indices, torch.zeros(self.padding_states - len(possible_atom_indices), self.padding_atoms, device=self.device, dtype=torch.int64)])
        possible_sub_indices = torch.cat([possible_sub_indices, torch.zeros(self.padding_states - len(possible_sub_indices), self.padding_atoms, self.max_arity+1, device=self.device, dtype=torch.int64)])
        return possible_states_next, possible_atom_indices, possible_sub_indices
    
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


    def end_in_false(self, atom_indices_shape: torch.Size, sub_indices_shape: torch.Size) -> Tuple[List[List[Term]], torch.Tensor, torch.Tensor]:
        " Return a state that ends in False"
        false_state = Term(predicate="False", args=[])

        derived_states_next = [[false_state]]
        derived_atom_indices_next = torch.zeros(atom_indices_shape, device=self.device, dtype=torch.int64)
        derived_sub_indices_next = torch.zeros(sub_indices_shape, device=self.device, dtype=torch.int64)

        if false_state not in self.index_manager.atom_to_index:
            self.index_manager.atom_to_index[false_state] = self.index_manager.next_atom_index
            self.index_manager.next_atom_index += 1

        false_sub_id = torch.tensor([self.index_manager.predicate_no + self.predicate_false_offset, 0, 0], device=self.device, dtype=torch.int64)
        derived_atom_indices_next[0, 0] = self.index_manager.atom_to_index[false_state]
        derived_sub_indices_next[0, 0] = false_sub_id
        return derived_states_next, derived_atom_indices_next, derived_sub_indices_next


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
        state = [Term(predicate=n[1].strip(), args=[n[0].strip(), n[2].strip()]) for n in negative_batch_str] # convert each negative back to Term format
        return state


    def limit_action_space(self, derived_states: List[List[Term]], derived_atom_indices: torch.Tensor, derived_sub_indices: torch.Tensor) -> Tuple[List[List[Term]], torch.Tensor, torch.Tensor]:
        """Limit the action space by removing states that have been visited to prevent loops.
        
        Args:
            derived_states: List of possible next states
            derived_atom_indices: Tensor of atom indices for next states
            derived_sub_indices: Tensor of sub-indices for next states
            
        Returns:
            Tuple containing:
            - filtered_states: List of states after removing visited ones
            - filtered_atom_indices: Tensor of atom indices after filtering
            - filtered_sub_indices: Tensor of sub-indices after filtering
        """
        # Create a mask identifying states that have not been visited
        unvisited_mask = [",".join(str(s) for s in state) not in self.memory for state in derived_states]
        
        # Filter the derived states list using the mask
        filtered_states = [state for state, is_unvisited in zip(derived_states, unvisited_mask) if is_unvisited]
        
        # Convert the mask to a tensor and pad it to match derived_atom_indices size
        unvisited_mask_tensor = torch.tensor(unvisited_mask, device=self.device)
        padded_mask = torch.cat([unvisited_mask_tensor, 
                               torch.zeros(derived_atom_indices.size(0) - unvisited_mask_tensor.size(0), 
                               device=self.device)])
        
        # Apply the mask to atom indices
        broadcast_mask_2d = padded_mask.unsqueeze(-1)
        filtered_atom_indices = derived_atom_indices * broadcast_mask_2d
        
        # Filter out rows that are all zeros and keep track of valid indices
        valid_atom_rows = torch.any(filtered_atom_indices != 0, dim=-1)
        valid_atom_indices = filtered_atom_indices[valid_atom_rows]
        
        # Pad the filtered atom indices back to original size
        filtered_atom_indices = torch.cat([
            valid_atom_indices, 
            torch.zeros(derived_atom_indices.size(0) - valid_atom_indices.size(0), 
                       derived_atom_indices.size(1), 
                       device=self.device, 
                       dtype=torch.int64)
        ])
        
        # Apply the mask to sub indices
        broadcast_mask_3d = broadcast_mask_2d.unsqueeze(-1)
        filtered_sub_indices = derived_sub_indices * broadcast_mask_3d
        
        # Filter out rows that are all zeros and keep track of valid indices
        valid_sub_rows = torch.any(filtered_sub_indices != 0, dim=(-1, -2))
        valid_sub_indices = filtered_sub_indices[valid_sub_rows]
        
        # Pad the filtered sub indices back to original size
        filtered_sub_indices = torch.cat([
            valid_sub_indices, 
            torch.zeros(derived_sub_indices.size(0) - valid_sub_indices.size(0), 
                       derived_sub_indices.size(1), 
                       derived_sub_indices.size(2), 
                       device=self.device, 
                       dtype=torch.int64)
        ])
        
        return filtered_states, filtered_atom_indices, filtered_sub_indices
    
        # """Limit the action space by removing the states that have been visited""" # RODRIGO: Could you rename var names to make it more readable?
        # mask = [",".join(str(s) for s in state) not in self.memory for state in derived_states]
        # cutted_derived_states = [state for state, m in zip(derived_states, mask) if m]
        # mask = torch.tensor(mask, device=self.device)
        # mask_broadcasted = torch.cat([mask, torch.zeros(derived_atom_indices.size(0)-mask.size(0))])
        # mask_broadcasted = mask_broadcasted.unsqueeze(-1)
        # cutted_derived_atom_indices = derived_atom_indices * mask_broadcasted
        # valid_id = torch.any(cutted_derived_atom_indices!=0, dim=-1)
        # valid = cutted_derived_atom_indices[valid_id]
        # cutted_derived_atom_indices = torch.cat([valid, torch.zeros(derived_atom_indices.size(0)-valid.size(0), derived_atom_indices.size(1), device=self.device, dtype=torch.int64)])
        # mask_broadcasted = mask_broadcasted.unsqueeze(-1)
        # cutted_derived_sub_indices = derived_sub_indices * mask_broadcasted
        # valid_id = torch.any(cutted_derived_sub_indices!=0, dim=(-1, -2))
        # valid = cutted_derived_sub_indices[valid_id]
        # cutted_derived_sub_indices = torch.cat([valid, torch.zeros(derived_sub_indices.size(0)-valid.size(0), derived_sub_indices.size(1), derived_sub_indices.size(2), device=self.device, dtype=torch.int64)])
        # return cutted_derived_states, cutted_derived_atom_indices, cutted_derived_sub_indices