from typing import List, Optional, Tuple, Dict, Union
import random
from math import prod
from utils import Term, is_variable, extract_var, print_state_transition
from unification.prolog_unification import get_next_state_prolog

import torch
from tensordict import TensorDict, TensorDictBase, NonTensorData

import torch
from tensordict import TensorDict, TensorDictBase, NonTensorData
from typing import Optional
import gymnasium as gym
import numpy as np

from dataset import Rule
import janus_swi as janus
from dataset import DataHandler_corruptions

class IndexManager():

    def __init__(self, constants: set, 
                predicates: set,
                constant_no: int,
                predicate_no: int,
                variable_no: int,
                max_atom: int = 10,
                max_arity: int = 2,
                device: torch.device = torch.device("cpu")):
        
        self.device = device
        self.constants = constants
        self.predicates = predicates
        self.constant_no = constant_no
        self.variable_no = variable_no
        self.predicate_no = predicate_no
        self.max_atom = max_atom  # Maximum number of atoms in a state
        self.max_arity = max_arity # Maximum arity of the predicates

        # LOCAL INDEXES
        self.atom_to_index = {} # Map atom to index
        self.atom_id_to_sub_id = {} # Map atom index to sub-indices of predicates and arguments
        self.variable_str2idx = {} # Map variable to index
        self.next_atom_index = 1  # Next available index. 0 is reserved for padding
        self.next_var_index = constant_no+1 # Next available index. 0 is reserved for padding

        self.create_global_idx()

    def create_global_idx(self)-> Tuple[dict, dict]:
        '''Create a global index for a list of terms. Start idx counting from 1'''
        self.constant_str2idx = {term: i + 1 for i, term in enumerate(self.constants)}
        self.predicate_str2idx = {term: i + 1 for i, term in enumerate(self.predicates)}

    def reset_atom_var(self):
        '''Reset the atom and variable dicts and indices'''
        self.atom_to_index = {}
        self.atom_id_to_sub_id = {}
        self.variable_str2idx = {}
        self.next_atom_index = 1
        self.next_var_index = self.constant_no+1

    def get_atom_sub_index(self, batch_index:int, state: List[Term]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the atom and sub index for a state"""
        # Get variables
        full_state = ",".join(str(s) for s in state)
        vars = extract_var(full_state)
        for var in vars:
            if (var != "True") and (var != "False") and (var+str(batch_index) not in self.variable_str2idx):
                if self.next_var_index > self.constant_no + self.variable_no:
                    raise ValueError(f"Exceeded the maximum number of variables: {self.variable_no}")
                else:
                    index = self.next_var_index
                    self.variable_str2idx[var+str(batch_index)] = index
                    self.next_var_index += 1

        # Get atom_index and sub_index
        atom_index =torch.zeros(self.max_atom, device=self.device, dtype=torch.int64)
        sub_index = torch.zeros(self.max_atom, self.max_arity+1, device=self.device, dtype=torch.int64)
        for i, atom in enumerate(state):
            if atom not in self.atom_to_index:
                self.atom_to_index[atom] = self.next_atom_index
                atom_index[i] = self.next_atom_index
                self.next_atom_index += 1
            else:
                atom_index[i] = self.atom_to_index[atom]
            atom_id = atom_index[i].item()
            if atom_id not in self.atom_id_to_sub_id:
                try:
                    if atom.predicate == 'True':
                        sub_index[i, 0] = self.predicate_no + 1
                    elif atom.predicate == 'False':
                        sub_index[i, 0] = self.predicate_no + 2
                    else:
                        sub_index[i, 0] = self.predicate_str2idx[atom.predicate]
                    for j, arg in enumerate(atom.args):
                        if is_variable(arg):
                            sub_index[i, j+1] = self.variable_str2idx[arg+str(batch_index)]
                        else:
                            sub_index[i, j+1] = self.constant_str2idx[arg]
                except Exception as e:
                    print("The following key is not in dict:", e)
                self.atom_id_to_sub_id[atom_id] = sub_index[i]
            else:
                sub_index[i] = self.atom_id_to_sub_id[atom_id]

        return atom_index, sub_index
    






class LogicEnv_gym(gym.Env):
    batch_locked = False  # Allow dynamic batch sizes
    
    def __init__(self, 
                max_depth: int = 10,
                seed: Optional[int] = None,
                device: torch.device = torch.device("cpu"),
                index_manager: Optional[IndexManager] = None,
                data_handler: Optional[DataHandler_corruptions] = None,
                eval=False
                ):
        
        '''Initialize the environment'''
        super().__init__()
        self.device = device

        self.max_arity=data_handler.max_arity # Maximum arity of the predicates
        self.max_atom = 10  # Maximum number of atoms in a state
        self.padding = 15 # Maximum number of possible next states
        self.max_depth = max_depth # Maximum depth of the proof tree
        self.index_manager = index_manager

        seed = seed if seed is not None else torch.empty((), dtype=torch.int64).random_().item()
        self._set_seed(seed)
        self._make_spec()

        self.facts=data_handler.facts,        

        self.train_queries, self.train_labels = data_handler.train_queries, data_handler.train_labels
        self.valid_queries, self.valid_labels = data_handler.valid_queries, data_handler.valid_labels
        self.test_queries, self.test_labels = data_handler.test_queries, data_handler.test_labels
        
        self.janus_file=data_handler.janus_path

        self.eval = eval

        self.eval_idx = 0 # Index to go through all the eval queries
        self.eval_len = len(self.valid_queries) # Number of eval queries (to reset the index)

    def _set_seed(self, seed:int):
        '''Set the seed for the environment'''
        rng = torch.manual_seed(seed)
        self.rng = rng


    def _make_spec(self):
        '''Create the observation and action specs'''
        obs_spaces = {
            'sub_index': gym.spaces.Box(
                low=float('-inf'),
                high=float('inf'),
                shape=torch.Size([self.max_atom])+torch.Size([self.max_arity+1]),
                dtype=np.int64,
            ),
            'atom_index': gym.spaces.Box(
                low=float('-inf'),
                high=float('inf'),
                shape=torch.Size([self.max_atom]),
                dtype=np.int64,
            ),
            'derived_atom_indices': gym.spaces.Box(
                low=float('-inf'),
                high=float('inf'),
                shape=torch.Size([self.padding])+torch.Size([self.max_atom]),
                dtype=np.int64,
            ),
            'derived_sub_indices': gym.spaces.Box(
                low=float('-inf'),
                high=float('inf'),
                shape=torch.Size([self.padding])+torch.Size([self.max_atom])+torch.Size([self.max_arity+1]),
                dtype=np.int64,
            ),
            
        }
        self.observation_space = gym.spaces.Dict(obs_spaces)
        self.action_space = gym.spaces.Discrete(100)

    def update_action_space(self, derived_states: List[List[List[Term]]]):
        '''Update the action space based on the possible next states 
        To be called every time the possible next states are updated'''
        # max actions is the number of possible next states in each batch
        max_actions = [len(derived_state) for derived_state in derived_states]
        max_actions =  torch.tensor(max_actions, device=self.device)
        self.action_space = gym.spaces.Discrete(max_actions.max().item())


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

        # self.current_query = query_str
        # print('removing fact:', self.current_query)
        # janus.query(f"retract(({query_str})).")

        # 1. load the original facts as they are, and skip the line with the query
        facts = []
        query_found = False
        with open(self.janus_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                if query_str not in line:
                    facts.append(line)
                else:
                    query_found = True

        # if query not found, either it is an error or it is a val query, so we can skip it
        if not query_found: 
            # raise ValueError(f"Query {query_str} not found in {self.janus_file}")
            return None
        
        # 2. save a _tmp file with the new facts
        tmp_file = self.janus_file.replace('.pl', '_tmp.pl')
        with open(tmp_file, "w") as f:
            for line in facts:
                f.write(line)
        # 3. consult janus with the new file
        janus.query("retractall((:- _)).") # Removes any clauses (facts and rules).
        janus.query("retractall(_).") # Removes all facts.
        janus.consult(tmp_file)

    def reset(self, seed: Optional[int]= None, options=None) -> TensorDictBase:
        if self.eval:
            # state, label = self.get_random_queries(self.valid_queries, self.valid_labels, 1, seed=seed)
            if self.eval_idx == self.eval_len:
                self.eval_idx = 0
            state, label = [self.valid_queries[self.eval_idx]], [self.valid_labels[self.eval_idx]]
            self.eval_idx += 1
        else:
            state, label = self.get_random_queries(self.train_queries, self.train_labels, 1, seed=seed)
            if label[0] == 1:
                self.new_consult_janus(state[0])

        return self._reset(state, label)
    
    def reset_from_query(self, query, label) -> TensorDictBase:

        return self._reset([query], [label]) 

    def _reset(self, query, label) -> TensorDictBase:
        '''Reset the environment to the initial state'''    
        self.current_depth = torch.zeros(1, dtype=torch.long, device=self.device)

        states = []
        atom_indices = []
        sub_indices = []
        for i in range(1):
            state, label = query, label
            states.append(state)
            atom_index, sub_index = self.index_manager.get_atom_sub_index(i, state)
            atom_indices.append(atom_index)
            sub_indices.append(sub_index)
        atom_indices = torch.stack(atom_indices)
        sub_indices = torch.stack(sub_indices)

        derived_states, derived_atom_indices, derived_sub_indices = self.get_next_states_batch(states)
        self.update_action_space(derived_states)

        self.tensordict = TensorDict(
            {
                "atom_index": atom_indices,
                "sub_index": sub_indices,
                "state": NonTensorData(data=states),
                "label": torch.tensor(label[0], device=self.device),
                "done": torch.zeros(1, dtype=torch.bool, device=self.device),
                "reward": torch.zeros(1, dtype=torch.float32, device=self.device)*(-1),
                "derived_states": NonTensorData(data=derived_states),
                "derived_atom_indices": derived_atom_indices,
                "derived_sub_indices": derived_sub_indices,
            },
        )

        sub_index = self.tensordict['sub_index'].cpu().numpy()
        atom_index = self.tensordict['atom_index'].cpu().numpy()
        derived_atom_indices = self.tensordict['derived_atom_indices'].cpu().numpy()
        derived_sub_indices = self.tensordict['derived_sub_indices'].cpu().numpy()
        obs = {'sub_index':sub_index, 
               'atom_index':atom_index, 
               'derived_atom_indices':derived_atom_indices, 
               'derived_sub_indices':derived_sub_indices}
        # print('reset...')
        # print('obs', [(key, value.shape) for key, value in obs.items()])
        # print_state_transition(self.tensordict['state'], self.tensordict['derived_states'],self.tensordict['reward'], self.tensordict['done'])
        return obs, {}

    def step(self, action) -> TensorDictBase:
        '''
        Given the current state, possible next states, an action, and return the next state.
        (It should be: given the current state, and an action, return the next state, but we need to modify it for our case)
        '''
        # print('\nStep')      
        # print('action in step:', action)
        # distr = action[0,1:]
        # action = int(action[0,0])
        # print('final action in step:', action)
        self.tensordict['label'] = 1   # !!!!!!!!!!!!!!!!DELETEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE

        action = np.array([action])
        action = torch.tensor(action, device=self.device)  
        actions = action
        derived_atom_indices = self.tensordict["derived_atom_indices"]
        derived_states = self.tensordict["derived_states"]
        derived_sub_indices = self.tensordict["derived_sub_indices"]

        # Given the actions, use the stored derived_states to get the state_next and derived_states_next
        states_next = []
        atom_indices_next = []
        sub_indices_next = []
        for i, (derived_state, action) in enumerate(zip(derived_states, actions.view(-1))):
            if action >= len(derived_state):
                raise ValueError(f"State {i} of the batch. Invalid action ({action}). Max action: {len(derived_states[i])}, derived_states: {derived_states[i]}")
            next_state = derived_state[action.item()]
            next_atom_index = derived_atom_indices[i][action.item()]
            next_sub_index = derived_sub_indices[i][action.item()]
            states_next.append(next_state)
            atom_indices_next.append(next_atom_index)
            sub_indices_next.append(next_sub_index)
        atom_indices_next = torch.stack(atom_indices_next)
        sub_indices_next = torch.stack(sub_indices_next)

        done_next, rewards_next = self.get_done_reward(states_next,self.tensordict['label'].item())
        # Get next possible states for the new states, as well as rewards and done
        derived_states_next, derived_atom_indices_next, derived_sub_indices_next = self.get_next_states_batch(states_next)
        self.update_action_space(derived_states_next)

        self.current_depth += 1
        self.exceeded_max_depth = (self.current_depth >= self.max_depth)
        # print('exceeded_max_depth: current depth, max depth', self.current_depth,self.max_depth  ) if self.exceeded_max_depth else None
        done_next = done_next | self.exceeded_max_depth

        if done_next:
            self.tensordict['label'] = None # Reset the label to None, to be sure that it is not used in the next step

        tensordict = TensorDict(
            {
                "atom_index": atom_indices_next,
                "sub_index": sub_indices_next,
                "state": NonTensorData(data=states_next),
                "label": self.tensordict['label'],
                "label": self.tensordict['label'],
                "done": done_next,
                "reward": rewards_next,
                "derived_states": NonTensorData(data=derived_states_next),
                "derived_atom_indices": derived_atom_indices_next,
                "derived_sub_indices": derived_sub_indices_next,
            },
            )

        self.tensordict = tensordict
        self.tensordict["action"] = actions


        sub_index = self.tensordict['sub_index'].cpu().numpy()
        atom_index = self.tensordict['atom_index'].cpu().numpy()
        derived_atom_indices = self.tensordict['derived_atom_indices'].cpu().numpy()
        derived_sub_indices = self.tensordict['derived_sub_indices'].cpu().numpy()
        obs = {'sub_index':sub_index, 
               'atom_index':atom_index, 
               'derived_atom_indices':derived_atom_indices, 
               'derived_sub_indices':derived_sub_indices}


        rewards = self.tensordict['reward'].cpu().numpy()
        dones = self.tensordict['done'].cpu().numpy()
        # for rewards, get an float value, for dones, get a boolean value
        if rewards.shape[0] != 1 or dones.shape[0] != 1 or self.exceeded_max_depth.shape[0] != 1:
            not_valid_shape = f"Invalid shape. rewards: {rewards.shape}, dones: {dones.shape}, exceeded_max_depth: {self.exceeded_max_depth.shape}"
            raise ValueError(not_valid_shape)
        rewards = rewards[0]
        dones = dones[0]
        truncated = bool(self.exceeded_max_depth[0])
        # if dones and not self.eval:
        #     print('     adding fact:', self.current_query, dones)
        #     janus.query(f"assertz({self.current_query}).")
        # print_state_transition(self.tensordict['state'], self.tensordict['derived_states'],self.tensordict['reward'], self.tensordict['done'], action=self.tensordict['action'],truncated=truncated)
        return obs, rewards, dones, truncated, {}


    
    def get_next_states_batch(self,states: List[List[Term]]) -> Tuple[List[List[List[Term]]], torch.Tensor, torch.Tensor]:
        """Get next possible states and their indices for all states in the batch"""
        possible_states_next_batch = []
        possible_atom_indices_next_batch = []
        possible_sub_indices_next_batch = []
        for i, state in enumerate(states):
            possible_states_next = get_next_state_prolog(state)

            # Store states and get their indices
            possible_states_next_batch.append(possible_states_next)
            
            # Get indices for all possible next states
            possible_atom_indices = []
            possible_sub_indices = []
            for s in possible_states_next:
                atom_idx, sub_idx = self.index_manager.get_atom_sub_index(i, s)
                possible_atom_indices.append(atom_idx)
                possible_sub_indices.append(sub_idx)
            possible_atom_indices_next_batch.append(torch.stack(possible_atom_indices))
            possible_sub_indices_next_batch.append(torch.stack(possible_sub_indices))

        # Do padding to possible_indices_next_batch, possible_sub_indices_next_batch with 0s
        max_len = max(len(indices) for indices in possible_atom_indices_next_batch)
        if max_len > self.padding:
            raise ValueError(f"Padding is too small. Max length of indices: {max_len}")
        possible_atom_indices_next_batch = [torch.cat([indices, torch.zeros(self.padding - len(indices), self.max_atom, device=self.device, dtype=torch.int64)]) for indices in possible_atom_indices_next_batch]
        possible_atom_indices_next_batch = torch.stack(possible_atom_indices_next_batch)
        possible_sub_indices_next_batch = [torch.cat([sub_indices, torch.zeros(self.padding - len(sub_indices), self.max_atom, self.max_arity+1, device=self.device, dtype=torch.int64)]) for sub_indices in possible_sub_indices_next_batch]
        possible_sub_indices_next_batch = torch.stack(possible_sub_indices_next_batch)

        return possible_states_next_batch, possible_atom_indices_next_batch, possible_sub_indices_next_batch
    
    def get_done_reward(self,states: List[List[Term]], label: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get done and reward keys for all states in the batch. To be called in the step() method"""
        assert label is not None, f"Label is None"

        batch_size = len(states)

        any_atom_false = [any([atom.predicate == 'False' for atom in state]) for state in states]
        all_atoms_true = [all([atom.predicate == 'True' for atom in state]) for state in states]

        done_batch = [any_atom_false[i] or all_atoms_true[i] for i in range(batch_size)]
        successful_batch = [all_atoms_true[i] for i in range(batch_size)]

        done_batch = torch.tensor(done_batch, device=self.device)
        successful_batch = torch.tensor(successful_batch, device=self.device)
        label_tensor = torch.tensor(label, device=self.device) # ATTENTION: THIS IS NOT INTENDED FOR THE BATCH MODE
        # rewards_batch = torch.where(
        #     done_batch & successful_batch, 
        #     torch.ones(batch_size, device=self.device), 
        #     torch.zeros(batch_size, device=self.device)
        # )
        
        rewards_batch = torch.where(
            (done_batch & successful_batch & (label_tensor == 1)),
            torch.ones(batch_size, device=self.device),
            torch.zeros(batch_size, device=self.device)
        )
        # if done_batch[0] and label == 0:
            # print('done, successful, reward', done_batch[0], successful_batch[0], rewards_batch[0])

        return done_batch, rewards_batch
    
    @staticmethod
    def get_random_queries(queries: List[Rule], labels: List[int], n: int, seed: Optional[int] = None) -> List[Term]:
        if not seed: # choose a random seed
            random_instance = random.Random()
        else:
            random_instance = random.Random(seed)
        assert n <= len(queries), f"Number of queries ({n}) is greater than the number of queries ({len(queries)})"
        sampled_indices = random_instance.sample(range(len(queries)), n)
        sampled_queries = [queries[i] for i in sampled_indices]
        sampled_labels = [labels[i] for i in sampled_indices]
        
        return sampled_queries, sampled_labels
    
  
