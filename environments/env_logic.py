from typing import List, Optional, Tuple, Dict, Union
import random
from math import prod
from utils import Term, Rule, is_variable, extract_var, get_rules_from_file
from unification.prolog_unification import get_next_state_prolog

import torch
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec, DiscreteTensorSpec, NonTensorSpec
from torchrl.envs import EnvBase
from tensordict import TensorDict, TensorDictBase, NonTensorData
import re
import janus_swi as janus

# janus.consult("./data/ancestor.pl")
# janus.consult("./data/countries_s1_train.pl")


class BatchedDiscreteTensorSpec(DiscreteTensorSpec):
    '''Batched Discrete Tensor Spec'''
    def __init__(self, max_actions: Union[int, torch.Tensor], shape: torch.Size, device: str = "cpu"):
        super().__init__(n=max_actions,shape=shape, device=device)
        if isinstance(max_actions, int):
            max_actions = torch.tensor([max_actions], device=device)
        self.max_actions = max_actions

    def sample(self):
        # Generate a random action for each element in the batch
        actions = torch.stack([
            torch.randint(0, max_actions, (1,), device=self.device)
            for max_actions in self.max_actions
        ])
        return actions.squeeze()

    def rand(self):
        # Consistent with sample() 
        return self.sample()




class BatchLogicProofEnv(EnvBase):
    batch_locked = False  # Allow dynamic batch sizes
    
    def __init__(self, batch_size=None, knowledge_f=None, test_f=None, seed=None, max_arity=1, constant_str2idx=None, predicate_str2idx=None, constant_no=0, predicate_no=0, variable_no=0, device="cpu"):
        '''Initialize the environment'''
        super().__init__(device=device)
        self.batch_size = torch.Size([batch_size])
        self.max_arity = max_arity # Maximum arity of the predicates
        self.max_atom = 10  # Maximum number of atoms in a state
        self._make_spec(batch_size=self.batch_size)
        
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

        # self._has_dynamic_specs = True # Allow dynamic specs
        self.padding = 15 # Maximum number of possible next states
        self.max_depth = 10 # Maximum depth of the proof tree
        self.atom_to_index = {} # Map atom to index
        self.atom_id_to_sub_id = {} # Map atom index to sub-indices of predicates and arguments
        self.predicate_str2idx = predicate_str2idx # Global index
        self.constant_str2idx = constant_str2idx # Global index
        self.variable_str2idx = {} # Map variable to index
        self.next_atom_index = 1  # Next available index. 0 is reserved for padding
        self.constant_no = constant_no # Number of constants
        self.predicate_no = predicate_no # Number of predicates
        self.next_var_index = constant_no+1 # Next available index. 0 is reserved for padding
        self.variable_no = variable_no # Max number of variables

        self.knowledge_f = knowledge_f
        self.test_f = test_f


    def _set_seed(self, seed: Optional[int]):
        '''Set the seed for the environment'''
        rng = torch.manual_seed(seed)
        self.rng = rng

    @staticmethod
    def gen_params(batch_size=None) -> TensorDictBase:
        '''Generate a initial state of the environment'''
        if batch_size is None:
            batch_shape = torch.Size([])
        elif isinstance(batch_size, int):
            batch_shape = torch.Size([batch_size])
        else:
            batch_shape = torch.Size(batch_size)

        td = TensorDict(
            {
                "done": torch.zeros((*batch_shape, 1), dtype=torch.bool),
                "reward": torch.zeros((*batch_shape, 1), dtype=torch.float32) * (-1),
            },
            batch_size=[],
        )

        if batch_size:
            td = td.expand(batch_size).contiguous()
        return td

    def reset_atom_var(self):
        '''Reset the atom and variable dicts and indices'''
        self.atom_to_index = {}
        self.atom_id_to_sub_id = {}
        self.variable_str2idx = {}
        self.next_atom_index = 1
        self.next_var_index = self.constant_no+1


    def _make_spec(self, batch_size=None):
        '''Create the observation and action specs'''
        if batch_size is None:
            batch_size = torch.Size([])
            
        self.action_spec = BatchedDiscreteTensorSpec(
            max_actions=100,
            shape=batch_size,
            device=self.device
        )

        self.atom_index_spec = DiscreteTensorSpec(
            n=1000000,
            shape=batch_size+torch.Size([self.max_atom]),
            device=self.device
        )

        self.sub_index_spec = DiscreteTensorSpec(
            n=1000000,
            shape=batch_size+torch.Size([self.max_atom])+torch.Size([self.max_arity+1]), # +1 for the predicate
            device=self.device
        )

        self.reward_spec = UnboundedContinuousTensorSpec(shape=batch_size,device=self.device)
        self.done_spec = BoundedTensorSpec(shape=batch_size, minimum=0, maximum=1, dtype=torch.bool, device=self.device)
        self.possible_states_next_spec = NonTensorSpec(shape=batch_size)

        self.observation_spec = CompositeSpec(
            atom_index=self.atom_index_spec,
            sub_index = self.sub_index_spec,
            reward=self.reward_spec,
            done=self.done_spec,
            derived_states=self.possible_states_next_spec,
            derived_atom_indices=self.atom_index_spec,
            derived_sub_indices=self.sub_index_spec,
            shape=batch_size,
            device=self.device,
            state=NonTensorSpec(shape=batch_size),
        )

    def _reset(self, tensordict: Optional[TensorDictBase] = None) -> TensorDictBase:
        '''Reset the environment to the initial state'''
        # print('Resetting...\n')
        if tensordict is None or tensordict.is_empty():
            tensordict = self.gen_params(batch_size=self.batch_size)

        batch_shape = tensordict.shape
        self.current_depth = torch.zeros(batch_shape, dtype=torch.long, device=self.device)

        states = []
        atom_indices = []
        sub_indices = []
        for i in range(prod(batch_shape) if batch_shape else 1):
            seed = torch.randint(0, 1000, (1,), generator=self.rng).item()
            if self.test_f:
                state = [self.get_test_query(seed, self.test_f)]
            else:
                state = [self.get_random_query(seed, self.knowledge_f)]
            states.append(state)
            atom_index, sub_index = self.get_atom_sub_index(i, state)
            atom_indices.append(atom_index)
            sub_indices.append(sub_index)
        atom_indices = torch.stack(atom_indices)
        sub_indices = torch.stack(sub_indices)
        # print('states',states)
        # print('indices',indices)
        # print('Initial state:',[[str(atom) for atom in state] for state in states])
        # Get next possible states and their indices

        derived_states, derived_atom_indices, derived_sub_indices = self.get_next_states_batch(states)
        self.update_action_space(derived_states)

        out = TensorDict(
            {
                "atom_index": atom_indices,
                "sub_index": sub_indices,
                "state": NonTensorData(data=states, batch_size=batch_shape),
                "done": torch.zeros(batch_shape, dtype=torch.bool, device=self.device),
                "reward": torch.zeros(batch_shape, dtype=torch.float32, device=self.device)*(-1),
                "derived_states": NonTensorData(data=derived_states, batch_size=batch_shape),
                "derived_atom_indices": derived_atom_indices,
                "derived_sub_indices": derived_sub_indices,
            },
            batch_size=batch_shape,
        )
        # print('\nReset done:')
        # print_td(out)
        return out

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        '''
        Given the current state, possible next states, an action, and return the next state.
        (It should be: given the current state, and an action, return the next state, but we need to modify it for our case)
        '''
        # print('Stepping...\n')       
        actions = tensordict["action"]
        derived_atom_indices = tensordict["derived_atom_indices"]
        derived_states = tensordict["derived_states"]
        derived_sub_indices = tensordict["derived_sub_indices"]

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

        done_next, rewards_next = self.get_done_reward(states_next)
        # Get next possible states for the new states, as well as rewards and done
        derived_states_next, derived_atom_indices_next, derived_sub_indices_next = self.get_next_states_batch(states_next)
        self.update_action_space(derived_states_next)

        
        self.current_depth += 1
        exceeded_max_depth = (self.current_depth >= self.max_depth)
        done_next = done_next | exceeded_max_depth

        next = TensorDict(
            {
                "atom_index": atom_indices_next,
                "sub_index": sub_indices_next,
                "state": NonTensorData(data=states_next, batch_size=tensordict.shape),
                "done": done_next,
                "reward": rewards_next,
                "derived_states": NonTensorData(data=derived_states_next, batch_size=tensordict.shape),
                "derived_atom_indices": derived_atom_indices_next,
                "derived_sub_indices": derived_sub_indices_next,
            },
            batch_size=tensordict.shape,
            )

        # print('\nStep done:')
        # print_td(next)
        return next


    
    def get_next_states_batch(self,states: List[List[Term]]) -> Tuple[List[List[List[Term]]], torch.Tensor, torch.Tensor]:
        """Get next possible states and their indices for all states in the batch"""
        possible_states_next_batch = []
        possible_atom_indices_next_batch = []
        possible_sub_indices_next_batch = []
        for i, state in enumerate(states):
            # print('    State:',[str(atom) for atom in state])
            possible_states_next = get_next_state_prolog(state)

            # Store states and get their indices
            possible_states_next_batch.append(possible_states_next)
            
            # Get indices for all possible next states
            possible_atom_indices = []
            possible_sub_indices = []
            for s in possible_states_next:
                atom_idx, sub_idx = self.get_atom_sub_index(i, s)
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
    
    def get_done_reward(self,states: List[List[Term]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get done and reward keys for all states in the batch. To be called in the step() method"""

        batch_size = len(states)

        any_atom_false = [any([atom.predicate == 'False' for atom in state]) for state in states]
        all_atoms_true = [all([atom.predicate == 'True' for atom in state]) for state in states]

        done_batch = [any_atom_false[i] or all_atoms_true[i] for i in range(batch_size)]
        successful_batch = [all_atoms_true[i] for i in range(batch_size)]

        done_batch = torch.tensor(done_batch, device=self.device)
        successful_batch = torch.tensor(successful_batch, device=self.device)
        rewards_batch = torch.where(
            done_batch & successful_batch, 
            torch.ones(batch_size, device=self.device), 
            torch.zeros(batch_size, device=self.device)
        )

        return done_batch, rewards_batch,

    def update_action_space(self, derived_states: List[List[List[Term]]]):
        '''Update the action space based on the possible next states 
        To be called every time the possible next states are updated'''
        # max actions is the number of possible next states in each batch
        max_actions = [len(derived_state) for derived_state in derived_states]
        max_actions =  torch.tensor(max_actions, device=self.device)
        
        self.action_spec = BatchedDiscreteTensorSpec(
            max_actions=max_actions,
            shape=(self.batch_size),
            # shape=(),
            device=self.device
        )

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

    def get_random_query(self, seed: int = 0, knowledge_f: str = None) -> Term:
        """Generate a random query"""
        rules = get_rules_from_file(knowledge_f)
        predicates = set()
        constants = set()
        for rule in rules:
            # proof_first not related to query generation
            if not rule.head.predicate == "proof_first":
                predicates.add((rule.head.predicate, len(rule.head.args)))
                constants.update([arg for arg in rule.head.args if not is_variable(arg)])
        random.seed(seed)
        predicate_random_choice = random.choice(list(predicates))
        predicate, arity = predicate_random_choice[0], predicate_random_choice[1]
        constants_list = list(constants)
        constants = random.sample(constants_list, arity)
        
        return Term(predicate, constants) 

    def get_test_query(self, seed: int = 0, test_f: str = None) -> Term:
        '''Get a query from the test file'''
        queries = get_rules_from_file(test_f)
        random.seed(seed)
        query = random.choice(queries)
        return query.head

help(DiscreteTensorSpec)