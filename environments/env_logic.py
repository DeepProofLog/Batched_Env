from typing import List, Optional, Tuple, Dict, Union
import random
from math import prod
from utils import Term, Rule, is_variable
from unification.prolog_unification import get_next_state_prolog

import torch
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec, DiscreteTensorSpec, NonTensorSpec
from torchrl.envs import EnvBase
from tensordict import TensorDict, TensorDictBase, NonTensorData
import re
import janus_swi as janus

# janus.consult("./data/ancestor.pl")
janus.consult("./data/countries_s1_train.pl")


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
    
    def __init__(self, batch_size=None, seed=None, device="cpu"):
        
        super().__init__(device=device)
        self.batch_size = torch.Size([batch_size])
        self._make_spec(batch_size=self.batch_size)
        
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

        # self._has_dynamic_specs = True # Allow dynamic specs
        self.state_to_index = {} # Map state to index
        self.next_index = 1  # Next available index. 0 is reserved for padding
        self.padding = 15 # Maximum number of possible next states
        self.max_depth = 10 # Maximum depth of the proof tree

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

    @staticmethod
    def gen_params(batch_size=None) -> TensorDictBase:
        '''Generate a initial state of the environment'''
        if batch_size is None:
            batch_shape = torch.Size([])
        else:
            batch_shape = torch.Size([batch_size])

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

    def _make_spec(self, batch_size=None):

        if batch_size is None:
            batch_size = torch.Size([])
            
        self.action_spec = BatchedDiscreteTensorSpec(
            max_actions=100,
            shape=batch_size,
            device=self.device
        )

        self.index_spec = DiscreteTensorSpec(
            n=1000000,
            shape=batch_size,
            device=self.device
        )

        self.reward_spec = UnboundedContinuousTensorSpec(shape=batch_size,device=self.device)
        self.done_spec = BoundedTensorSpec(shape=batch_size, minimum=0, maximum=1, dtype=torch.bool, device=self.device)
        self.possible_states_next_spec = NonTensorSpec(shape=batch_size)

        self.observation_spec = CompositeSpec(
            index=self.index_spec,
            reward=self.reward_spec,
            done=self.done_spec,
            derived_states=self.possible_states_next_spec,
            derived_indices=self.index_spec,
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
        indices = []
        for _ in range(prod(batch_shape) if batch_shape else 1):
            seed = torch.randint(0, 1000, (1,), generator=self.rng).item()
            state = [self.get_test_query(seed)]
            # state = [self.get_random_query(seed)]
            states.append(state)
            indices.append(self.get_state_index(str(state)))
        indices = torch.tensor(indices, device=self.device).view(batch_shape)
        # print('states',states)
        # print('indices',indices)
        # print('Initial state:',[[str(atom) for atom in state] for state in states])
        # Get next possible states and their indices
        derived_states, derived_indices = self.get_next_states_batch(states)
        self.update_action_space(derived_states)

        out = TensorDict(
            {
                "index": indices,
                "state": NonTensorData(data=states, batch_size=batch_shape),
                "done": torch.zeros(batch_shape, dtype=torch.bool, device=self.device),
                "reward": torch.zeros(batch_shape, dtype=torch.float32, device=self.device)*(-1),
                "derived_states": NonTensorData(data=derived_states, batch_size=batch_shape),
                "derived_indices": derived_indices,
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
        derived_indices = tensordict["derived_indices"]
        derived_states = tensordict["derived_states"]

        # Given the actions, use the stored derived_states to get the state_next and derived_states_next
        states_next = []
        indices_next = []        
        for i, (derived_state, action) in enumerate(zip(derived_states, actions.view(-1))):
            if action >= len(derived_state):
                raise ValueError(f"State {i} of the batch. Invalid action ({action}). Max action: {len(derived_states[i])}, derived_states: {derived_states[i]}")
            next_state = derived_state[action.item()]
            next_index = derived_indices[i][action.item()]
            states_next.append(next_state)
            indices_next.append(next_index)
        indices_next = torch.tensor(indices_next, device=self.device).view(tensordict.shape)

        done_next, rewards_next = self.get_done_reward(states_next)
        # Get next possible states for the new states, as well as rewards and done
        derived_states_next, derived_indices_next = self.get_next_states_batch(states_next)
        self.update_action_space(derived_states_next)

        
        self.current_depth += 1
        exceeded_max_depth = (self.current_depth >= self.max_depth)
        done_next = done_next | exceeded_max_depth

        next = TensorDict(
            {
                "index": indices_next,
                "state": NonTensorData(data=states_next, batch_size=tensordict.shape),
                "done": done_next,
                "reward": rewards_next,
                "derived_states": NonTensorData(data=derived_states_next, batch_size=tensordict.shape),
                "derived_indices": derived_indices_next,
            },
            batch_size=tensordict.shape,
            )

        # print('\nStep done:')
        # print_td(next)
        return next
    
    def get_next_states_batch(self,states: List[List[Term]]) -> Tuple[List[List[List[Term]]], torch.Tensor]:
        """Get next possible states and their indices for all states in the batch"""
        possible_states_next_batch = []
        possible_indices_next_batch = []
        for state in states:
            # print('    State:',[str(atom) for atom in state])
            possible_states_next = get_next_state_prolog(state)

            # Store states and get their indices
            possible_states_next_batch.append(possible_states_next)
            
            # Get indices for all possible next states
            possible_indices = [self.get_state_index([state]) for state in possible_states_next]
            possible_indices_next_batch.append(torch.tensor(possible_indices, device=self.device))

        # Do padding to possible_indices_next_batch with 0s
        max_len = max(len(indices) for indices in possible_indices_next_batch)
        if max_len > self.padding:
            raise ValueError(f"Padding is too small. Max length of indices: {max_len}")
        possible_indices_next_batch = [torch.cat([indices, torch.zeros(self.padding - len(indices), device=self.device, dtype=torch.long)]) for indices in possible_indices_next_batch]
        possible_indices_next_batch = torch.stack(possible_indices_next_batch)

        return possible_states_next_batch, possible_indices_next_batch
    
    def get_done_reward(self,states: List[List[Term]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get done and reward keys for all states in the batch. To be called in the step() method"""

        done_batch = []
        successful_batch = []
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

    def get_state_index(self, state: List[Term]) -> int:
        """Get the index of a state. If the state is not in the dictionary, add it"""
        state_key = str(state)
        if state_key not in self.state_to_index:
            index = self.next_index
            self.state_to_index[state_key] = index
            self.next_index += 1
        return self.state_to_index[state_key]



    def get_random_query(self, seed: int = 0) -> Term:
        """Generate a random query"""

        rules = self.get_rules_from_file("data/countries_s1_test.pl")
        # rules = self.get_rules_from_file("data/ancestor.pl")
        predicates = set()
        constants = set()
        for rule in rules:
            predicates.add((rule.head.predicate, len(rule.head.args)))
            constants.update([arg for arg in rule.head.args if not is_variable(arg)])
        random.seed(seed)
        predicate_random_choice = random.choice(list(predicates))
        predicate, arity = predicate_random_choice[0], predicate_random_choice[1]
        constants_list = list(constants)
        constants = random.sample(constants_list, arity)
        
        return Term(predicate, constants) 

    def get_test_query(self, seed: int = 0) -> Term:
        queries = self.get_rules_from_file("data/countries_s1_test.pl")
        random.seed(seed)
        query = random.choice(queries)
        return query.head

    def get_atom_from_string(self, atom_str: str) -> Term:
        predicate, args = atom_str.split("(")
        args = args[:-1].split(",")
        # remove any  ")" in the strings in args
        args = [re.sub(r'\)', '', arg) for arg in args]
        return Term(predicate, args)

    def get_rules_from_file(self, file_path: str) -> List[Rule]:
        """Get rules from a file"""
        with open(file_path, "r") as f:
            lines = f.readlines()
        rules = []        
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                # if there's no :-, it's a fact, split predicate
                if ":-" not in line:    
                    head = line.strip()
                    rule = Rule(self.get_atom_from_string(head), [])
                else:
                    head, body = line.strip().split(":-")
                    body = body.strip().split("),")
                    # Append the ) to all except the last element
                    body = body = [b + ")" for j, b in enumerate(body) if j != len(body) - 1] + [body[-1]]
                    body[-1] = body[-1][:-1] # remove the last "."
                    body = [self.get_atom_from_string(b) for b in body]

                    head_atom = self.get_atom_from_string(head)
                    rule = Rule(head_atom, body)
                rules.append(rule)
        return rules