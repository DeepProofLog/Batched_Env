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



class LogicProofEnv(EnvBase):
    
    def __init__(self, knowledge_f=None, test_f=None, seed=None, max_arity=1, constant_str2idx=None, predicate_str2idx=None, constant_no=0, predicate_no=0, variable_no=0, device="cpu"):
        '''Initialize the environment'''
        super().__init__(device=device)
        self.max_arity = max_arity # Maximum arity of the predicates
        self.max_atom = 10  # Maximum number of atoms in a state
        self.padding = 15  # Maximum number of possible next states
        self._make_spec()
        
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

        # self._has_dynamic_specs = True # Allow dynamic specs
        # TODO: do this with stepcount(max_steps)
        # self.max_depth = 10 # Maximum depth of the proof tree
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


    def _make_spec(self):
        '''Create the observation and action specs'''

        self.action_spec = DiscreteTensorSpec(
            n=self.padding,
            device=self.device
        )

        self.atom_index_spec = DiscreteTensorSpec(
            n=1000000,
            shape=torch.Size([self.max_atom]),
            device=self.device
        )

        self.sub_index_spec = DiscreteTensorSpec(
            n=1000000,
            shape=torch.Size([self.max_atom, self.max_arity+1]), # +1 for the predicate
            device=self.device
        )

        self.derived_atom_index_spec = DiscreteTensorSpec(
            n=1000000,
            shape=torch.Size([self.padding, self.max_atom]),
            device=self.device
        )

        self.derived_sub_index_spec = DiscreteTensorSpec(
            n=1000000,
            shape=torch.Size([self.padding, self.max_atom, self.max_arity+1]), # +1 for the predicate
            device=self.device
        )

        self.reward_spec = UnboundedContinuousTensorSpec(shape=torch.Size([1]), device=self.device)
        self.done_spec = BoundedTensorSpec(minimum=0, maximum=1, shape=torch.Size([1]),dtype=torch.bool, device=self.device)
        self.states_spec = NonTensorSpec()

        self.observation_spec = CompositeSpec(
            atom_index=self.atom_index_spec,
            sub_index = self.sub_index_spec,
            reward=self.reward_spec,
            done=self.done_spec,
            derived_states=self.states_spec,
            derived_atom_indices=self.derived_atom_index_spec,
            derived_sub_indices=self.derived_sub_index_spec,
            device=self.device,
            state=self.states_spec,
        )

    def _reset(self, tensordict: Optional[TensorDictBase] = None) -> TensorDictBase:
        '''Reset the environment to the initial state'''
        # print('Resetting...\n')

        '''Reset the atom and variable dicts and indices'''
        self.atom_to_index = {}
        self.atom_id_to_sub_id = {}
        self.variable_str2idx = {}
        self.next_atom_index = 1
        self.next_var_index = self.constant_no + 1


        seed = torch.randint(0, 1000, (1,), generator=self.rng).item()
        if self.test_f:
            state = [self.get_test_query(seed, self.test_f)]
        else:
            state = [self.get_random_query(seed, self.knowledge_f)]
        atom_index, sub_index = self.get_atom_sub_index(state)
        # print('states',states)
        # print('indices',indices)
        # print('Initial state:',[[str(atom) for atom in state] for state in states])
        # Get next possible states and their indices

        derived_states, derived_atom_indices, derived_sub_indices = self.get_next_states(state)
        self.update_action_space(derived_states)
        derived_atom_indices = torch.cat([derived_atom_indices, torch.zeros(self.padding-derived_atom_indices.size(0), self.max_atom, device=self.device, dtype=torch.int64)])
        derived_sub_indices = torch.cat([derived_sub_indices, torch.zeros(self.padding-derived_sub_indices.size(0), self.max_atom, self.max_arity+1, device=self.device, dtype=torch.int64)])
        derived_states = derived_states + [[]] * (self.padding - len(derived_states))


        out = TensorDict(
            {
                "atom_index": atom_index,
                "sub_index": sub_index,
                "state": NonTensorData(data=state),
                "done": torch.zeros(torch.Size([1]), dtype=torch.bool, device=self.device),
                "reward": torch.zeros(torch.Size([1]), dtype=torch.float32, device=self.device)*(-1),
                "derived_states": NonTensorData(data=derived_states),
                "derived_atom_indices": derived_atom_indices,
                "derived_sub_indices": derived_sub_indices,
            },
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
        action = tensordict["action"]
        derived_atom_indices = tensordict["derived_atom_indices"]
        derived_states = tensordict["derived_states"]
        derived_sub_indices = tensordict["derived_sub_indices"]

        # Given the actions, use the stored derived_states to get the state_next and derived_states_next
        if action >= len(derived_states):
            raise ValueError(f"Invalid action ({action}). Max action: {len(derived_states)}, derived_states: {derived_states}")
        next_state = derived_states[action.item()]
        next_atom_index = derived_atom_indices[action.item()]
        next_sub_index = derived_sub_indices[action.item()]

        done_next, reward_next = self.get_done_reward(next_state)
        # Get next possible states for the new states, as well as rewards and done
        derived_states_next, derived_atom_indices_next, derived_sub_indices_next = self.get_next_states(next_state)
        self.update_action_space(derived_states_next)
        derived_atom_indices_next = torch.cat([derived_atom_indices_next,
                                          torch.zeros(self.padding - derived_atom_indices_next.size(0), self.max_atom,
                                                      device=self.device, dtype=torch.int64)])
        derived_sub_indices_next = torch.cat([derived_sub_indices_next,
                                         torch.zeros(self.padding - derived_sub_indices_next.size(0), self.max_atom,
                                                     self.max_arity + 1, device=self.device, dtype=torch.int64)])
        derived_states_next = derived_states_next + [[]]*(self.padding - len(derived_states_next))

        # self.current_depth += 1
        # exceeded_max_depth = (self.current_depth >= self.max_depth)
        # done_next = done_next | exceeded_max_depth

        next = TensorDict(
            {
                "atom_index": next_atom_index,
                "sub_index": next_sub_index,
                "state": NonTensorData(data=next_state),
                "done": done_next,
                "reward": reward_next,
                "derived_states": NonTensorData(data=derived_states_next),
                "derived_atom_indices": derived_atom_indices_next,
                "derived_sub_indices": derived_sub_indices_next,
            },
            )

        # print('\nStep done:')
        # print_td(next)
        return next


    
    def get_next_states(self,state: List[Term]) -> Tuple[List[List[Term]], torch.Tensor, torch.Tensor]:
        """Get next possible states and their indices for all states in the batch"""
        possible_states_next = get_next_state_prolog(state)

        # Get indices for all possible next states
        possible_atom_indices = []
        possible_sub_indices = []
        for s in possible_states_next:
            atom_idx, sub_idx = self.get_atom_sub_index(s)
            possible_atom_indices.append(atom_idx)
            possible_sub_indices.append(sub_idx)

        possible_atom_indices = torch.stack(possible_atom_indices)
        possible_sub_indices = torch.stack(possible_sub_indices)

        return possible_states_next, possible_atom_indices, possible_sub_indices
    
    def get_done_reward(self,state: List[Term]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get done and reward keys for all states in the batch. To be called in the step() method"""

        any_atom_false = any([atom.predicate == 'False' for atom in state])
        all_atoms_true = all([atom.predicate == 'True' for atom in state])

        done = torch.tensor([any_atom_false or all_atoms_true], device=self.device)
        #TODO: is this the reward we want?
        reward = 1.0 if all_atoms_true else 0.0
        reward = torch.tensor([reward], device=self.device)

        return done, reward

    def update_action_space(self, derived_states: List[List[Term]]):
        '''Update the action space based on the possible next states 
        To be called every time the possible next states are updated'''
        
        self.action_spec = DiscreteTensorSpec(
            n=len(derived_states),
            device=self.device
        )

    def get_atom_sub_index(self, state: List[Term]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the atom and sub index for a state"""
        # Get variables
        full_state = ",".join(str(s) for s in state)
        vars = extract_var(full_state)
        for var in vars:
            if (var != "True") and (var != "False") and (var not in self.variable_str2idx):
                if self.next_var_index > self.constant_no + self.variable_no:
                    raise ValueError(f"Exceeded the maximum number of variables: {self.variable_no}")
                else:
                    index = self.next_var_index
                    self.variable_str2idx[var] = index
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
                            sub_index[i, j+1] = self.variable_str2idx[arg]
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
