from typing import List, Optional, Tuple, Dict, Union
import random
from math import prod
from utils import Term, is_variable, extract_var, print_state_transition, get_rule_from_string
from unification.prolog_unification import get_next_state_prolog, get_next_state_prolog_mnist

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

class IndexManager():

    def __init__(self, constants: set, 
                predicates: set,
                variables:set,
                constant_no: int,
                predicate_no: int,
                variable_no: int,
                rules: List[Rule],
                constants_images: set = (),
                constant_images_no: int = 0,
                rule_depend_var: bool = True,
                max_atom: int = 10,
                max_arity: int = 2,
                device: torch.device = torch.device("cpu")):
        
        self.device = device
        self.constants = constants
        self.predicates = predicates
        self.variables = variables # if rule_depend_var is the number of variables in the rules, if not, it is set in the runner
        self.constant_no = constant_no
        self.variable_no = variable_no
        self.predicate_no = predicate_no

        self.constants_images = constants_images
        self.constant_images_no = constant_images_no

        self.rules = rules
        self.rule_depend_var = rule_depend_var # If True, the variables are dependent on the rules, if False, the variables are set in the runner
        self.max_atom = max_atom  # Maximum number of atoms in a state
        self.max_arity = max_arity # Maximum arity of the predicates

        # LOCAL INDEXES
        self.atom_to_index = {} # Map atom to index
        self.atom_id_to_sub_id = {} # Map atom index to sub-indices of predicates and arguments
        self.next_atom_index = 1  # Next available index. 0 is reserved for padding
        if not self.rule_depend_var:
            self.variable_str2idx = {} # Map variable to index
            self.next_var_index = constant_no+1 # Next available index. 0 is reserved for padding

        self.create_global_idx()
        if self.rule_depend_var:
            self.rule_features_vars()

    def create_global_idx(self):
        '''Create a global index for a list of terms. Start idx counting from 1
        If there are images, reserve the first indexes for the images'''
        if self.constant_images_no>0:
            constants_wout_images = [const for const in self.constants if const not in self.constants_images]
            self.constant_str2idx = {term: i + 1 for i, term in enumerate(sorted(self.constants_images))}
            self.constant_str2idx.update({term: i + 1 + self.constant_images_no for i, term in enumerate(sorted(constants_wout_images))})
            self.constant_idx2str = {i + 1: term for i, term in enumerate(sorted(self.constants_images))}
            self.constant_idx2str.update({i + 1 + self.constant_images_no: term for i, term in enumerate(sorted(constants_wout_images))})
        else:
            self.constant_str2idx = {term: i + 1 for i, term in enumerate(sorted(self.constants))}
            self.constant_idx2str = {i + 1: term for i, term in enumerate(sorted(self.constants))}

        self.predicate_str2idx = {term: i + 1 for i, term in enumerate(sorted(self.predicates))}
        self.predicate_idx2str = {i + 1: term for i, term in enumerate(sorted(self.predicates))}

        if self.rule_depend_var:
            self.variable_str2idx = {term: i + 1 + self.constant_no for i, term in enumerate(sorted(self.variables))}
            self.variable_idx2str = {i + 1 + self.constant_no: term for i, term in enumerate(sorted(self.variables))}


    def rule_features_vars(self):
        """Create a dictionary with the features (body predicates) and variables of the rules?????"""
        self.rule_feats_vars = {}
        for i in range(len(self.rules)):
            rule = self.rules[i]
            if rule.head.predicate not in self.rule_feats_vars:
                self.rule_feats_vars[rule.head.predicate] = [f'RULE{i}_{arg}' for arg in rule.head.args]
            feature = ""
            vars = []
            for atom in rule.body:
                feature = feature+atom.predicate
                vars.append([f'RULE{i}_{arg}' for arg in atom.args])
            self.rule_feats_vars[feature] = vars

    def reset_atom(self):
        '''Reset the atom and variable dicts and indices'''
        self.atom_to_index = {}
        self.atom_id_to_sub_id = {}
        self.next_atom_index = 1
        if not self.rule_depend_var:
            self.variable_str2idx = {}
            self.next_var_index = self.constant_no+1

    def substitute_variables(self, state: List[Term]) -> List[Term]:
        """Substitute variables in a state by the variables in the rule????"""
        if not ((len(state) == 1 and (state[0].predicate == 'True' or state[0].predicate == 'False')) or (not extract_var(",".join(str(s) for s in state)))):
            state_feat = "".join(atom.predicate for atom in state)
            assert state_feat in self.rule_feats_vars, f"State feature not in rule_feats_vars: {state_feat}"
            for i in range(len(state)):
                atom = state[i]
                for j in range(len(atom.args)):
                    if is_variable(atom.args[j]):
                        atom.args[j] = self.rule_feats_vars[state_feat][i][j]
        return state


    def get_atom_sub_index(self, state: List[Term]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the atom and sub index for a state"""
        if self.rule_depend_var:
            state = self.substitute_variables(state)
        else:
            #Get variables
            full_state = ",".join(str(s) for s in state)
            vars = extract_var(full_state)
            for var in vars:
                if (var != "True") and (var != "False") and (var!= "End") and (var not in self.variable_str2idx):
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
                    elif atom.predicate == 'End':
                        sub_index[i, 0] = self.predicate_no + 3 # idx reserved for the action 'end of the proof'
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
    






class LogicEnv_gym(gym.Env):
    batch_locked = False  # Allow dynamic batch sizes
    
    def __init__(self, 
                max_depth: int = 10,
                seed: Optional[int] = None,
                device: torch.device = torch.device("cpu"),
                index_manager: Optional[IndexManager] = None,
                data_handler: Optional[DataHandler] = None,
                corruption_mode: Optional[str] = None,
                train_neg_pos_ratio: int = 1,
                limit_space: bool = True,
                dynamic_consult: bool = True,
                eval=False,
                end_proof_action: bool = False,
                ):
        
        '''Initialize the environment'''
        super().__init__()
        self.device = device

        self.corruption_mode = corruption_mode

        self.max_arity=data_handler.max_arity # Maximum arity of the predicates
        self.max_atom = 10  # Maximum number of atoms in a state
        self.padding = 20 # Maximum number of possible next states
        self.max_depth = max_depth # Maximum depth of the proof tree
        self.index_manager = index_manager
        self.predicates_arity = data_handler.predicates_arity

        self.seed = seed if seed is not None else torch.empty((), dtype=torch.int64).random_().item()
        self._set_seed(self.seed)
        self._make_spec()
        self.current_seed = self.seed # seed to select random queries

        self.dataset_name = data_handler.name
        self.facts=data_handler.facts

        if 'static' in self.corruption_mode or 'dynamic' in self.corruption_mode:
            self.counter = 0  # Determine whether to sample from positive or negative queries in KGE settings

        if self.corruption_mode == "dynamic":
            self.sampler = data_handler.sampler
            self.triples_factory = data_handler.triples_factory

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

        self.dynamic_consult = dynamic_consult
        self.current_query = None

        self.eval = eval
        self.eval_dataset = 'validation' # by default, evaluate on the validation set. It can be changed to 'test' or 'train'
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
        if self.last_query and not self.last_query == query:
            janus.query_once(f"asserta({str(self.last_query)}).")

        if query in self.facts:
            self.last_query = query
            janus.query_once(f"retract({str(self.last_query)}).")
        else:
            self.last_query = None


    def reset(self, seed: Optional[int]= None, options=None):
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

            if self.eval_idx == self.eval_len: # reset the index
                self.eval_idx = 0
            state, label = eval_dataset[self.eval_idx], 1
            self.eval_idx += 1

        else:
            
            if self.corruption_mode == "dynamic":
                state, _ = self.get_random_queries(self.train_queries, n=1, seed=self.seed)
                label = 1
                if self.counter % (int(self.train_neg_pos_ratio) + 1) != 0:
                    state = self.get_negatives(state)
                    assert len(state) == 1, f"Length of negatives: {len(state)}"
                    state = state[0] # In train there shuold be only one negative
                    label = 0
                self.counter += 1

            elif 'static' in self.corruption_mode:
                if self.counter % (int(self.train_neg_pos_ratio) + 1) == 0:
                    state, _ = self.get_random_queries(self.train_queries, n=1, seed=self.seed) 
                    label = 1
                else:
                    state, _ = self.get_random_queries(self.neg_train_queries, n=1, seed=self.seed) 
                    label = 0
                self.counter += 1

            else:  # Default case
                state, _ = self.get_random_queries(self.train_queries, n=1, seed=self.seed)
                label = 1
                
            if label == 1 and not self.dynamic_consult:
                self.new_consult_janus(state)
            elif label == 1 and self.dynamic_consult and state in self.facts:
                janus.query_once(f"retract({str(state)}).") # RODRIGO:DONT WE HAVE TO ADD THE STATE AFTERWARDS?
            # elif label == 1:   RODRIGO:CAN THIS BE DELETED?
                # self.dynamic_consult_janus(state)

        if self.dynamic_consult:
            self.current_query = state

        return self._reset([state], label)


    def reset_from_query(self, query, label, consult_janus=False):
        ''' Reset the environment from a given query and label. Consult janus is needed when 
        doing eval of train queries that are in the facts'''
        if consult_janus and label == 1:
            if not self.dynamic_consult:
                self.new_consult_janus(query)
            elif self.dynamic_consult and query in self.facts: 
                    janus.query_once(f"retract({str(query)}).") # RODRIGO:DONT WE HAVE TO ADD THE STATE AFTERWARDS?
            # self.dynamic_consult_janus(query) RODRIGO:CAN THIS BE DELETED?
        if self.dynamic_consult:
            self.current_query = query

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
        # print('\nresetting')
        # print_state_transition(self.tensordict['state'], self.tensordict['derived_states'],self.tensordict['reward'], self.tensordict['done'])
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

        if done_next:
            if self.dynamic_consult:
                if self.tensordict['label'] == 1 and self.current_query in self.facts:
                    janus.query_once(f"asserta({str(self.current_query)}).")
                    self.current_query = None
            self.tensordict['label'] = None # Reset the label to None, to be sure that it is not used in the next step

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
        # RODRIGO: CAN WE REMOVE THIS?
        # for rewards, get an float value, for dones, get a boolean value
        # if rewards.shape[0] != 1 or dones.shape[0] != 1 or self.exceeded_max_depth.shape[0] != 1:
        #     not_valid_shape = f"Invalid shape. rewards: {rewards.shape}, dones: {dones.shape}, exceeded_max_depth: {self.exceeded_max_depth.shape}"
        #     raise ValueError(not_valid_shape)
        truncated = bool(self.exceeded_max_depth)
        # RODRIGO: CAN WE REMOVE THIS?
        # if dones and not self.eval:
        #     print('     adding fact:', self.current_query, dones)
        #     janus.query(f"assertz({self.current_query}).")
        # if dones:
        #     print('dones,truncated,rewards', dones,truncated,rewards)

        # print_state_transition(self.tensordict['state'], self.tensordict['derived_states'],self.tensordict['reward'], self.tensordict['done'], action=self.tensordict['action'],truncated=truncated)
        return obs, reward, done, truncated, {}


    
    def get_next_states(self,state: List[Term]) -> Tuple[List[List[Term]], torch.Tensor, torch.Tensor]:
        """Get next possible states and their indices for all states in the batch"""

        # print('\nstate:', state)
        if self.end_proof_action: # filter the end of the proof action
            assert len(state) > 0, f"State is empty"
            if len(state) > 1:
                state = [atom for atom in state if atom.predicate != 'End']
            else:
                if state[0].predicate == 'End':
                    state = [Term(predicate='False', args=[])]
        # print('updated state:', state)
        possible_states_next = get_next_state_prolog(state, verbose=0) if self.dataset_name != "mnist_addition" else get_next_state_prolog_mnist(state, verbose=1)
        # print('possible_states_next:', possible_states_next)
        if self.end_proof_action:
            append = False
            for next_state in possible_states_next:
                if any([atom.predicate != 'True' and atom.predicate != 'False' for atom in next_state]):
                    append = True
                    break
            if append:
                possible_states_next.append([Term(predicate='End', args=[])])
        # print('updated possible_states_next:', possible_states_next,'\n')
        possible_atom_indices = []
        possible_sub_indices = []

        for s in possible_states_next:
            atom_idx, sub_idx = self.index_manager.get_atom_sub_index(s)
            possible_atom_indices.append(atom_idx)
            possible_sub_indices.append(sub_idx)

        possible_atom_indices = torch.stack(possible_atom_indices)
        possible_sub_indices = torch.stack(possible_sub_indices)

        # Do padding with 0s
        if len(possible_atom_indices) > self.padding:
            raise ValueError(f"Padding is too small. number of next states: {len(possible_atom_indices)}, padding: {self.padding}")
        
        possible_atom_indices = torch.cat([possible_atom_indices, torch.zeros(self.padding - len(possible_atom_indices), self.max_atom, device=self.device, dtype=torch.int64)])
        possible_sub_indices = torch.cat([possible_sub_indices, torch.zeros(self.padding - len(possible_sub_indices), self.max_atom, self.max_arity+1, device=self.device, dtype=torch.int64)])
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
        self.current_seed += 1
        random_instance = random.Random(self.current_seed)

        assert n <= len(queries), f"Number of queries ({n}) is greater than the number of queries ({len(queries)})"
        sampled_indices = random_instance.sample(range(len(queries)), n)
        sampled_queries = [queries[i] for i in sampled_indices]
        sampled_labels = [labels[i] for i in sampled_indices] if labels else [None]

        if n == 1:
            return sampled_queries[0], sampled_labels[0]
        else:
            return sampled_queries, sampled_labels


    def limit_action_space(self, derived_states: List[List[Term]], derived_atom_indices: torch.Tensor, derived_sub_indices: torch.Tensor) -> Tuple[List[List[Term]], torch.Tensor, torch.Tensor]:
        """Limit the action space by removing the states that have been visited""" # RODRIGO: Could you rename var names to make it more readable?
        mask = [",".join(str(s) for s in state) not in self.memory for state in derived_states]
        cutted_derived_states = [state for state, m in zip(derived_states, mask) if m]
        mask = torch.tensor(mask, device=self.device)
        mask_broadcasted = torch.cat([mask, torch.zeros(derived_atom_indices.size(0)-mask.size(0))])
        mask_broadcasted = mask_broadcasted.unsqueeze(-1)
        cutted_derived_atom_indices = derived_atom_indices * mask_broadcasted
        valid_id = torch.any(cutted_derived_atom_indices!=0, dim=-1)
        valid = cutted_derived_atom_indices[valid_id]
        cutted_derived_atom_indices = torch.cat([valid, torch.zeros(derived_atom_indices.size(0)-valid.size(0), derived_atom_indices.size(1), device=self.device, dtype=torch.int64)])
        mask_broadcasted = mask_broadcasted.unsqueeze(-1)
        cutted_derived_sub_indices = derived_sub_indices * mask_broadcasted
        valid_id = torch.any(cutted_derived_sub_indices!=0, dim=(-1, -2))
        valid = cutted_derived_sub_indices[valid_id]
        cutted_derived_sub_indices = torch.cat([valid, torch.zeros(derived_sub_indices.size(0)-valid.size(0), derived_sub_indices.size(1), derived_sub_indices.size(2), device=self.device, dtype=torch.int64)])
        return cutted_derived_states, cutted_derived_atom_indices, cutted_derived_sub_indices


    def end_in_false(self, atom_indices_shape: torch.Size, sub_indices_shape: torch.Size) -> Tuple[List[List[Term]], torch.Tensor, torch.Tensor]:
        " Return a state that ends in False"
        false_state = Term(predicate="False", args=[])

        derived_states_next = [[false_state]]
        derived_atom_indices_next = torch.zeros(atom_indices_shape, device=self.device, dtype=torch.int64)
        derived_sub_indices_next = torch.zeros(sub_indices_shape, device=self.device, dtype=torch.int64)

        if false_state not in self.index_manager.atom_to_index:
            self.index_manager.atom_to_index[false_state] = self.index_manager.next_atom_index
            self.index_manager.next_atom_index += 1

        false_sub_id = torch.tensor([self.index_manager.predicate_no + 2, 0, 0], device=self.device, dtype=torch.int64)
        derived_atom_indices_next[0, 0] = self.index_manager.atom_to_index[false_state]
        derived_sub_indices_next[0, 0] = false_sub_id
        return derived_states_next, derived_atom_indices_next, derived_sub_indices_next


    def get_negatives(self, state, all_negatives=False):
        query = [(state.args[0], state.predicate, state.args[1])] # convert query to (cte, pred, cte) format
        positive_batch = self.triples_factory.map_triples(np.array(query))
        if all_negatives:
            negative_batch = self.sampler.corrupt_batch_all(positive_batch)
        else:
            negative_batch = self.sampler.corrupt_batch(positive_batch)
        negative_batch_str = []
        for batch in negative_batch:
            for n in batch:
                negative_batch_str.append((self.index_manager.constant_idx2str[n[0].item()],self.index_manager.predicate_idx2str[n[1].item()],
                                        self.index_manager.constant_idx2str[n[2].item()]))
        state = [Term(predicate=n[1].strip(), args=[n[0].strip(), n[2].strip()]) for n in negative_batch_str] # convert each negative back to Term format
        return state
