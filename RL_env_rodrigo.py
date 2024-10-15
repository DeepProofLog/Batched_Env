from typing import List, Optional, Tuple, Dict
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class Term:
    def __init__(self, predicate: str, args: List[str]):
        self.predicate = predicate  # Predicate name
        self.args = args  # List of arguments (constants or variables)

    def __str__(self):
        return f"{self.predicate}({', '.join(self.args)})"

class Rule:
    def __init__(self, head: Term, body: List[Term]):
        self.head = head
        self.body = body

    def __str__(self):
        body_str = ", ".join(str(term) for term in self.body)
        return f"{self.head} :- {body_str}"
    

def apply_substitution(term: Term, substitution: Dict[str, str]) -> Term:
    """Apply the substitution to a given term."""
    substituted_args = [substitution.get(arg, arg) for arg in term.args]
    return Term(term.predicate, substituted_args)

def is_variable(arg: str) -> bool:
    """Check if an argument is a variable (uppercase) or constant (lowercase)."""
    return arg.isupper()

def unify_atom(rule: Rule, atom: Term, verbose=False) -> Tuple[List[Term], bool]:
    """
    Unify two terms and return the resulting child proofs (atoms) that need to be proven,
    along with the updated substitution as a list of atom paths.
    """
    print('---------------------------------') if verbose else None
    rule_head = rule.head
    rule_body = rule.body
    print (f'Unifying atom {atom} with rule head {rule_head} and rule body {rule_body}') if verbose else None
    # Apply substitution to the current atom and rule head
    term1 = atom
    term2 = rule_head
    if term1.predicate != term2.predicate or len(term1.args) != len(term2.args):
        print('Unification failed. Predicates or number of arguments do not match') if verbose else None
        print('---------------------------------') if verbose else None
        return None, False
    
    # Create an empty list to store the substitutions
    assignment = {}
    # Perform unification
    is_fact = not rule_body
    if is_fact:
        for arg_atom, arg_rulehead in zip(term1.args, term2.args):
            if is_variable(arg_atom): # if arg_atom is a variable, assign it to arg_rulehead
                assignment[arg_atom] = arg_rulehead
            else: # if arg_atom is a constant, check if it is the same as arg_rulehead
                if arg_atom != arg_rulehead:
                    print('Unification failed. Constants do not match') if verbose else None
                    print('---------------------------------') if verbose else None
                    return None, False
    else:
        for arg_atom, arg_rulehead in zip(term1.args, term2.args):
            if not is_variable(arg_atom): # if arg_atom is a constant, assign it to arg_rulehead
                assignment[arg_rulehead] = arg_atom
            else: # if arg_atom is a variable, assign it to arg_rulehead
                assignment[arg_rulehead] = arg_atom
                
    print(f'Unification successful') if verbose else None
    print(f'Substitution: {assignment}') if verbose else None
    # Apply the substitution to the body of the rule
    unified_body = [apply_substitution(term, assignment) for term in rule_body]

    print(f"Unification: {atom} -> {', '.join(str(body) for body in unified_body) if unified_body else '[]'}") if verbose else None
    print(f'---------------------') if verbose else None

    return unified_body, True


def unify_state( state: List[Term], rules: List[Rule], action: List[int], verbose=int) -> Tuple[List[Term], bool]:
    """
    Given a state and a atom of that state, unify that atom with the head of the rule.
    The main point is that if the atom unifies with a fact, then the rest of the atoms of the state need to update their variables value 
    
    Please take into account for the new state that the atoms need to have the same order as they had in the original state
    """
    atom, rule = state[action[0]], rules[action[1]]
    print('---------------------------------') if verbose>=1 else None
    print(f'Atom: {atom}') if verbose>=1 else None
    print(f'Rule: {rule}') if verbose>=1 else None
    print(f'action: {action}')
    print(f'State: {[str(atom) for atom in state]}')

    # Do unification with the atom and the head of the rule
    child_proof, unified = unify_atom(rule, atom, verbose=False)
    print(f'Unified: {unified}, unification: {[str(atom) for atom in child_proof] if child_proof else []}') if verbose>=1 else None
    if unified: # the atom unifies with the head of the rule
        new_state = []
        if not child_proof:   # If the rule has no body, we've found a fact (leaf node). Dont include the fact in the new state
            print('Found a fact') if verbose==2 else None
            # check which vars in the original atom have been assigned a value. This is useful for the other atoms in the state
            vars_assigned = {arg: value for arg, value in zip(atom.args, rule.head.args) if is_variable(arg)} # If there's a var in atom, check its assigned value
            print(f'Vars assigned: {vars_assigned}') if verbose==2 else None
            for i, state_atom in enumerate(state):
                print(f'Doing unification for atom {state_atom}') if verbose==2 else None
                if state_atom != atom:
                    # apply the substitution to the rest of the atoms in the state
                    new_state.append(Term(state_atom.predicate, [vars_assigned.get(arg, arg) for arg in state_atom.args]))
                    print(f'    Applying substitution to atom {state_atom}') if verbose==2 else None
                    print(f'    New state: {[str(state_atom) for state_atom in new_state]}') if verbose==2 else None

        else: # if the rule has a body, then the atom is not a fact. It is a rule, so we need to include the body of the rule in the new state
            # the rest of the atoms of the state dont need to update their variables value because the rule is not a fact, has a body
            for i, state_atom in enumerate(state):
                if state_atom != atom:
                    new_state.append(state_atom)
                else:
                    new_state.extend(child_proof)
        print(f'New state: {[str(state_atom) for state_atom in new_state]}') if verbose>=1 else None
        print('---------------------------------') if verbose>=1 else None
        return new_state, True
    else:
        print('---------------------------------') if verbose>=1 else None
        return None, False
    
def get_next_state(state: List[Term], rules: List[Rule], action: List[int], verbose=int) -> Tuple[List[Term], bool]:
    '''
    Given a state, an action, and the rules, return the next state of the environment, and whether it is a finished state
    '''

    new_state, unified = unify_state(state, rules, action, verbose=1)
    if unified and not new_state:
        done = successful_end = True
        return new_state, done, successful_end
    elif unified and new_state:
        done = successful_end = False
        return new_state, done, successful_end
    else:
        done = True
        successful_end = False
        return new_state, done, successful_end
    

def action_selection(len_state: int, len_rules: int, ordered=True, num_actions='all') -> Tuple[List[int], List[int]]:
    """
    Select the action to take given the current state and the available rules.
    """
    if ordered:
        indeces_state = list(range(len_state))
        indeces_rule = list(range(len_rules))
    else:
        indeces_state = random.sample(range(len_state), len_state)
        indeces_rule = random.sample(range(len_rules), len_rules)
    
    # For every index in indeces_state, select the index in indeces_rule, e.g. if 
    # indeces_state = [0, 1] and indeces_rule = [0, 1, 2], then the actions are [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    indeces = [(i, j) for i in indeces_state for j in indeces_rule]

    if num_actions == 'all':
        return indeces
    else:
        return indeces[:num_actions]




class LogicProofEnv(gym.Env):
    def __init__(self, initial_state, rules, max_depth=10):
        super(LogicProofEnv, self).__init__()
        self.state = initial_state  # The initial state of the environment (list of Term objects)
        self.rules = rules  # The set of rules
        self.max_depth = max_depth
        self.current_depth = 0
        
        # Define the action space: (index in state, index in rules).
        self.action_space = spaces.MultiDiscrete([len(self.state), len(self.rules)])

        # Define the observation space. You can use an integer representation of the state
        self.observation_space = spaces.Discrete(100000)  # A large number to represent possible states
    
    def get_query(self):
        predicates = set()
        constants = set()
        for rule in self.rules: # get the predicates from the heads of the rules
            predicates.add((rule.head.predicate, len(rule.head.args)))
            constants.update([arg for arg in rule.head.args if not is_variable(arg)])
        # print(f'Predicates: {predicates}, Constants: {constants}')
        # choose a random predicate
        predicate_random_choice = random.choice(list(predicates))
        predicate, arity = predicate_random_choice[0], predicate_random_choice[1]
        # based on the arity of the predicate, get the constants
        constants = random.sample(constants, arity)
        return Term(predicate, constants)

    def reset(self):
        # Reset the environment to the initial state and return the initial observation
        self.current_depth = 0

        # # I can choose a specific state (query)
        # self.state = [Term("parent", ["charlie", "A"]), Term("ancestor", ["A", "alice"])]   # [Term("ancestor", ["charlie", "alice"])]
        
        # I can choose a random state (query). From the rules, get all the predicates and constantes (arguments) 
        self.state = [self.get_query()]
        self.action_space = spaces.MultiDiscrete([len(self.state), len(self.rules)]) # Update the action space
        print(f'Initial state: {[str(atom) for atom in self.state]}')
        return self.state_to_observation(self.state)
    
    def step(self, action):
        # Take the action and update the state, return the next observation, reward, done, and info
        state_idx, rule_idx = action
        
        # Apply unification
        new_state, done, successful_end = get_next_state(self.state, self.rules, action, verbose=1)  
        
        if done and successful_end:
            reward = 1
        else:
            reward = 0
        
        self.current_depth += 1
        done = self.current_depth >= self.max_depth or not new_state  # Episode ends if max depth or proof found

        if new_state:
            self.state = new_state
        
        return self.state_to_observation(self.state), reward, done, {}
    
    def state_to_observation(self, state):
        # Convert the current state (list of Term objects) into an observation (integer)
        return np.array([hash(str(atom)) for atom in state])  # A simple hash-based encoding of state




# Define the rules
rules = [
    Rule(Term("parent", ["bob", "alice"]), []),
    Rule(Term("parent", ["charlie", "bob"]), []),
    Rule(Term("parent", ["charlie", "mary"]), []),
    Rule(Term("ancestor", ["X", "Y"]), [Term("parent", ["X", "Y"])]),
    Rule(Term("ancestor", ["X", "Y"]), [Term("parent", ["X", "Z"]), Term("ancestor", ["Z", "Y"])]),
]

states = [ 
            [Term("ancestor", ["charlie", "alice"])],
            [Term("parent", ["charlie", "A"]),Term("ancestor", ["A", "alice"])]
         ]
state = states[1]

# Find the proof
env = LogicProofEnv(state, rules)
obs = env.reset()
done = False

counter = 0
action_sequences = None # [[0,1],[0,3],[0,0]]
while not done:
    print('\n\n***********************************************')

    if action_sequences is None:
        action = env.action_space.sample()
    else: 
        # print(f'counter: {counter}, action_sequences: {action_sequences}')
        action = action_sequences[counter]

    print(f"Step {env.current_depth}")
    print(f"State: {[str(atom) for atom in env.state]}")        
    print(f"Action: {action}")

    obs, reward, done, info = env.step(action)
    print(f"Reward: {reward}, Done: {done}. Info: {info}")
    counter += 1
