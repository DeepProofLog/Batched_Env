import re
from typing import List, Dict
import torch
from tensordict import TensorDict, TensorDictBase, NonTensorData

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
    """Check if an argument is a variable."""
    return arg[0].isupper() or arg[0] == '_'

def extract_var(state: str)-> list:
    '''Extract unique variables from a state: start with uppercase letter or underscore'''
    pattern = r'\b[A-Z_][a-zA-Z0-9_]*\b'
    vars = re.findall(pattern, state)
    return list(dict.fromkeys(vars))

def get_max_arity(file_path:str)-> int:
    '''Get the maximum arity of the predicates in the file'''
    max_arity = 0
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            clauses = re.findall(r'\w+\(.*?\)', line)
            for clause in clauses:
                predicate, args = clause.split("(")
                arity = len(args.split(","))
                if arity > max_arity:
                    max_arity = arity
    return max_arity


def print_rollout(data):
    """Prints each batch and then transposes to print each step in rollout data."""
    # Print data by batch
    for i, batch_data in enumerate(data):
        print(f'Batch {i}')
        print_td(batch_data)
        print('\n')
    print('\n')

    # Print data by step after transposing
    data = data.transpose(0, 1)
    for i, step_data in enumerate(data):
        print(f'Step {i}:', [[str(atom) for atom in state] for state in step_data['state']])
        for j, state in enumerate(step_data['state']):
            print(f'     Step {i}, Batch {j}:', [str(atom) for atom in state])
    print('\n')


def print_td(td: TensorDictBase, next=False, exclude_states=False):
    """Prints keys and values of a TensorDict, with optional flags for next states and excluding states."""
    print_title = 'Next TensorDict' if next else 'TensorDict'
    print(f'{"="*10} {print_title} {"="*10}')
    
    for key, value in td.items():
        if (key == 'derived_states' and not exclude_states):
            value_data = value.data
            print(f'Key: {key}', value_data)
            for i, batch in enumerate(value_data):
                for j, next_state in enumerate(batch):
                    print(f'     {i}, {j} next_possible_state:', [str(atom) for atom in next_state])

        elif (key == 'state' and not exclude_states):
            value_data = value.data
            for i, state in enumerate(value_data):
                print(f'     {i} state:', [str(atom) for atom in state])

        elif key == 'next':
            print(f'Key: {key}')
            print_td(value, next=True, exclude_states=exclude_states)

        elif key not in {'state', 'derived_states'}:
            if isinstance(value, torch.Tensor):
                print(f'Key: {key} Shape: {value.shape} Values:\n{value}')
            elif isinstance(value, list):
                print(f'Key: {key} Length: {len(value)} Values:\n{value}')
            else:
                print(f'Key: {key} Value:\n{value}')
    
    print("="*30 if not next else "^"*30)