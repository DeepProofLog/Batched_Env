import re
from typing import List, Dict, Tuple
import torch
from tensordict import TensorDict, TensorDictBase, NonTensorData
import pickle
import numpy as np

class Term:
    def __init__(self, predicate: str, args: List[str]):
        self.predicate = predicate  # Predicate name
        self.args = args  # List of arguments (constants or variables)

    def __str__(self):
        return f"{self.predicate}({', '.join(self.args)})"

    def __repr__(self):
        return f"{self.predicate}({', '.join(self.args)})"
    
class Rule:
    def __init__(self, head: Term, body: List[Term]):
        self.head = head
        self.body = body

    def __str__(self):
        body_str = ", ".join(str(term) for term in self.body)
        return f"{self.head} :- {body_str}"    

    def __repr__(self):
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

def get_atom_from_string(atom_str: str) -> Term:
    predicate, args = atom_str.split("(")
    args = args[:-1].split(",")
    # remove any  ")" in the strings in args
    args = [re.sub(r'\)', '', arg) for arg in args]
    return Term(predicate, args)

def get_rules_from_file(file_path: str) -> List[Rule]:
    """Get rules from a file"""
    rules = []
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            # if there's no :-, it's a fact, split predicate
            if ":-" not in line:
                head = line.strip()
                rule = Rule(get_atom_from_string(head), [])
            else:
                head, body = line.strip().split(":-")
                body = re.findall(r'\w+\(.*?\)', body)
                body = [get_atom_from_string(b) for b in body]

                head_atom = get_atom_from_string(head)
                rule = Rule(head_atom, body)
            rules.append(rule)
    return rules

def get_constants_predicates(rules: List[Rule]) -> Tuple[set, set]:
    """Get the set of constants and predicates from a list of rules"""
    predicates = set()
    constants = set()
    for rule in rules:
        if not rule.head.predicate == "proof_first":
            # get from head the predicate and the constants if they are not variables
            predicates.add(str(rule.head.predicate))
            constants.update([str(arg) for arg in rule.head.args if not is_variable(arg)])
            # get from body the predicates and the constants if they are not variables
            for atom in rule.body:
                predicates.add(str(atom.predicate))
                constants.update([str(arg) for arg in atom.args if not is_variable(arg)])
    return constants, predicates


def create_global_idx(file_path:str)-> Tuple[dict, dict]:
    '''Create a global index for a list of terms. Start idx counting from 1'''
    rules = get_rules_from_file(file_path)
    constants, predicates = get_constants_predicates(rules)
    constant_str2idx = {term: i + 1 for i, term in enumerate(constants)}
    predicate_str2idx = {term: i + 1 for i, term in enumerate(predicates)}
    return constant_str2idx, predicate_str2idx


def read_embeddings(file_c:str, file_p:str, constant_str2idx:dict, predicate_str2idx:dict)-> Tuple[dict, dict]:
    '''Read embeddings from a file'''
    with open(file_c, 'rb') as f:
        constant_embeddings = pickle.load(f)
    with open(file_p, 'rb') as f:
        predicate_embeddings = pickle.load(f)
    # in cte embeddings the key is the domain (we ignore it) and the value is a dict, whose key is the constant and the value is the embedding
    constant_embeddings = {
        constant: emb
        for domain, domain_dict in constant_embeddings.items()
        for constant, emb in domain_dict.items()
    }
    # in pred embeddings as key take the first str until ( and the value is the embedding
    predicate_embeddings = {
        pred.split('(')[0]: emb
        for pred, emb in predicate_embeddings.items()
    }
    # using the str2idx dicts, create the idx2emb dicts
    constant_idx2emb = {constant_str2idx[constant]: emb for constant, emb in constant_embeddings.items()}
    predicate_idx2emb = {predicate_str2idx[predicate]: emb for predicate, emb in
                              predicate_embeddings.items()}

    # order the embeddings by index
    constant_idx2emb = {idx: constant_idx2emb[idx] for idx in sorted(constant_idx2emb)}
    predicate_idx2emb = {idx: predicate_idx2emb[idx] for idx in sorted(predicate_idx2emb)}
    return constant_idx2emb, predicate_idx2emb

def create_embed_tables(constant_idx2emb:dict, predicate_idx2emb:dict, var_no:int)-> Tuple[dict, dict]:
    '''Create embedding tables for constants + variables and predicates'''
    # embeddings ndarray to tensor
    constant_idx2emb = torch.tensor(np.stack([constant_idx2emb[key] for key in constant_idx2emb.keys()]), dtype=torch.float32)
    predicate_idx2emb = torch.tensor(np.stack([predicate_idx2emb[key] for key in predicate_idx2emb.keys()]), dtype=torch.float32)
    # TODO: better ways to do variable and T/F embeddings?
    # random embeddings for True, False and variables
    embed_dim = constant_idx2emb.size(1)
    for i in range(2):
        predicate_idx2emb = torch.cat([predicate_idx2emb, torch.rand(1, embed_dim)], dim=0)
    for i in range(var_no):
        constant_idx2emb= torch.cat([constant_idx2emb, torch.rand(1, embed_dim)], dim=0)
    return constant_idx2emb, predicate_idx2emb


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