from os.path import join 
import janus_swi as janus
from typing import List,Tuple
import re
from utils import is_variable,Term
    
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
    

def get_atom_from_string(atom_str: str) -> Term:
    predicate, args = atom_str.split("(")
    args = args[:-1].split(",")
    # remove any  ")" in the strings in args
    args = [re.sub(r'\)', '', arg) for arg in args]
    return Term(predicate, args)


def get_rules_from_file(file_path: str) -> List[Rule]:
    """Get rules from a file"""
    rules = []
    queries = []
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            # if there's no :-, it's a fact, split predicate
            if ":-" not in line:
                head = line.strip()
                queries.append(get_atom_from_string(head))
            else:
                head, body = line.strip().split(":-")
                body = re.findall(r'\w+\(.*?\)', body)
                body = [get_atom_from_string(b) for b in body]
                head_atom = get_atom_from_string(head)
                rules.append(Rule(head_atom, body))
    rules = [rule for rule in rules if not rule.head.predicate != "proof_first"]
    return queries,rules


class DataHandler():
    def __init__(self, dataset_name: str,
                    base_path: str,
                    train_file: str = None,
                    valid_file: str = None,
                    test_file: str = None):
        self.dataset_name = dataset_name
        self.base_path = base_path
        self.train_file = train_file
        self.valid_file = valid_file
        self.test_file = test_file
        
        base_path  = join(base_path, dataset_name)
        train_path = join(base_path, train_file) 
        valid_path = join(base_path, valid_file) if valid_file else None
        test_path = join(base_path, test_file)

        self.train_queries, self.rules = get_rules_from_file(train_path)
        self.facts = self.train_queries
        self.valid_queries, _ = get_rules_from_file(valid_path)
        self.test_queries, _ = get_rules_from_file(test_path)
        janus.consult(train_path)
        self.predicates, self.constants = self.get_predicates_and_constants()

        self.max_arity = self.get_max_arity(train_path)
        self.constant_no, self.predicate_no = len(self.constants), len(self.predicates)

    def get_predicates_and_constants(self) -> Tuple[set, set]:
        predicates = set()
        constants = set()
        for rule in self.rules:
            # proof_first not related to query generation
            if not rule.head.predicate == "proof_first":
                # predicates.add((rule.head.predicate, len(rule.head.args)))
                predicates.add((rule.head.predicate))
                constants.update([arg for arg in rule.head.args if not is_variable(arg)])
        for atom in self.facts:
            # predicates.add((atom.predicate, len(atom.args)))
            predicates.add((atom.predicate))
            constants.update([arg for arg in atom.args if not is_variable(arg)])
        return predicates, constants

    def get_max_arity(self, file_path:str)-> int:
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

