from os.path import join 
import janus_swi as janus
from typing import List,Tuple
import re
from utils import is_variable,Term
import json
from collections import defaultdict
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



def get_queries_labels(path:str)-> Tuple[List[Term],List[bool]]:
    '''Get queries and labels from a file'''
    queries = []
    labels = []
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            query, label = line.strip().split("\t")
            queries.append(get_atom_from_string(query))
            labels.append(1 if label == "True" else (0 if label == "False" else '?'))
    return queries, labels


class DataHandler_corruptions():
    '''
    Instead of the normal test,valid txt files, we load a file with the corruptions
    Each query will have a label indicating true or false, i.e. if the query is a corruption or not
    For each set, we will have queries and labels, instead of just queries
    For train, we will load facts from the original train file, and train queries in the same way as valid, test

    if I want to do the original config, with no corruptions, just apply get_rules_from_file to the val,test file
    '''
    def __init__(self, dataset_name: str,
                    base_path: str,
                    janus_file: str,
                    # facts_file: str,
                    train_file: str = None,
                    valid_file: str = None,
                    test_file: str = None,
                    use_validation_as_train: bool = False,
                    use_only_positives: bool = False):
        
        base_path  = join(base_path, dataset_name)
        janus_path = join(base_path, janus_file)
        self.janus_path = janus_path
        # facts_path = join(base_path, facts_file)
        train_path = join(base_path, train_file) 
        valid_path = join(base_path, valid_file) if valid_file else None
        test_path = join(base_path, test_file)
        janus.consult(janus_path)

        self.facts, self.rules = get_rules_from_file(janus_path)
        self.train_queries, self.train_labels = get_queries_labels(train_path)
        self.valid_queries, self.valid_labels = get_queries_labels(valid_path)
        self.test_queries, self.test_labels = get_queries_labels(test_path)

        self.train_corruptions = self.get_corruptions(self.train_queries, join(base_path, "train_label_corruptions.json"))
        self.valid_corruptions = self.get_corruptions(self.valid_queries, join(base_path, "valid_label_corruptions.json"))
        self.test_corruptions = self.get_corruptions(self.test_queries, join(base_path, "test_label_corruptions.json"))

        if use_only_positives:
            self.train_queries = [query for query,label in zip(self.train_queries,self.train_labels) if label == 1]
            self.valid_queries = [query for query,label in zip(self.valid_queries,self.valid_labels) if label == 1]
            self.test_queries = [query for query,label in zip(self.test_queries,self.test_labels) if label == 1]
            self.train_labels = [1 for _ in self.train_queries]
            self.valid_labels = [1 for _ in self.valid_queries]
            self.test_labels = [1 for _ in self.test_queries]
        
        if use_validation_as_train:
            self.train_queries, self.train_labels = self.valid_queries, self.valid_labels
            self.valid_queries, self.valid_labels = self.test_queries, self.test_labels

        self.predicates, self.constants = self.get_predicates_and_constants()
        self.max_arity = self.get_max_arity(janus_path)
        self.constant_no, self.predicate_no = len(self.constants), len(self.predicates)

    def get_corruptions(self, queries: List[Term], file_path: str) -> dict[Term, List[Term]]:
        '''Get corruptions from the json file, in the format
        {"locatedInCR(armenia,asia).": [["locatedInCR(armenia,oceania).", true], ["locatedInCR(armenia,europe).", true], ["locatedInCR(armenia,africa).", true], [
        return a dictionary with the query as key, and a list of corruptions that are true as value
        '''
        dict_ = defaultdict(list)
        with open(file_path, "r") as f:
            corruptions_dict = json.load(f)
            for query, corruptions in corruptions_dict.items():
                query = get_atom_from_string(query[:-1])
                assert query in queries, f"Query {query} not in queries"
                for corruption, is_provable in corruptions:
                    corruption = get_atom_from_string(corruption[:-1])
                    if is_provable:
                        dict_[query].append(corruption)
        return dict_

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