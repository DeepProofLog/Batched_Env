from os.path import join 
import janus_swi as janus
from typing import List,Tuple,Dict,Optional
import re
from utils import is_variable,Term
import json
from collections import defaultdict
import re

import math
import torch
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

def get_predicates_and_arguments(rules: List[Rule], facts: List[Term]) -> Tuple[set[str], dict[str,int], set[str], set[str]]:
    predicates = set()
    constants = set()
    variables = set()
    predicates_arity = {}
    for i in range(len(rules)):
        # proof_first not related to query generation
        rule = rules[i]
        if not rule.head.predicate == "proof_first":
            # predicates.add((rule.head.predicate, len(rule.head.args)))
            predicate = rule.head.predicate
            predicates.add((predicate))
            for arg in rule.head.args:
                if is_variable(arg):
                    variables.add(f"RULE{i}_{arg}")
                else:
                    constants.add(arg)
            if predicate not in predicates_arity:
                predicates_arity[predicate] = len(rule.head.args)
            for atom in rule.body:
                predicates.add((atom.predicate))
                for arg in atom.args:
                    if is_variable(arg):
                        variables.add(f"RULE{i}_{arg}")
                    else:
                        constants.add(arg)
                if atom.predicate not in predicates_arity:
                    predicates_arity[atom.predicate] = len(atom.args)
    for atom in facts:
        # predicates.add((atom.predicate, len(atom.args)))
        predicates.add((atom.predicate))
        constants.update([arg for arg in atom.args if not is_variable(arg)])
        if atom.predicate not in predicates_arity:
            predicates_arity[atom.predicate] = len(atom.args)
    return predicates, predicates_arity, constants, variables

def get_rules_from_file(file_path: str) -> Tuple[List[Term], List[Rule]]:
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
            elif line.startswith(":-"):
                continue
            else:
                head, body = line.strip().split(":-")
                body = re.findall(r'\w+\(.*?\)', body)
                body = [get_atom_from_string(b) for b in body]
                head_atom = get_atom_from_string(head)
                rules.append(Rule(head_atom, body))
    rules = [rule for rule in rules if rule.head.predicate != "proof_first"]
    return queries,rules



# def get_queries_labels(path:str)-> Tuple[List[Term],List[bool]]:
#     '''Get queries and labels (whether they are provable) from {set}_queries.txt file'''
#     queries = []
#     labels = []
#     with open(path, "r") as f:
#         lines = f.readlines()
#         for line in lines:
#             query, label = line.strip().split("\t")
#             queries.append(get_atom_from_string(query))
#             labels.append(1 if label == "True" else (0 if label == "False" else '?'))
#     return queries, labels




# def get_queries(path:str)-> Tuple[List[Term],List[Term]]:
#     '''Get queries from a file, and the corresponding negative queries (togehter with their labels indicating if they are provable)
#     from {set}_label_corruptions.txt file'''
#     pos_queries = []
#     neg_queries = []
#     with open(path, "r") as f:
#         dicts = json.load(f)
#         for key, value in dicts.items():
#             pos_queries.append(get_atom_from_string(key))
#             for q, l in value:
#                 if l:
#                     neg_queries.append(get_atom_from_string(q))
#     return pos_queries, neg_queries



def get_queries(path:str, non_provable_queries: bool = False)-> List[Term]:
    '''Get queries and labels (whether they are provable) from {set}_label.txt file'''
    queries = []
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if non_provable_queries:
                query = line.strip()
                queries.append(get_atom_from_string(query))
            else:
                query, label = line.strip().split("\t")
                if label == "True" or non_provable_queries:
                    queries.append(get_atom_from_string(query))
    return queries

def get_corruptions(queries: List[Term], file_path: str) -> dict[Term, List[Term]]:
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
                if is_provable:
                    corruption = get_atom_from_string(corruption[:-1])
                    dict_[query].append(corruption)
    return dict_

def get_corruptions_dict(file_path: str, non_provable_corruptions: bool = False) -> dict[Term, List[Term]]:
    '''Get corruptions from the json file, in the format
    {"locatedInCR(armenia,asia).": [["locatedInCR(armenia,oceania).", true], ["locatedInCR(armenia,europe).", true], ["locatedInCR(armenia,africa).", true], [
    return a dictionary with the query as key, and a list of corruptions that are true as value
    '''
    dict_ = defaultdict(list)
    with open(file_path, "r") as f:
        corruptions_dict = json.load(f)
        for query, corruptions in corruptions_dict.items():
            query = get_atom_from_string(query[:-1])
            for corruption, is_provable in corruptions:
                if is_provable or non_provable_corruptions:
                    corruption = get_atom_from_string(corruption[:-1])
                    dict_[query].append(corruption)
    return dict_



from data.mnist_addition.MNIST import addition, multiplication
import numpy as np

class DataHandlerMnist:
    """Handles MNIST data with corruption support in a simplified structure"""
    
    def __init__(self, dataset_name: str, base_path: str, janus_file: str, name: str = None):

        self.dataset_name = name
        self.corruption_mode = None
        self.n_digits = 2

        # Basic setup
        self.base_path = join(base_path, dataset_name)
        self.janus_path = join(self.base_path, janus_file)
        self.max_arity = 5

        # Load data
        self._load_janus_data()
        self._load_datasets()
        self._process_datasets()
        self._setup_constants()
        # self._print_debug_info()
        
    def _load_janus_data(self):
        """Load Janus facts and rules"""
        janus.consult(self.janus_path)
        with open(self.janus_path, "r") as f:
            self.janus_facts = f.readlines()
        self.facts, self.rules = get_rules_from_file(self.janus_path)

    def _load_datasets(self):
        """Load all datasets at once"""
        self.datasets = {
            'train': addition(n=self.n_digits, dataset="train", seed=42),
            'valid': addition(n=self.n_digits, dataset="val", seed=42),
            'test': addition(n=self.n_digits, dataset="test", seed=42)
        }

    def _process_datasets(self):
        """Process all datasets similarly"""
        queries = {}
        self.images = defaultdict(list)
        self.image_strings = {}
        d = 0  # Image counter
        
        for set_name in ['train', 'valid', 'test']:
            # Store raw data
            labels, digits, l1, l2 = [], [], [], []
            for i in range(10):  # 10 samples per set
                dl1, dl2, label, dgts = self.datasets[set_name][i]
                labels.append(label)
                digits.append(dgts)
                l1.append(dl1)
                l2.append(dl2)
            
            # Create image strings
            img_strings = []
            for dgts in digits:
                group = []
                for digit in dgts:
                    group.extend([f"im_{d+i}" for i in range(len(digit))])
                    d += len(digit)
                img_strings.append(group)
            
            # Create queries
            queries[set_name] = [
                Term("addition", [*imgs, str(label)])
                for imgs, label in zip(img_strings, labels)
            ]
            # Store queries
            setattr(self, f'{set_name}_queries', queries[set_name])
            # Store processed data
            setattr(self, f'{set_name}_labels', labels)

            self.image_strings[set_name] = img_strings
            
            # Store images
            im_dict = defaultdict(list)
            for dgts, dl1, dl2 in zip(digits, l1, l2):
                im_dict[str(dgts)] = (dl1, dl2)
            self.images.update(im_dict)

    def _setup_constants(self):
        """Setup constants and predicates"""
        # Collect all image IDs
        all_images = []
        for set_name in ['train', 'valid', 'test']:
            all_images.extend([img for group in self.image_strings[set_name] for img in group])
        
        # Collect all labels
        all_labels = []
        for set_name in ['train', 'valid', 'test']:
            all_labels.extend(map(str, getattr(self, f'{set_name}_labels')))
        
        # Set constants
        self.constants = set(all_images + all_labels)
        self.constants_images = set(all_images)
        
        # Predicate setup
        self.predicates = {"addition", "digit"}
        self.predicates_arity = {"addition": 3, "digit": 2}
        self.variables = set(['RULE0_Z','RULE0_Y', 'RULE0_X', 'RULE0_Y2', 'RULE0_X2'])
        self.constant_no, self.constant_images_no, self.predicate_no, self.variable_no = \
            len(self.constants), len(self.constants_images), len(self.predicates), len(self.variables)

    def _print_debug_info(self):
        """Print debugging information"""
        print(f"\nTrain queries: {len(self.train_queries), self.train_queries}")
        print(f"\nValid queries: {len(self.valid_queries), self.valid_queries}")
        print(f"\nTest queries: {len(self.test_queries), self.test_queries}")
        print(f"\nImages: {len(self.images), [(i, c[0].shape, c[1].shape) for i, c in self.images.items()]}")
        print(f"\nConstant images: {self.constant_images_no, self.constants_images}")
        print(f"\nConstants: {self.constant_no, self.constants}")
        print(f"\nPredicates: {self.predicate_no, self.predicates}")
        print(f"\nVariables: {self.variable_no, self.variables}")
        print(f"\nMax arity: {self.max_arity}")



class DataHandlerKGE():
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
                    train_file: str = None,
                    valid_file: str = None,
                    test_file: str = None,
                    # str, with an optinoal None
                    corruption_mode: Optional[str] = None,
                    non_provable_corruptions: bool = False,
                    non_provable_queries: bool = False):

        self.dataset_name = dataset_name
        
        base_path  = join(base_path, dataset_name)
        janus_path = join(base_path, janus_file)
        self.janus_path = janus_path
        janus.consult(janus_path)

        train_path = join(base_path, train_file) 
        valid_path = join(base_path, valid_file) if valid_file else None
        test_path = join(base_path, test_file)

        self.facts, self.rules = get_rules_from_file(janus_path)

        if 'static' in corruption_mode:
            self.train_corruptions = get_corruptions_dict(train_path,non_provable_corruptions)
            self.valid_corruptions = get_corruptions_dict(valid_path,non_provable_corruptions)
            self.test_corruptions   = get_corruptions_dict(test_path,non_provable_corruptions)

            # renamed from pos_train_queries to train_queries to be consistent with valid and test
            # we dont need neg_train_queries, as we have the corruptions in the train_corruptions that we use in evaluation
            self.train_queries, self.neg_train_queries = list(self.train_corruptions.keys()), list(self.train_corruptions.values())
            self.neg_train_queries = [item for sublist in self.neg_train_queries for item in sublist]
            self.valid_queries = list(self.valid_corruptions.keys())
            self.test_queries = list(self.test_corruptions.keys())

        elif 'dynamic' in corruption_mode:
            self.train_queries = get_queries(train_path, non_provable_queries)
            self.valid_queries = get_queries(valid_path, non_provable_queries)
            self.test_queries = get_queries(test_path, non_provable_queries)
            self.train_corruptions = self.valid_corruptions = self.test_corruptions = self.neg_train_queries = None

            # if there're queries with predicates not in the rules, exclude them
            rules_head_predicates = set([rule.head.predicate for rule in self.rules])
            exclude_train = len([query for query in self.train_queries if query.predicate not in rules_head_predicates])
            exclude_valid = len([query for query in self.valid_queries if query.predicate not in rules_head_predicates])    
            exclude_test = len([query for query in self.test_queries if query.predicate not in rules_head_predicates])
            if exclude_train>0:
                print(f"Number of train queries excluded: {exclude_train}. Ratio excluded: {round(exclude_train/len(self.train_queries),3)}")
            if exclude_valid>0:
                print(f"Number of valid queries excluded: {exclude_valid}. Ratio excluded: {round(exclude_valid/len(self.valid_queries),3)}")
            if exclude_test>0:
                print(f"Number of test queries excluded: {exclude_test}. Ratio excluded: {round(exclude_test/len(self.test_queries),3)}")
            
            # filter the train queries whose predicates are not in the rules
            self.train_queries = [query for query in self.train_queries if query.predicate in rules_head_predicates]
            self.valid_queries = [query for query in self.valid_queries if query.predicate in rules_head_predicates]
            self.test_queries = [query for query in self.test_queries if query.predicate in rules_head_predicates]


        self.janus_facts = []
        with open(janus_path, "r") as f:
            self.janus_facts = f.readlines()

        self.predicates, self.predicates_arity, self.constants, self.variables = get_predicates_and_arguments(self.rules, self.facts)
        self.max_arity = get_max_arity(janus_path)
        self.constant_no, self.predicate_no, self.variable_no = len(self.constants), len(self.predicates), len(self.variables)


        self.entity2domain = None
        self.domain2entity = None
        if corruption_mode == "dynamic":
            if 'countries' in self.dataset_name or 'ablation' in self.dataset_name:
                # load the domain file
                domain_file = join(base_path, "domain2constants.txt")
                self.entity2domain = {}
                self.domain2entity = defaultdict(list)
                with open(domain_file, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip()
                        domain, *entities = line.split()
                        for entity in entities:
                            self.entity2domain[entity] = domain
                            self.domain2entity[domain].append(entity)

           

class DataHandler:
    ''' class that calls to DataHandlerMnist if the dataset is mnist_addition, and DataHandlerKGE otherwise'''
    def __init__(self, dataset_name: str, 
                 base_path: str, 
                 janus_file: str, 
                 train_file: str = None, 
                 valid_file: str = None, 
                 test_file: str = None, 
                 corruption_mode: Optional[str] = None, 
                 name: str = None,
                 non_provable_corruptions: bool = False,
                 non_provable_queries: bool = False):
        if dataset_name == "mnist_addition":
            self.info = DataHandlerMnist(dataset_name, 
                                         base_path, 
                                         janus_file, 
                                         )
        else:
            self.info = DataHandlerKGE(dataset_name, 
                                       base_path, 
                                       janus_file, 
                                       train_file, 
                                       valid_file, 
                                       test_file, 
                                       corruption_mode, 
                                       non_provable_corruptions,
                                       non_provable_queries)


