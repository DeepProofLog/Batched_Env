from os.path import join 
import janus_swi as janus
from typing import List, Tuple, Dict, Optional, Set
import re
import json
from collections import defaultdict
import numpy as np
import torch
from utils import is_variable, Term

# Rule class definition
class Rule:
    """Represents a logical rule with a head term and body terms."""
    def __init__(self, head: Term, body: List[Term]):
        self.head = head
        self.body = body

    def __str__(self):
        body_str = ", ".join(str(term) for term in self.body)
        return f"{self.head} :- {body_str}"    

    def __repr__(self):
        return self.__str__()
    

def get_atom_from_string(atom_str: str) -> Term:
    """
    Parse a string representation of an atom into a Term object.
    
    Args:
        atom_str: String representation of an atom (e.g., "predicate(arg1,arg2)")
        
    Returns:
        Term object representing the atom
    """
    try:
        predicate, args_str = atom_str.split("(", 1)
        args = args_str[:-1].split(",")
        # Remove any ")" in the arguments
        args = [re.sub(r'\)', '', arg) for arg in args]
        return Term(predicate, args)
    except Exception as e:
        raise ValueError(f"Failed to parse atom string '{atom_str}': {str(e)}")

def get_max_arity(file_path:str)-> int:
    '''Get the maximum arity of the predicates in the file'''
    max_arity = 0
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                clauses = re.findall(r'\w+\(.*?\)', line)
                for clause in clauses:
                    try:
                        predicate, args = clause.split("(", 1)
                        arity = len(args.split(","))
                        if arity > max_arity:
                            max_arity = arity
                    except ValueError:
                        continue  # Skip malformed clauses
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_path} not found")
    return max_arity

def get_predicates_and_arguments(rules: List[Rule], facts: List[Term]) -> Tuple[Set[str], Dict[str, int], Set[str], Set[str]]:
    """
    Extract predicates, their arities, constants, and variables from rules and facts.
    
    Args:
        rules: List of Rule objects
        facts: List of Term objects representing facts
        
    Returns:
        Tuple containing:
        - Set of predicate names
        - Dictionary mapping predicate names to their arities
        - Set of constants
        - Set of variables
    """
    predicates = set()
    constants = set()
    variables = set()
    predicates_arity = {}
    
    # Process rules
    for i, rule in enumerate(rules):
        # Skip proof_first predicates (not related to query generation)
        if rule.head.predicate == "proof_first":
            continue
            
        # Process head
        predicate = rule.head.predicate
        predicates.add(predicate)
        for arg in rule.head.args:
            if is_variable(arg):
                variables.add(f"RULE{i}_{arg}")
            else:
                constants.add(arg)
        if predicate not in predicates_arity:
            predicates_arity[predicate] = len(rule.head.args)
            
        # Process body
        for atom in rule.body:
            predicates.add(atom.predicate)
            for arg in atom.args:
                if is_variable(arg):
                    variables.add(f"RULE{i}_{arg}")
                else:
                    constants.add(arg)
            if atom.predicate not in predicates_arity:
                predicates_arity[atom.predicate] = len(atom.args)
    
    # Process facts
    for atom in facts:
        predicates.add(atom.predicate)
        constants.update([arg for arg in atom.args if not is_variable(arg)])
        if atom.predicate not in predicates_arity:
            predicates_arity[atom.predicate] = len(atom.args)
            
    return predicates, predicates_arity, constants, variables

def get_rules_from_file(file_path: str) -> Tuple[List[Term], List[Rule]]:
    """
    Parse a file to extract facts (queries) and rules.
    
    Args:
        file_path: Path to the file containing facts and rules
        
    Returns:
        Tuple containing:
        - List of Term objects representing facts/queries
        - List of Rule objects
    """
    rules = []
    queries = []
    
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                if not line or line.startswith("%") or line.startswith(":-"):
                    continue
                if ":-" not in line: # If there's no ":-", it's a fact
                    head = line.strip()
                    queries.append(get_atom_from_string(head))
                # Otherwise it's a rule
                else:
                    head, body = line.strip().split(":-")
                    body = re.findall(r'\w+\(.*?\)', body)
                    body = [get_atom_from_string(b) for b in body]
                    head_atom = get_atom_from_string(head)
                    rules.append(Rule(head_atom, body))
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_path} not found")
        
    # Filter out rules with "proof_first" head predicate
    rules = [rule for rule in rules if rule.head.predicate != "proof_first"]
    return queries, rules


def get_rules_from_rules_file(file_path: str) -> List[Rule]:
    """
    Parse a file to extract rules.
    
    Args:
        file_path: Path to the file containing rules
        
    Returns:
        List of Rule objects
    """
    rules = []
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                if not line or line.startswith("%"):
                    continue
                else:
                    line = line.split(':')
                    # third element is the rule itself. Split by ->
                    rule = line[2].split('->')
                    # second element is the head of the rule
                    rule_head = rule[1]
                    # remove the \n from the head and the space
                    rule_head = rule_head.strip()
                    # first element is the body of the rule
                    rule_body = rule[0]
                    # split the body by ,
                    rule_body = rule_body.split(', ')
                    for i in range(len(rule_body)):
                        if rule_body[i][-1] == " ":
                            rule_body[i] = rule_body[i][:-1]
                    # rule_body = re.findall(r'\w+\(.*?\)', rule_body)
                    rule_body = [get_atom_from_string(b) for b in rule_body]
                    head_atom = get_atom_from_string(rule_head)
                    # put all the arguments in capital letters so that they are recognised as variables
                    for i in range(len(rule_body)):
                        rule_body[i].args = [arg.upper() for arg in rule_body[i].args]
                    for i in range(len(head_atom.args)):
                        head_atom.args[i] = head_atom.args[i].upper()
                    rules.append(Rule(head_atom, rule_body))
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_path} not found")
    return rules

def get_queries(path: str, non_provable_queries: bool = True) -> List[Term]:
    """
    Get queries and labels (whether they are provable) from {set}_label.txt file.
    
    Args:
        path: Path to the file containing queries and labels
        non_provable_queries: If True, include all queries regardless of provability
        
    Returns:
        List of Term objects representing queries
    """
    queries = []
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            # print(line)
            if non_provable_queries:
                query = line.strip()
                # print('query',query)
                # print('q',get_atom_from_string(query).args[1])
                queries.append(get_atom_from_string(query))
            else:
                query, label = line.strip().split("\t")
                if label == "True" or non_provable_queries:
                    queries.append(get_atom_from_string(query))
    # print('\nqueries', queries[:50])
    # print(sdhvb)
    return queries

def get_filtered_queries(path: str, depth: str) -> List[Term]:
    """Get queries from a file, filtering by depth: query depth
    If the depth of a query is -1, it is not provable
    Args:
        depth: depth of the query. Can be 'number' or '<number'. In the last
          case, take all the queries with depth < int. If None, take all the queries
    """


    queries = []
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            query = line.strip().split(" ")[0]
            query_depth = line.strip().split(" ")[1]
            query_depth = int(query_depth) if query_depth != "-1" else -1
            if depth == None or (depth.startswith('<') and query_depth < int(depth[1:]) and query_depth>=0)\
                or (depth.startswith('>') and query_depth > int(depth[1:]))  or (depth == str(query_depth)):
                queries.append(get_atom_from_string(query))
    print(f"Number of queries with depth {depth}: {len(queries)}/ {len(lines)}")
    return queries



def get_corruptions_dict(file_path: str, non_provable_corruptions: bool = False) -> Dict[Term, List[Term]]:
    """
    Get corruptions dictionary from the json file, in the format
    {"locatedInCR(armenia,asia).": [["locatedInCR(armenia,oceania).", true], ["locatedInCR(armenia,europe).", true], ["locatedInCR(armenia,africa).", true], [
    
    Args:
        file_path: Path to the JSON file containing corruptions
        non_provable_corruptions: If True, include corruptions regardless of provability
        
    Returns:
        Dictionary with the query as key and a list of corruptions
    """
    dict_ = defaultdict(list)
    try:
        with open(file_path, "r") as f:
            corruptions_dict = json.load(f)
            for query, corruptions in corruptions_dict.items():
                query = get_atom_from_string(query[:-1])
                for corruption, is_provable in corruptions:
                    if is_provable or non_provable_corruptions:
                        corruption = get_atom_from_string(corruption[:-1])
                        dict_[query].append(corruption)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_path} not found")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in {file_path}")
    return dict_


class DataHandler:
    '''
    Handles knowledge graph embedding datasets with corruption support.
    
    Instead of the normal test,valid txt files, we load a file with the corruptions.
    Each query will have a label indicating true or false, i.e. if the query is a corruption or not.
    For each set, we will have queries and labels, instead of just queries.
    For train, we will load facts from the original train file, and train queries in the same way as valid, test.
    '''
    def __init__(self, dataset_name: str,
                base_path: str,
                janus_file: str = None,
                train_file: str = None,
                valid_file: str = None,
                test_file: str = None,
                rules_file: str = None,
                facts_file: str = None,
                n_eval_queries: int = None,
                n_test_queries: int = None,
                corruption_mode: Optional[str] = None,
                non_provable_corruptions: bool = False,
                non_provable_queries: bool = False,
                train_depth: str = None,
                valid_depth: str = None,
                test_depth: str = None):
        """
        Initialize KGE data handler.
        
        Args:
            dataset_name: Name of the dataset
            base_path: Base path to the dataset directory 
            janus_file: Name of the Janus prolog file
            train_file: Name of the training file
            valid_file: Name of the validation file
            test_file: Name of the test file
            corruption_mode: Mode for handling corruptions ("static" or "dynamic")
            non_provable_corruptions: Whether to include non-provable corruptions
            non_provable_queries: Whether to include non-provable queries
        """
        self.dataset_name = dataset_name
        
        base_path = join(base_path, dataset_name)
        janus_path = join(base_path, janus_file) if janus_file else None
        self.janus_path = janus_path
        if janus_file:
            try:
                janus.consult(janus_path)
            except Exception as e:
                raise RuntimeError(f"Failed to consult Janus file: {e}")

        train_path = join(base_path, train_file) 
        valid_path = join(base_path, valid_file) if valid_file else None
        test_path = join(base_path, test_file)
        rules_file = join(base_path, rules_file) if rules_file else None
        facts_file = join(base_path, facts_file) if facts_file else None

        if janus_file:
            self.facts, self.rules = get_rules_from_file(janus_path,)
        else:
            self.facts = get_queries(facts_file)
            self.rules = get_rules_from_rules_file(rules_file)

        if corruption_mode and 'static' in corruption_mode:
            self.train_corruptions = get_corruptions_dict(train_path, non_provable_corruptions)
            self.valid_corruptions = get_corruptions_dict(valid_path, non_provable_corruptions)
            self.test_corruptions = get_corruptions_dict(test_path, non_provable_corruptions)

            # renamed from pos_train_queries to train_queries to be consistent with valid and test
            # we dont need neg_train_queries, as we have the corruptions in the train_corruptions that we use in evaluation
            self.train_queries, self.neg_train_queries = list(self.train_corruptions.keys()), list(self.train_corruptions.values())
            self.neg_train_queries = [item for sublist in self.neg_train_queries for item in sublist]
            self.valid_queries = list(self.valid_corruptions.keys())
            self.test_queries = list(self.test_corruptions.keys())

        elif corruption_mode and 'dynamic' in corruption_mode:
            self.train_corruptions = self.valid_corruptions = self.test_corruptions = self.neg_train_queries = None
            if not train_depth:
                self.train_queries = get_queries(train_path, non_provable_queries)
            else:
                self.train_queries = get_filtered_queries(train_path, train_depth)
            if not valid_depth:
                self.valid_queries = get_queries(valid_path, non_provable_queries)
            else:
                self.valid_queries = get_filtered_queries(valid_path, valid_depth)
            if not test_depth:
                self.test_queries = get_queries(test_path, non_provable_queries)
            else:
                self.test_queries = get_filtered_queries(test_path, test_depth)

               
            # Filter queries with predicates not in the rules
            rules_head_predicates = set(rule.head.predicate for rule in self.rules)
            
            # Calculate exclusion statistics
            exclude_train = len([q for q in self.train_queries if q.predicate not in rules_head_predicates])
            exclude_valid = len([q for q in self.valid_queries if q.predicate not in rules_head_predicates])    
            exclude_test = len([q for q in self.test_queries if q.predicate not in rules_head_predicates])
            
            # Log exclusion information
            if exclude_train > 0 and self.train_queries:
                print(f"Number of train queries excluded: {exclude_train}. Ratio excluded: {round(exclude_train/len(self.train_queries),3)}")
            if exclude_valid > 0 and self.valid_queries:
                print(f"Number of valid queries excluded: {exclude_valid}. Ratio excluded: {round(exclude_valid/len(self.valid_queries),3)}")
            if exclude_test > 0 and self.test_queries:
                print(f"Number of test queries excluded: {exclude_test}. Ratio excluded: {round(exclude_test/len(self.test_queries),3)}")
            
            # Filter the queries
            self.train_queries = [q for q in self.train_queries if q.predicate in rules_head_predicates]
            self.valid_queries = [q for q in self.valid_queries if q.predicate in rules_head_predicates]
            self.test_queries = [q for q in self.test_queries if q.predicate in rules_head_predicates]
            # test = self.test_queries.copy()
            # self.test_queries = [q for q in self.valid_queries if q.predicate in rules_head_predicates]
            # self.valid_queries = [q for q in test if q.predicate in rules_head_predicates]

            # np.random.seed(42)
            # self.test_queries = np.random.choice(self.test_queries, 50, replace=False).tolist()
            self.valid_queries = self.valid_queries[:n_eval_queries]
            self.test_queries = self.test_queries[:n_test_queries]

        # Load Janus facts
        if janus_file:
            self.janus_facts = []
            with open(janus_path, "r") as f:
                self.janus_facts = f.readlines()
        else:
            self.janus_facts = None

        # Extract predicates, constants, variables
        self.predicates, self.predicates_arity, self.constants, self.variables = get_predicates_and_arguments(
            self.rules, self.facts)
        self.max_arity = get_max_arity(janus_path) if janus_file else 2
        self.constant_no = len(self.constants)
        self.predicate_no = len(self.predicates)
        self.variable_no = len(self.variables)

        # Setup domain mapping for countries dataset
        self.entity2domain = None
        self.domain2entity = None
        if corruption_mode == "dynamic":
            if 'countries' in self.dataset_name or 'ablation' in self.dataset_name:
                # Load the domain file
                domain_file = join(base_path, "domain2constants.txt")
                try:
                    self.entity2domain = {}
                    self.domain2entity = defaultdict(list)
                    with open(domain_file, "r") as f:
                        lines = f.readlines()
                        for line in lines:
                            line = line.strip()
                            if line:
                                domain, *entities = line.split()
                                for entity in entities:
                                    self.entity2domain[entity] = domain
                                    self.domain2entity[domain].append(entity)
                except FileNotFoundError:
                    print(f"Warning: Domain file {domain_file} not found")

        # print(f"\n\nlast train queries: {len(self.train_queries)}, {self.train_queries[:50]}")
        # print(f"\n\nvalid queries: {len(self.valid_queries)}, {self.valid_queries}")
        # print(f"\n\ntest queries: {len(self.test_queries)}, {self.test_queries}")
        # print(f"\n\nfacts: {len(self.facts)}, {self.facts}")
        # print(f"\n\nrules: {len(self.rules)}, {self.rules}")
        # print(f"\n\nconstants: {len(self.constants)}, {self.constants}")
        # print(f"\n\npredicates: {len(self.predicates)}, {self.predicates}")
        # print(f"variables: {len(self.variables)}, {self.variables}")
        # print(f"max arity: {self.max_arity}")
        # print(sòdjn)






















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




# def get_corruptions(queries: List[Term], file_path: str) -> Dict[Term, List[Term]]:
#     """
#     Get corruptions from the json file.
    
#     Args:
#         queries: List of valid queries
#         file_path: Path to the JSON file containing corruptions
        
#     Returns:
#         Dictionary with the query as key and a list of corruptions that are true as value
#     """
#     dict_ = defaultdict(list)
#     try:
#         with open(file_path, "r") as f:
#             corruptions_dict = json.load(f)
#             for query, corruptions in corruptions_dict.items():
#                 query = get_atom_from_string(query[:-1])
#                 assert query in queries, f"Query {query} not in queries"
#                 for corruption, is_provable in corruptions:
#                     if is_provable:
#                         corruption = get_atom_from_string(corruption[:-1])
#                         dict_[query].append(corruption)
#     except FileNotFoundError:
#         raise FileNotFoundError(f"File {file_path} not found")
#     except json.JSONDecodeError:
#         raise ValueError(f"Invalid JSON format in {file_path}")
#     return dict_

 

# # Import MNIST-related code
# # from data.mnist_addition.MNIST import addition, multiplication

# class DataHandlerMnist:
#     """Handles MNIST data with corruption support in a simplified structure"""
    
#     def __init__(self, dataset_name: str, base_path: str, janus_file: str, name: str = None):
#         """
#         Initialize MNIST data handler.
        
#         Args:
#             dataset_name: Name of the dataset
#             base_path: Base path to the dataset directory
#             janus_file: Name of the Janus prolog file
#             name: Optional alternative name for the dataset
#         """
#         self.dataset_name = name
#         self.corruption_mode = None
#         self.n_digits = 2

#         # Basic setup
#         self.base_path = join(base_path, dataset_name)
#         self.janus_path = join(self.base_path, janus_file)
#         self.max_arity = 5

#         # Load data
#         self._load_janus_data()
#         self._load_datasets()
#         self._process_datasets()
#         self._setup_constants()
        
#     def _load_janus_data(self):
#         """Load Janus facts and rules"""
#         janus.consult(self.janus_path)
#         with open(self.janus_path, "r") as f:
#             self.janus_facts = f.readlines()
#         self.facts, self.rules = get_rules_from_file(self.janus_path)

#     def _load_datasets(self):
#         """Load all datasets at once"""
#         self.datasets = {
#             'train': addition(n=self.n_digits, dataset="train", seed=42),
#             'valid': addition(n=self.n_digits, dataset="val", seed=42),
#             'test': addition(n=self.n_digits, dataset="test", seed=42)
#         }

#     def _process_datasets(self):
#         """Process all datasets similarly"""
#         queries = {}
#         self.images = defaultdict(list)
#         self.image_strings = {}
#         d = 0  # Image counter
        
#         for set_name in ['train', 'valid', 'test']:
#             # Store raw data
#             labels, digits, l1, l2 = [], [], [], []
#             for i in range(10):  # 10 samples per set
#                 dl1, dl2, label, dgts = self.datasets[set_name][i]
#                 labels.append(label)
#                 digits.append(dgts)
#                 l1.append(dl1)
#                 l2.append(dl2)
            
#             # Create image strings
#             img_strings = []
#             for dgts in digits:
#                 group = []
#                 for digit in dgts:
#                     group.extend([f"im_{d+i}" for i in range(len(digit))])
#                     d += len(digit)
#                 img_strings.append(group)
            
#             # Create queries
#             queries[set_name] = [
#                 Term("addition", [*imgs, str(label)])
#                 for imgs, label in zip(img_strings, labels)
#             ]
#             # Store queries
#             setattr(self, f'{set_name}_queries', queries[set_name])
#             # Store processed data
#             setattr(self, f'{set_name}_labels', labels)

#             self.image_strings[set_name] = img_strings
            
#             # Store images
#             im_dict = defaultdict(list)
#             for dgts, dl1, dl2 in zip(digits, l1, l2):
#                 im_dict[str(dgts)] = (dl1, dl2)
#             self.images.update(im_dict)

#     def _setup_constants(self):
#         """Setup constants and predicates"""
#         # Collect all image IDs
#         all_images = []
#         for set_name in ['train', 'valid', 'test']:
#             all_images.extend([img for group in self.image_strings[set_name] for img in group])
        
#         # Collect all labels
#         all_labels = []
#         for set_name in ['train', 'valid', 'test']:
#             all_labels.extend(map(str, getattr(self, f'{set_name}_labels')))
        
#         # Set constants
#         self.constants = set(all_images + all_labels)
#         self.constants_images = set(all_images)
        
#         # Predicate setup
#         self.predicates = {"addition", "digit"}
#         self.predicates_arity = {"addition": 3, "digit": 2}
#         self.variables = set(['RULE0_Z','RULE0_Y', 'RULE0_X', 'RULE0_Y2', 'RULE0_X2'])
#         self.constant_no = len(self.constants)
#         self.constant_images_no = len(self.constants_images)
#         self.predicate_no = len(self.predicates)
#         self.variable_no = len(self.variables)

#     def _print_debug_info(self):
#         """Print debugging information about loaded data"""
#         print(f"\nTrain queries: {len(self.train_queries)}, {self.train_queries}")
#         print(f"\nValid queries: {len(self.valid_queries)}, {self.valid_queries}")
#         print(f"\nTest queries: {len(self.test_queries)}, {self.test_queries}")
#         print(f"\nImages: {len(self.images)}, {[(i, c[0].shape, c[1].shape) for i, c in self.images.items()]}")
#         print(f"\nConstant images: {self.constant_images_no}, {self.constants_images}")
#         print(f"\nConstants: {self.constant_no}, {self.constants}")
#         print(f"\nPredicates: {self.predicate_no}, {self.predicates}")
#         print(f"\nVariables: {self.variable_no}, {self.variables}")
#         print(f"\nMax arity: {self.max_arity}")








# class DataHandler:
#     """
#     Factory class that creates appropriate data handler based on dataset name.
    
#     Creates DataHandlerMnist for mnist_addition dataset, and DataHandlerKGE otherwise.
#     """
#     def __init__(self, dataset_name: str, 
#                  base_path: str, 
#                  janus_file: str, 
#                  train_file: str = None, 
#                  valid_file: str = None, 
#                  test_file: str = None, 
#                  rules_file: str = None,
#                  facts_file: str = None,
#                  n_eval_queries: int = None,
#                  n_test_queries: int = None,
#                  corruption_mode: Optional[str] = None, 
#                  name: str = None,
#                  non_provable_corruptions: bool = False,
#                  non_provable_queries: bool = False):
#         """
#         Initialize data handler.
        
#         Args:
#             dataset_name: Name of the dataset
#             base_path: Base path to the dataset directory
#             janus_file: Name of the Janus prolog file
#             train_file: Name of the training file
#             valid_file: Name of the validation file
#             test_file: Name of the test file
#             corruption_mode: Mode for handling corruptions
#             name: Optional alternative name for the dataset
#             non_provable_corruptions: Whether to include non-provable corruptions
#             non_provable_queries: Whether to include non-provable queries
#         """
#         if dataset_name == "mnist_addition":
#             self.info = DataHandlerMnist(
#                 dataset_name=dataset_name,
#                 base_path=base_path,
#                 janus_file=janus_file,
#                 name=name
#             )
#         else:
#             self.info = DataHandlerKGE(
#                 dataset_name=dataset_name,
#                 base_path=base_path,
#                 janus_file=janus_file,
#                 train_file=train_file,
#                 valid_file=valid_file,
#                 test_file=test_file,
#                 rules_file=rules_file,
#                 facts_file=facts_file,
#                 n_eval_queries=n_eval_queries,
#                 n_test_queries=n_test_queries,
#                 corruption_mode=corruption_mode,
#                 non_provable_corruptions=non_provable_corruptions,
#                 non_provable_queries=non_provable_queries
#             )


