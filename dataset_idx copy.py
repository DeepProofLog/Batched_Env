from os.path import join
import janus_swi as janus
from typing import List, Tuple, Dict, Optional, Set, FrozenSet
import re
from collections import defaultdict
from utils import is_variable, Term, Rule, get_atom_from_string, get_rule_from_string # Assuming these are compatible

import torch
# Import the new IndexManager and tensor conversion functions
from index_manager_idx import IndexManager
from python_unification_idx import facts_to_tensor_im, rules_to_tensor_im, state_to_tensor_im # Added state_to_tensor_im for potential use

CLAUSE_FINDER = re.compile(r'\w+\((.*?)\)') # Only capture content inside parentheses

def get_max_arity(file_path: str) -> int:
    """Optimized version using line-by-line iteration and simplified parsing."""
    max_arity = 0
    try:
        with open(file_path, "r") as f:
            for line in f: # Iterate line by line
                # --- Early exit/skip conditions ---
                if line.startswith("one_step_list"):
                    break # Assuming we don't need to read past this
                line = line.strip() # Strip whitespace once
                if not line or line.startswith('%'): # Skip empty/comment lines
                     continue

                # --- Find potential arguments content ---
                for match in CLAUSE_FINDER.finditer(line):
                    args_content = match.group(1)
                    if not args_content: 
                        arity = 0 
                    else:
                        arity = args_content.count(',') + 1
                    max_arity = max(max_arity, arity)

    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_path} not found")
    return max_arity


def get_predicates_and_arguments(rules: List[Rule], facts: List[Term]) -> Tuple[Set[str], Dict[str, int], Set[str], Set[str]]:
    """
    Extract predicates, their arities, constants, and variables from rules and facts.
    (This function can remain as is, as it operates on Term/Rule objects before tensorization)
    """
    predicates = set()
    constants = set()
    variables = set()
    predicates_arity = {}
    
    for i, rule in enumerate(rules):
        if rule.head.predicate == "proof_first":
            continue
            
        predicate = rule.head.predicate
        predicates.add(predicate)
        for arg in rule.head.args:
            if is_variable(arg):
                variables.add(f"RULE{i}_{arg}")
            else:
                constants.add(arg)
        if predicate not in predicates_arity:
            predicates_arity[predicate] = len(rule.head.args)
            
        for atom in rule.body:
            predicates.add(atom.predicate)
            for arg in atom.args:
                if is_variable(arg):
                    variables.add(f"RULE{i}_{arg}")
                else:
                    constants.add(arg)
            if atom.predicate not in predicates_arity:
                predicates_arity[atom.predicate] = len(atom.args)
    
    for atom in facts:
        predicates.add(atom.predicate)
        constants.update([arg for arg in atom.args if not is_variable(arg)])
        if atom.predicate not in predicates_arity:
            predicates_arity[atom.predicate] = len(atom.args)
            
    return predicates, predicates_arity, constants, variables


def get_rules_from_janus_file(file_path: str) -> Tuple[List[Term], List[Rule]]:
    """
    Parse a file to extract facts (queries) and rules.
    (This function can remain as is)
    """
    rules = []
    queries = []
    
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                if not line or line.startswith("%") or line.startswith(":-"):
                    continue
                if line.startswith("one_step_list"):
                    break
                if ":-" not in line: 
                    head = line.strip()
                    queries.append(get_atom_from_string(head))
                else:
                    head, body = line.strip().split(":-")
                    body = re.findall(r'\w+\(.*?\)', body)
                    body = [get_atom_from_string(b) for b in body]
                    head_atom = get_atom_from_string(head)
                    rules.append(Rule(head_atom, body))
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_path} not found")
        
    rules = [rule for rule in rules if rule.head.predicate != "proof_first"]
    return queries, rules


def get_rules_from_rules_file(file_path: str) -> List[Rule]:
    """
    Parse a file to extract rules. Handles commas within arguments.
    (This function can remain as is)
    """
    rules = []
    atom_pattern = re.compile(r'[a-zA-Z0-9_]+\(.*?\)|[a-zA-Z0-9_]+')

    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line_num, line in enumerate(lines, 1): 
                line = line.strip()
                if not line or line.startswith("%"):
                    continue
                else:
                    line_parts = line.split(':')
                    if len(line_parts) < 3:
                        print(f"Warning: Skipping malformed line {line_num} (missing ':'): {line}")
                        continue

                    rule_parts = line_parts[2].split('->')
                    if len(rule_parts) != 2:
                        print(f"Warning: Skipping malformed rule on line {line_num} (missing '->'): {line_parts[2]}")
                        continue

                    rule_head_str = rule_parts[1].strip()
                    rule_body_str = rule_parts[0].strip()
                    rule_body_strs = atom_pattern.findall(rule_body_str)
                    rule_body_strs = [atom.strip() for atom in rule_body_strs if atom.strip()]

                    try:
                        initial_body_atoms = [get_atom_from_string(b) for b in rule_body_strs]
                        initial_head_atom = get_atom_from_string(rule_head_str)
                        updated_body_atoms = []
                        for atom in initial_body_atoms:
                            upper_args = tuple(arg.upper() for arg in atom.args)
                            updated_atom = type(atom)(predicate=atom.predicate, args=upper_args)
                            updated_body_atoms.append(updated_atom)
                        upper_head_args = tuple(arg.upper() for arg in initial_head_atom.args)
                        updated_head_atom = type(initial_head_atom)(predicate=initial_head_atom.predicate, args=upper_head_args)
                        rules.append(Rule(updated_head_atom, updated_body_atoms))
                    except ValueError as ve:
                        print(f"Error parsing atoms on line {line_num}: {ve}")
                        continue 
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_path} not found")
    except Exception as e:
        print(f"An unexpected error occurred while processing the rules file: {e}")
        raise
    return rules


def get_queries(path: str) -> List[Term]:
    """
    (This function can remain as is)
    """
    queries = []
    with open(path, "r") as f:
        for line in f: 
            line = line.strip()
            if line: 
                queries.append(get_atom_from_string(line))
    return queries

def get_filtered_queries(path: str,
                         depth: Optional[Set[int]], 
                         name: str) -> List[Term]:
    """
    (This function can remain as is)
    """
    queries = []
    try:
         with open(path, "r") as f:
              lines_count = 0 
              for line in f: 
                   lines_count += 1
                   line = line.strip()
                   if not line: continue
                   parts = line.split(" ", 1) 
                   if len(parts) == 2:
                        query_str = parts[0]
                        depth_str = parts[1]
                        try:
                             query_depth = int(depth_str) if depth_str != "-1" else -1
                             if depth is None or query_depth in depth:
                                  queries.append(get_atom_from_string(query_str))
                        except ValueError:
                             print(f"Warning: Skipping line in {name} due to non-integer depth: {line}")
                        except IndexError:
                             print(f"Warning: Skipping line in {name} due to missing depth: {line}")
                   else: 
                       print(f"Warning: Skipping malformed line in {name} (expected 'query depth'): {line}")
    except FileNotFoundError:
         raise FileNotFoundError(f"File {path} not found")
    print(f"Number of queries with depth {depth} in {name}: {len(queries)} / {lines_count}")
    return queries


class DataHandler:
    def __init__(self, dataset_name: str,
                base_path: str,
                # --- New arguments for tensor processing ---
                index_manager: IndexManager, # Expecting an instance of the new IndexManager
                max_rule_atoms: int, # Max atoms in a rule (head + body) for padding rules_tensor
                device: torch.device = torch.device("cpu"),
                # --- End new arguments ---
                janus_file: str = None,
                train_file: str = None,
                valid_file: str = None,
                test_file: str = None,
                rules_file: str = None,
                facts_file: str = None,
                n_eval_queries: int = None,
                n_test_queries: int = None,
                corruption_mode: Optional[str] = None,
                train_depth: Optional[Set] = None,
                valid_depth: Optional[Set] = None,
                test_depth: Optional[Set] = None):

        self.dataset_name = dataset_name
        self.index_manager = index_manager # Use the passed IndexManager (from index_manager_idx)
        self.device = device
        
        base_path = join(base_path, dataset_name)
        janus_path = join(base_path, janus_file) if janus_file else None
        self.janus_path = janus_path
        if janus_file:
            try:
                janus.consult(janus_path)
            except Exception as e:
                raise RuntimeError(f"Failed to consult Janus file: {e}")

        train_path = join(base_path, train_file) 
        valid_path = join(base_path, valid_file)
        test_path = join(base_path, test_file)
        rules_file_path = join(base_path, rules_file) if rules_file else None # Renamed to avoid conflict
        facts_file_path = join(base_path, facts_file) if facts_file else None # Renamed

        # Load facts and rules as Term/Rule objects first
        self.facts_terms: List[Term] = get_queries(facts_file_path)
        self.rules_objects: List[Rule] = get_rules_from_rules_file(rules_file_path)

        # --- Tensor Conversion ---
        # Convert facts to tensor and set of indexed tuples
        self.facts_tensor = facts_to_tensor_im(self.facts_terms, self.index_manager)
        self.facts_as_set_indices: FrozenSet[Tuple[int, int, int]] = frozenset(
            tuple(f.tolist()) for f in self.facts_tensor
        )
        
        # Convert rules to tensor
        # max_rule_atoms needs to be determined or passed. For now, it's an argument.
        # If not passed, it could be calculated: max(1 + len(r.body) for r in self.rules_objects) if self.rules_objects else 0
        self.rules_tensor, self.rule_lengths_tensor = rules_to_tensor_im(
            self.rules_objects, max_rule_atoms, self.index_manager
        )
        # --- End Tensor Conversion ---

        self.train_corruptions = self.valid_corruptions = self.test_corruptions = self.neg_train_queries = None
        if not train_depth:
            self.train_queries = get_queries(train_path)
        else:
            self.train_queries = get_filtered_queries(train_path, train_depth, "train")
        if not valid_depth:
            self.valid_queries = get_queries(valid_path)
        else:
            self.valid_queries = get_filtered_queries(valid_path, valid_depth, "valid")
        if not test_depth:
            self.test_queries = get_queries(test_path)
        else:
            self.test_queries = get_filtered_queries(test_path, test_depth, "test")
               
        rules_head_predicates = set(rule.head.predicate for rule in self.rules_objects)
        
        exclude_train = len([q for q in self.train_queries if q.predicate not in rules_head_predicates])
        exclude_valid = len([q for q in self.valid_queries if q.predicate not in rules_head_predicates])    
        exclude_test = len([q for q in self.test_queries if q.predicate not in rules_head_predicates])
        
        if exclude_train > 0 and self.train_queries:
            print(f"Number of train queries excluded: {exclude_train}. Ratio excluded: {round(exclude_train/len(self.train_queries),3)}")
        if exclude_valid > 0 and self.valid_queries:
            print(f"Number of valid queries excluded: {exclude_valid}. Ratio excluded: {round(exclude_valid/len(self.valid_queries),3)}")
        if exclude_test > 0 and self.test_queries:
            print(f"Number of test queries excluded: {exclude_test}. Ratio excluded: {round(exclude_test/len(self.test_queries),3)}")
        
        self.train_queries = [q for q in self.train_queries if q.predicate in rules_head_predicates]
        self.valid_queries = [q for q in self.valid_queries if q.predicate in rules_head_predicates]
        self.test_queries = [q for q in self.test_queries if q.predicate in rules_head_predicates]

        self.valid_queries = self.valid_queries[:n_eval_queries]
        self.test_queries = self.test_queries[:n_test_queries]

        if janus_file:
            self.janus_facts_str: List[str] = [] # Store as strings
            with open(janus_path, "r") as f:
                self.janus_facts_str = f.readlines()
        else:
            self.janus_facts_str = None

        # These are derived from IndexManager, which should be initialized with all constants/predicates
        self.predicates = self.index_manager.predicates 
        self.constants = self.index_manager.constants
        # self.variables = ... # IndexManager handles variables internally now
        self.predicates_arity = {} # Can be populated if needed, or rely on IndexManager's max_arity
        for p_str, p_idx in self.index_manager.predicate_str2idx.items():
            # This is a simplification; true arity might need parsing from rules/facts if not fixed
            # For now, let's assume IndexManager's max_arity is sufficient or arity is checked elsewhere
            related_rules = [r for r in self.rules_objects if r.head.predicate == p_str]
            related_facts = [f for f in self.facts_terms if f.predicate == p_str]
            if related_rules:
                self.predicates_arity[p_str] = len(related_rules[0].head.args)
            elif related_facts:
                 self.predicates_arity[p_str] = len(related_facts[0].args)
            # else:
                # self.predicates_arity[p_str] = self.index_manager.max_arity # Default if not found

        self.max_arity = self.index_manager.max_arity
        self.constant_no = self.index_manager.constant_no
        self.predicate_no = self.index_manager.predicate_no
        # self.variable_no = self.index_manager.variable_no # Fixed + dynamic vars

        self.entity2domain = None
        self.domain2entity = None
        if corruption_mode:
            if 'countries' in self.dataset_name or 'ablation' in self.dataset_name:
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
