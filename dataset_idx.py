from os.path import join 
import janus_swi as janus
from typing import List, Tuple, Dict, Optional, Set
import re
import json
from collections import defaultdict
from utils import is_variable, Term, Rule, get_atom_from_string, get_rule_from_string



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
                # Use finditer for potentially better memory usage than findall on long lines
                for match in CLAUSE_FINDER.finditer(line):
                    args_content = match.group(1)
                    if not args_content: # Handles predicate() -> arity 0 or 1 based on convention
                        arity = 0 # Or 1 if 'pred()' implies arity 1 even with no args listed
                    else:
                        # Count commas and add 1 for arity
                        arity = args_content.count(',') + 1
                    max_arity = max(max_arity, arity)

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


def get_rules_from_janus_file(file_path: str) -> Tuple[List[Term], List[Rule]]:
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
                if line.startswith("one_step_list"):
                    break
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
    Parse a file to extract rules. Handles commas within arguments.

    Args:
        file_path: Path to the file containing rules

    Returns:
        List of Rule objects
    """
    rules = []
    # Regex to find atoms like predicate(arg1,arg2) or predicate
    # It matches:
    #   [a-zA-Z0-9_]+  : the predicate name
    #   \(.*?\)        : parentheses containing any characters (non-greedy)
    #   |              : OR
    #   [a-zA-Z0-9_]+  : the predicate name (for atoms without args/parentheses)
    atom_pattern = re.compile(r'[a-zA-Z0-9_]+\(.*?\)|[a-zA-Z0-9_]+')

    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line_num, line in enumerate(lines, 1): # Add line number for errors
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

                    # --- ATOM SPLITTING ---
                    # Use regex findall to extract each atom correctly
                    rule_body_strs = atom_pattern.findall(rule_body_str)
                    # Apply strip to clean up potential extra spaces captured by regex
                    rule_body_strs = [atom.strip() for atom in rule_body_strs if atom.strip()]
                    # --- End of Correction ---

                    try:
                        # Convert string representations to initial Term objects
                        initial_body_atoms = [get_atom_from_string(b) for b in rule_body_strs]
                        initial_head_atom = get_atom_from_string(rule_head_str)

                        # Create NEW Term objects with uppercase arguments for the body
                        updated_body_atoms = []
                        for atom in initial_body_atoms:
                            upper_args = tuple(arg.upper() for arg in atom.args)
                            updated_atom = type(atom)(predicate=atom.predicate, args=upper_args)
                            updated_body_atoms.append(updated_atom)

                        # Create a NEW Term object with uppercase arguments for the head
                        upper_head_args = tuple(arg.upper() for arg in initial_head_atom.args)
                        updated_head_atom = type(initial_head_atom)(predicate=initial_head_atom.predicate, args=upper_head_args)
                        rules.append(Rule(updated_head_atom, updated_body_atoms))
                        # print(f"Rule: {rules[-1]}")
                    except ValueError as ve:
                        print(f"Error parsing atoms on line {line_num}: {ve}")
                        print(f"  Problematic rule part: {line_parts[2]}")
                        print(f"  Extracted body strings: {rule_body_strs}")
                        print(f"  Extracted head string: {rule_head_str}")
                        continue # Skip this rule on error

    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_path} not found")
    except Exception as e:
        print(f"An unexpected error occurred while processing the rules file: {e}")
        raise
    return rules



def get_queries(path: str) -> List[Term]:
    queries = []
    with open(path, "r") as f:
        for line in f: # Line-by-line
            line = line.strip()
            if line: # Avoid processing empty lines
                # Use OPTIMIZED version
                queries.append(get_atom_from_string(line))
    return queries

def get_filtered_queries(path: str,
                                 depth: Optional[Set[int]], # Explicitly Set[int]
                                 name: str) -> List[Term]:
    queries = []
    try:
         with open(path, "r") as f:
              lines_count = 0 # For logging percentage
              for line in f: # Line-by-line
                   lines_count += 1
                   line = line.strip()
                   if not line: continue

                   parts = line.split(" ", 1) # Split only once, max 1 split needed
                   if len(parts) == 2:
                        query_str = parts[0]
                        depth_str = parts[1]
                        try:
                             query_depth = int(depth_str) if depth_str != "-1" else -1
                             # Check against the depth set
                             if depth is None or query_depth in depth:
                                  # Use OPTIMIZED version
                                  queries.append(get_atom_from_string(query_str))
                        except ValueError:
                             print(f"Warning: Skipping line in {name} due to non-integer depth: {line}")
                        except IndexError:
                             print(f"Warning: Skipping line in {name} due to missing depth: {line}")

                   else: # Handle lines without a depth specified? Or warn?
                       print(f"Warning: Skipping malformed line in {name} (expected 'query depth'): {line}")

    except FileNotFoundError:
         raise FileNotFoundError(f"File {path} not found")

    # Correct logging total lines count
    print(f"Number of queries with depth {depth} in {name}: {len(queries)} / {lines_count}")
    return queries





class DataHandler:
    """
    Load and prepare knowledge‐graph embedding datasets (rules, facts, queries).

    This handler can:
      - consult a Janus Prolog file
      - parse rules and facts
      - load train/valid/test queries with optional depth or provability filters
      - compute statistics and filter out queries whose predicates are not defined
      - (optionally) load domain mappings for dynamic corruptions

    Attributes:
        dataset_name (str)
        janus_path (Optional[str])
        facts (List[Term])
        rules (List[Rule])
        train_queries, valid_queries, test_queries (List[Term])
        predicates, constants, variables, predicates_arity (sets/dicts)
        max_arity, constant_no, predicate_no, variable_no (ints)
        entity2domain, domain2entity (dicts for dynamic corruption)
    """
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
                train_depth: Optional[Set] = None,
                valid_depth: Optional[Set] = None,
                test_depth: Optional[Set] = None):
        """
        Initialize dataset handler: consult Janus, read rules/facts, load and filter queries.

        Args:
            dataset_name (str): Name of the dataset subfolder.
            base_path (str): Root directory containing dataset folders.
            janus_file (str, optional): Filename of the Prolog file to consult.
            train_file (str, optional): Filename of the training queries.
            valid_file (str, optional): Filename of the validation queries.
            test_file (str, optional): Filename of the test queries.
            rules_file (str, optional): Filename of the standalone rules file.
            facts_file (str, optional): Filename of the facts file.
            n_eval_queries (int, optional): Max number of validation queries to keep.
            n_test_queries (int, optional): Max number of test queries to keep.
            corruption_mode (str, optional): "static" or "dynamic" corruption handling.
            train_depth (str, optional): Depth filter for training queries.
            valid_depth (str, optional): Depth filter for validation queries.
            test_depth (str, optional): Depth filter for test queries.
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
        valid_path = join(base_path, valid_file)
        test_path = join(base_path, test_file)
        rules_file_path = join(base_path, rules_file) if rules_file else None # Renamed to avoid conflict
        facts_file_path = join(base_path, facts_file) if facts_file else None # Renamed

        self.facts_terms: List[Term] = get_queries(facts_file_path)
        self.rules_objects: List[Rule] = get_rules_from_rules_file(rules_file_path)

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
            self.rules_objects, self.facts_terms)
        self.max_arity = get_max_arity(janus_path) if janus_file else 2
        self.constant_no = len(self.constants)
        self.predicate_no = len(self.predicates)
        self.variable_no = len(self.variables)

        # Setup domain mapping for countries dataset
        self.entity2domain = None
        self.domain2entity = None
        if corruption_mode:
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
