import re
from os.path import join
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict
from dataclasses import dataclass, field
import functools
from utils import is_variable, get_atom_from_string, Rule, Term



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
    """Reads a file and parses each line as a query Term."""
    queries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(get_atom_from_string(line.split(" ")[0]))
    return queries


def get_filtered_queries(path: str, depth: Optional[Set[int]], name: str) -> List[Term]:
    """Reads a query file and filters by a set of depths."""
    queries = []
    lines_count = 0
    try:
         with open(path, "r") as f:
              for line in f:
                   lines_count += 1
                   line = line.strip()
                   if not line: continue

                   parts = line.split(" ", 1)
                   if len(parts) == 2:
                        query_str, depth_str = parts
                        try:
                             query_depth = int(depth_str)
                             if depth is None or query_depth in depth:
                                  queries.append(get_atom_from_string(query_str))
                        except (ValueError, IndexError):
                             print(f"Warning: Skipping malformed line in {name}: {line}")
                   else:
                       # If no depth is specified, and we are filtering, we skip.
                       # If not filtering (depth is None), we could include it.
                       if depth is None:
                           queries.append(get_atom_from_string(line))

    except FileNotFoundError:
         raise FileNotFoundError(f"File {path} not found")

    print(f"Number of queries with depth {depth} in {name}: {len(queries)} / {lines_count}")
    return queries


def get_predicates_and_arguments(rules: List[Rule], facts: List[Term]) -> Tuple[Set[str], Dict[str, int], Set[str], Set[str]]:
    """Extracts predicates, arities, constants, and variables from rules and facts."""
    predicates, constants, variables = set(), set(), set()
    predicates_arity = {}

    for rule in rules:
        # Process head
        predicates.add(rule.head.predicate)
        predicates_arity.setdefault(rule.head.predicate, len(rule.head.args))
        for arg in rule.head.args:
            (variables if is_variable(arg) else constants).add(arg)
        # Process body
        for atom in rule.body:
            predicates.add(atom.predicate)
            predicates_arity.setdefault(atom.predicate, len(atom.args))
            for arg in atom.args:
                (variables if is_variable(arg) else constants).add(arg)

    for fact in facts:
        predicates.add(fact.predicate)
        predicates_arity.setdefault(fact.predicate, len(fact.args))
        constants.update(arg for arg in fact.args if not is_variable(arg))

    return predicates, predicates_arity, constants, variables

# --- Main Dataset Class ---

class DataHandler:
    """
    Loads and processes knowledge graph data (rules, facts, queries) from files.
    This class reads raw data files, converts them into structured Term and Rule
    objects, and extracts key statistics like predicates, constants, and arities,
    making them available as class attributes.
    """
    def __init__(self,
                 dataset_name: str,
                 base_path: str,
                 rules_file: str,
                 facts_file: str,
                 train_file: str,
                 valid_file: Optional[str] = None,
                 test_file: Optional[str] = None,
                 n_eval_queries: Optional[int] = None,
                 n_test_queries: Optional[int] = None,
                 train_depth: Optional[Set] = None,
                 valid_depth: Optional[Set] = None,
                 test_depth: Optional[Set] = None,
                 corruption_mode: Optional[str] = None):
        """
        Initializes the dataset by loading and processing all data from files.
        """
        self.dataset_name = dataset_name
        self.base_path = join(base_path, dataset_name)

        # --- Load Raw Data from Files ---
        print("Loading rules, facts, and queries...")
        rules_path = join(self.base_path, rules_file)
        facts_path = join(self.base_path, facts_file)
        train_path = join(self.base_path, train_file)

        self.rules_terms: List[Rule] = get_rules_from_rules_file(rules_path)
        self.facts_terms: List[Term] = get_queries(facts_path)

        # Load queries with optional depth filtering
        self.train_queries_terms = get_filtered_queries(train_path, train_depth, "train") if train_depth else get_queries(train_path)

        self.valid_queries_terms: Optional[List[Term]] = None
        if valid_file:
            valid_path = join(self.base_path, valid_file)
            self.valid_queries_terms = get_filtered_queries(valid_path, valid_depth, "valid") if valid_depth else get_queries(valid_path)
            self.valid_queries_terms = self.valid_queries_terms[:n_eval_queries]

        self.test_queries_terms: Optional[List[Term]] = None
        if test_file:
            test_path = join(self.base_path, test_file)
            self.test_queries_terms = get_filtered_queries(test_path, test_depth, "test") if test_depth else get_queries(test_path)
            self.test_queries_terms = self.test_queries_terms[:n_test_queries]

        print("Data loading complete.")

        # --- Filter queries with predicates not in the rules ---
        rules_head_predicates = set(rule.head.predicate for rule in self.rules_terms)
        
        def filter_and_log(queries, name):
            if not queries: return []
            initial_count = len(queries)
            filtered = [q for q in queries if q.predicate in rules_head_predicates]
            excluded_count = initial_count - len(filtered)
            if excluded_count > 0:
                print(f"Number of {name} queries excluded (predicate not in rule heads): {excluded_count}. Ratio: {excluded_count/initial_count:.3f}")
            return filtered

        self.train_queries_terms = filter_and_log(self.train_queries_terms, "train")
        self.valid_queries_terms = filter_and_log(self.valid_queries_terms, "valid")
        self.test_queries_terms = filter_and_log(self.test_queries_terms, "test")

        # --- Extract Predicates, Constants, and Arity ---
        print("\nExtracting predicates, constants, and other stats...")
        self.predicates, self.predicates_arity, self.constants, self.variables = get_predicates_and_arguments(self.rules_terms, self.facts_terms)

        self.max_arity: int = max(self.predicates_arity.values()) if self.predicates_arity else 0
        self.constant_count: int = len(self.constants)
        self.predicate_count: int = len(self.predicates)
        self.variable_count: int = len(self.variables)
        print("Extraction complete.")

        # --- Corruption Mode Setup ---
        self.entity2domain = None
        self.domain2entity = None
        if corruption_mode == 'dynamic' and ('countries' in self.dataset_name or 'ablation' in self.dataset_name):
            print("Loading domain mapping for dynamic corruption...")
            domain_file = join(self.base_path, "domain2constants.txt")
            try:
                self.entity2domain = {}
                self.domain2entity = defaultdict(list)
                with open(domain_file, "r") as f:
                    for line in f:
                        domain, *entities = line.strip().split()
                        for entity in entities:
                            self.entity2domain[entity] = domain
                            self.domain2entity[domain].append(entity)
                print("Domain mapping loaded.")
            except FileNotFoundError:
                print(f"Warning: Domain file for corruption not found at {domain_file}")

        # --- Final Summary ---
        print("\n--- Dataset Summary ---")
        print(f"Dataset Name:    {self.dataset_name}")
        print(f"Rules loaded:      {len(self.rules_terms)}")
        print(f"Facts loaded:      {len(self.facts_terms)}")
        print(f"Train queries:     {len(self.train_queries_terms)}")
        if self.valid_queries_terms:
            print(f"Validation queries: {len(self.valid_queries_terms)}")
        if self.test_queries_terms:
            print(f"Test queries:      {len(self.test_queries_terms)}")
        print(f"Predicates found:  {self.predicate_count}")
        print(f"Constants found:   {self.constant_count}")
        print(f"Max arity:         {self.max_arity}")
        print(f"Rules: {self.rules_terms[:3]}... (showing first 3)")
        print("-----------------------\n")

if __name__ == '__main__':
    # This is an example of how to use the ProverDataset class.
    # To run this, you would need to create a dummy data folder structure.
    #
    # Example structure:
    # /tmp/
    #   my_dataset/
    #     rules.txt
    #     facts.txt
    #     train.txt
    #
    
    # Create dummy files for demonstration
    try:
        import os
        base_path = "./data"
        dataset_name = "countries_s3"
        dataset_name = "wn18rr"
        dataset_path = os.path.join(base_path, dataset_name)

        print("--- Running Example Usage ---")
        # Initialize the dataset handler
        dataset = DataHandler(
            dataset_name=dataset_name,
            base_path=base_path,
            rules_file="rules.txt",
            facts_file="train.txt",
            train_file="train.txt",
            valid_file="valid.txt",
            test_file="test.txt",
        )

        # You can now access the loaded data as attributes
        print("First rule:", dataset.rules_terms[0])
        print("First fact:", dataset.facts_terms[0])
        print("First query:", dataset.train_queries_terms[0])
        print("First constants:", list(dataset.constants)[:5])

    except Exception as e:
        print(f"\nCould not run example. Please create the necessary folder structure and files.")
        print(f"Error: {e}")