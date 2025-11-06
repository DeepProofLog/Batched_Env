import os
from os.path import join 
from typing import List, Tuple, Dict, Optional, Set, Union
import re
import ast
from collections import defaultdict

# import janus_swi as janus
janus = None 

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
    atom_pattern = re.compile(r'[\w/.-]+\([^)]*\)')

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

def get_filtered_queries(
    path: str,
    depth: Optional[Set[int]],
    name: str,
    return_depths: bool = False,
) -> Union[List[Term], Tuple[List[Term], List[Optional[int]]]]:
    queries: List[Term] = []
    depths: List[Optional[int]] = []
    lines_count = 0
    try:
        with open(path, "r") as f:
            for line in f:
                lines_count += 1
                line = line.strip()
                if not line:
                    continue

                parts = line.split(" ", 1)
                if len(parts) == 2:
                    query_str, depth_str = parts
                    try:
                        query_depth = int(depth_str) if depth_str != "-1" else -1
                        if depth is None or query_depth in depth:
                            queries.append(get_atom_from_string(query_str))
                            depths.append(query_depth)
                    except ValueError:
                        print(f"Warning: Skipping line in {name} due to non-integer depth: {line}")
                    except IndexError:
                        print(f"Warning: Skipping line in {name} due to missing depth: {line}")
                else:
                    print(f"Warning: Skipping malformed line in {name} (expected 'query depth'): {line}")
    except FileNotFoundError:
        raise FileNotFoundError(f"File {path} not found")

    print(f"Number of queries with depth {depth} in {name}: {len(queries)} / {lines_count}")

    if return_depths:
        return queries, depths
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
                n_train_queries: int = None,
                n_eval_queries: int = None,
                n_test_queries: int = None,
                corruption_mode: Optional[str] = None,
                train_depth: Optional[Set[int]] = None,
                valid_depth: Optional[Set[int]] = None,
                test_depth: Optional[Set[int]] = None,
                prob_facts: bool = False,
                topk_facts: Optional[int] = None,
                topk_facts_threshold: Optional[float] = None):
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
            n_train_queries (int, optional): Max number of training queries to keep.
            n_eval_queries (int, optional): Max number of validation queries to keep.
            n_test_queries (int, optional): Max number of test queries to keep.
            corruption_mode (str, optional): "static" or "dynamic" corruption handling.
            train_depth (Optional[Set[int]]): Depth filter for training queries.
            valid_depth (Optional[Set[int]]): Depth filter for validation queries.
            test_depth (Optional[Set[int]]): Depth filter for test queries.
        """
        self.dataset_name = dataset_name

        base_path = join(base_path, dataset_name)
        janus_path = join(base_path, janus_file) if janus_file else None
        self.janus_path = janus_path
        if janus_file:
            if janus is None:
                print(f"Warning: janus_swi module not available; skipping consult of {janus_file}.")
            else:
                try:
                    janus.consult(janus_path)
                except Exception as e:
                    raise RuntimeError(f"Failed to consult Janus file: {e}")

        train_path = join(base_path, train_file)
        valid_path = join(base_path, valid_file)
        test_path = join(base_path, test_file)
        rules_file = join(base_path, rules_file) if rules_file else None
        facts_file = join(base_path, facts_file) if facts_file else None

        self.facts = get_queries(facts_file)
        self._probabilistic_facts: List[Term] = []

        def _resolve_prob_facts_path(ds_name: str) -> Optional[str]:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            candidate = os.path.join(base_dir, "top_k_scores", f"kge_top_{ds_name}.txt")
            return candidate if os.path.exists(candidate) else None

        def _load_probabilistic_facts(file_path: str,
                                       topk_limit: Optional[int],
                                       score_threshold: Optional[float]) -> List[Term]:
            loaded: List[Term] = []
            seen: Set[Term] = set()
            with open(file_path, "r", encoding="ascii") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue

                    parts = line.split()
                    if len(parts) < 2:
                        continue

                    fact_repr = parts[0]
                    try:
                        score = float(parts[1])
                    except ValueError:
                        continue

                    rank: Optional[int] = None
                    if len(parts) >= 3:
                        try:
                            rank = int(parts[2])
                        except ValueError:
                            rank = None

                    if topk_limit is not None and topk_limit >= 0 and rank is not None and rank > topk_limit:
                        continue
                    if score_threshold is not None and score < score_threshold:
                        continue

                    try:
                        fact = get_atom_from_string(fact_repr)
                    except ValueError:
                        continue

                    if fact in seen:
                        continue
                    seen.add(fact)
                    loaded.append(fact)
            print(f"Loaded {len(loaded)} probabilistic facts (topk={topk_limit}, threshold={score_threshold}).")
            return loaded

        if prob_facts:
            prob_facts_path = _resolve_prob_facts_path(self.dataset_name)
            if prob_facts_path is None:
                print(
                    f"Warning: Requested probabilistic facts but file kge_top_{self.dataset_name}.txt was not found.")
            else:
                self._probabilistic_facts = _load_probabilistic_facts(
                    prob_facts_path,
                    topk_limit=topk_facts,
                    score_threshold=topk_facts_threshold,
                )
                if self._probabilistic_facts:
                    existing_facts = set(self.facts)
                    new_fact_terms = [fact for fact in self._probabilistic_facts if fact not in existing_facts]
                    if new_fact_terms:
                        self.facts.extend(new_fact_terms)
                        print(
                            f"Loaded {len(new_fact_terms)} probabilistic facts from {prob_facts_path} "
                            f"(topk={topk_facts}, threshold={topk_facts_threshold})."
                            f" Facts augmented from {len(existing_facts)} to {len(self.facts)}."
                        )
                else:
                    print(
                        f"Warning: Probabilistic facts file {prob_facts_path} produced no usable entries after filtering.")
        self.rules = get_rules_from_rules_file(rules_file)
        self.train_corruptions = self.valid_corruptions = self.test_corruptions = self.neg_train_queries = None

        def _normalize_depth_filter(depth_filter: Optional[Union[Set[int], List[int], Tuple[int, ...], str]]) -> Optional[Set[int]]:
            """Helper to normalize depth filter input into a set of integers or None.
            What it does: Converts various input types into a standardized set of integers.
            More specifically:
             - If input is None, returns None (no filtering).
             - If input is a set/list/tuple, converts elements to integers and returns as a set.
             - If input is a string, attempts to parse it as a Python literal (e.g., '{1,2,3}' or '[1,2,3]')."""

            if depth_filter is None:
                return None
            if isinstance(depth_filter, set):
                return {int(d) for d in depth_filter}
            if isinstance(depth_filter, (list, tuple)):
                return {int(d) for d in depth_filter}
            if isinstance(depth_filter, str):
                value = depth_filter.strip()
                if value.lower() in {"none", ""}:
                    return None
                try:
                    parsed = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    raise ValueError(f"Invalid depth specification: {depth_filter}")
                return _normalize_depth_filter(parsed)
            raise ValueError(f"Unsupported depth filter type: {type(depth_filter)}")

        def load_queries_with_depth(path: str, depth_filter: Optional[Union[Set[int], List[int], Tuple[int, ...], str]], name: str) -> Tuple[List[Term], List[Optional[int]]]:
            normalized_filter = _normalize_depth_filter(depth_filter)
            depth_path = None
            if path.endswith("_depths.txt") and os.path.exists(path):
                depth_path = path
            elif path.endswith(".txt"):
                candidate = path.replace(".txt", "_depths.txt")
                if os.path.exists(candidate):
                    depth_path = candidate
            if normalized_filter is not None:
                source_path = depth_path if depth_path else path
                queries, depth_values = get_filtered_queries(source_path, normalized_filter, name, return_depths=True)
            elif depth_path:
                queries, depth_values = get_filtered_queries(depth_path, None, name, return_depths=True)
            else:
                queries = get_queries(path)
                depth_values = [None] * len(queries)
            return queries, depth_values

        self.train_queries, self.train_queries_depths = load_queries_with_depth(train_path, train_depth, "train")
        if self._probabilistic_facts:
            train_existing = set(self.train_queries)
            to_append = [fact for fact in self._probabilistic_facts if fact not in train_existing]
            if to_append:
                self.train_queries.extend(to_append)
                self.train_queries_depths.extend([None] * len(to_append))
        self.valid_queries, self.valid_queries_depths = load_queries_with_depth(valid_path, valid_depth, "valid")
        self.test_queries, self.test_queries_depths = load_queries_with_depth(test_path, test_depth, "test")

               
        # Filter queries with predicates not in the rules
        rules_head_predicates = set(rule.head.predicate for rule in self.rules)
        
        # Calculate exclusion statistics
        exclude_train = len([q for q in self.train_queries if q.predicate not in rules_head_predicates])
        # exclude_valid = len([q for q in self.valid_queries if q.predicate not in rules_head_predicates])    
        # exclude_test = len([q for q in self.test_queries if q.predicate not in rules_head_predicates])
        
        # Log exclusion information
        if exclude_train > 0 and self.train_queries:
            print(f"Number of train queries excluded: {exclude_train}. Ratio excluded: {round(exclude_train/len(self.train_queries),3)}")
        # if exclude_valid > 0 and self.valid_queries:
        #     print(f"Number of valid queries excluded: {exclude_valid}. Ratio excluded: {round(exclude_valid/len(self.valid_queries),3)}")
        # if exclude_test > 0 and self.test_queries:
        #     print(f"Number of test queries excluded: {exclude_test}. Ratio excluded: {round(exclude_test/len(self.test_queries),3)}")
        
        # Filter the queries and keep depth alignment
        train_filtered = [(q, d) for q, d in zip(self.train_queries, self.train_queries_depths)
                          if q.predicate in rules_head_predicates]
        if n_train_queries is not None:
            train_filtered = train_filtered[:n_train_queries]
        if train_filtered:
            self.train_queries, self.train_queries_depths = map(list, zip(*train_filtered))
        else:
            self.train_queries, self.train_queries_depths = [], []

        self.valid_queries = self.valid_queries[:n_eval_queries]
        self.valid_queries_depths = self.valid_queries_depths[:n_eval_queries]
        self.test_queries = self.test_queries[:n_test_queries]
        self.test_queries_depths = self.test_queries_depths[:n_test_queries]

        self.all_known_triples = self.train_queries + self.valid_queries + self.test_queries
        # self.valid_queries = self.test_queries.copy() # Use test queries as valid queries

        # Debug: Check actual depth values loaded
        if self.train_queries_depths:
            depth_counts = {}
            for d in self.train_queries_depths:
                depth_counts[d] = depth_counts.get(d, 0) + 1
            print(f"Train depth distribution: {depth_counts}")
        if self.valid_queries_depths:
            depth_counts = {}
            for d in self.valid_queries_depths:
                depth_counts[d] = depth_counts.get(d, 0) + 1
            print(f"Valid depth distribution: {depth_counts}")
        
        print(f"Dataset {dataset_name} loaded:")
        print(f"  Rules: {len(self.rules)}")
        print(f"  Facts: {len(self.facts)}")
        print(f"  Train queries: {len(self.train_queries)}")
        print(f"  Valid queries: {len(self.valid_queries)}")
        print(f"  Test queries: {len(self.test_queries)}")

        # Cache an immutable facts set once to let multiple env workers reuse it via copy-on-write
        self.facts_set = frozenset(self.facts)

        # # Load Janus facts
        # if janus_file:
        #     self.janus_facts = []
        #     with open(janus_path, "r") as f:
        #         self.janus_facts = f.readlines()
        # else:
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

    def __getstate__(self):
        """Custom pickle support - exclude unpicklable sampler."""
        state = self.__dict__.copy()
        # Remove the sampler as it's not picklable and can be reconstructed
        if 'sampler' in state:
            del state['sampler']
        return state
    
    def __setstate__(self, state):
        """Custom unpickle support - sampler will be None and must be reconstructed if needed."""
        self.__dict__.update(state)
        # Sampler will need to be reconstructed by calling code if needed
        if not hasattr(self, 'sampler'):
            self.sampler = None


 

# # Import MNIST-related code
# from data.mnist_addition.MNIST import addition, multiplication

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
#         self.facts, self.rules = get_rules_from_janus_file(self.janus_path)

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
#         print(f"\nTrain queries: {len(self.
#         print(f"\nValid queries: {len(self.valid_queries)}, {self.valid_queries}")
#         print(f"\nTest queries: {len(self.test_queries)}, {self.test_queries}")
#         print(f"\nImages: {len(self.images)}, {[(i, c[0].shape, c[1].shape) for i, c in self.images.items()]}")
#         print(f"\nConstant images: {self.constant_images_no}, {self.constants_images}")
#         print(f"\nConstants: {self.constant_no}, {self.constants}")
#         print(f"\nPredicates: {self.predicate_no}, {self.predicates}")
#         print(f"\nVariables: {self.variable_no}, {self.variables}")
#         print(f"\nMax arity: {self.max_arity}")