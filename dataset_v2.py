import logging
from os.path import join
from typing import List, Tuple, Dict, Optional, Set, Any
import re
from collections import defaultdict
from utils import Term, Rule

# # Assuming these are defined elsewhere or in utils
# class Term:
#     def __init__(self, predicate: str, args: List[any]):
#         self.predicate = predicate
#         self.args = args
#     def __str__(self): return f"{self.predicate}({','.join(map(str, self.args))})"
#     def __repr__(self): return str(self)
#     def __eq__(self, other): return isinstance(other, Term) and self.predicate == other.predicate and self.args == other.args
#     def __hash__(self):
#         try: return hash((self.predicate, tuple(self.args)))
#         except TypeError: return hash((self.predicate, tuple(map(str, self.args))))

# class Rule:
#     def __init__(self, head: Term, body: List[Term]):
#         self.head = head
#         self.body = body
#     def __str__(self): return f"{self.head} :- {', '.join(str(term) for term in self.body)}"
#     def __repr__(self): return self.__str__()

def is_variable(arg: Any) -> bool:
    return isinstance(arg, str) and arg.isupper() and len(arg) == 1

def get_atom_from_string(atom_str: str) -> Term:
    try:
        # Handle cases like 'True', 'False' which have no args
        if '(' not in atom_str:
             return Term(atom_str, [])
        predicate, args_str = atom_str.split("(", 1)
        # Remove trailing ')' and handle empty args
        args_str = args_str.rstrip(')')
        args = [arg.strip() for arg in args_str.split(",")] if args_str else []
        return Term(predicate, args)
    except Exception as e:
        raise ValueError(f"Failed to parse atom string '{atom_str}': {str(e)}")

def get_max_arity(rules: List[Rule], facts: List[Term])-> int:
    max_arity_val = 0
    for item_list in [rules, facts]:
        for item in item_list:
             if isinstance(item, Rule):
                  terms = [item.head] + item.body
             elif isinstance(item, Term):
                  terms = [item]
             else: continue # Skip unknown types

             for term in terms:
                  arity = len(term.args)
                  if arity > max_arity_val:
                       max_arity_val = arity
    return max_arity_val

def get_predicates_and_arguments(rules: List[Rule], facts: List[Term]) -> Tuple[Set[str], Dict[str, int], Set[str], Set[str]]:
    predicates = set()
    constants = set()
    variables = set()
    predicates_arity = {}

    items_to_process = rules + facts

    for item in items_to_process:
        if isinstance(item, Rule):
            terms = [item.head] + item.body
            rule_index = rules.index(item) # Get index for variable naming
            var_prefix = f"RULE{rule_index}_"
        elif isinstance(item, Term):
            terms = [item]
            var_prefix = "" # No rule prefix for fact variables (though facts shouldn't have vars)
        else:
            continue

        for term in terms:
            predicate = term.predicate
            predicates.add(predicate)
            if predicate not in predicates_arity:
                 predicates_arity[predicate] = len(term.args)
            # Check arity consistency
            elif predicates_arity[predicate] != len(term.args):
                 logging.warning(f"Inconsistent arity for predicate {predicate}. Found {len(term.args)} and {predicates_arity[predicate]}. Keeping first seen.")

            for arg in term.args:
                if is_variable(arg):
                     # Add prefix only if it's from a rule
                     variables.add(f"{var_prefix}{arg}" if var_prefix else arg)
                else:
                     constants.add(arg)

    return predicates, predicates_arity, constants, variables

def get_rules_from_rules_file(file_path: str) -> List[Rule]:
    rules = []
    try:
        with open(file_path, "r") as f:
            # Simple parsing assuming "head :- body1, body2." format
            for line in f:
                line = line.strip()
                if not line or line.startswith("%") or ":-" not in line:
                    continue
                head_str, body_part = line.split(":-", 1)
                head_atom = get_atom_from_string(head_str.strip())
                body_atoms = []
                if body_part.strip() != '.': # Handle rules with empty body? Unlikely.
                     body_strs = [s.strip() for s in body_part.strip().rstrip('.').split(',')]
                     body_atoms = [get_atom_from_string(b) for b in body_strs if b] # Ensure non-empty

                # Convert args to uppercase variables (adjust if needed)
                head_atom.args = [arg.upper() if isinstance(arg, str) else arg for arg in head_atom.args]
                for body_atom in body_atoms:
                     body_atom.args = [arg.upper() if isinstance(arg, str) else arg for arg in body_atom.args]

                rules.append(Rule(head_atom, body_atoms))
    except FileNotFoundError:
        raise FileNotFoundError(f"Rules file {file_path} not found")
    except Exception as e:
         logging.error(f"Error parsing rules file {file_path}: {e}", exc_info=True)
    return rules

def get_queries(path: str) -> List[Term]:
    """Reads queries from a file (one per line)."""
    queries = []
    try:
        with open(path, "r") as f:
            for line in f:
                query_str = line.strip()
                if query_str:
                    queries.append(get_atom_from_string(query_str))
    except FileNotFoundError:
        # Log warning instead of raising error? Depends on whether file is optional.
        logging.warning(f"Query file {path} not found.")
    except Exception as e:
        logging.error(f"Error reading query file {path}: {e}", exc_info=True)
    return queries


class DataHandler:
    """
    Handles loading and providing data (facts, rules, queries) for the RL environment.
    Simplified version without Janus and static corruption.
    """
    def __init__(self, dataset_name: str,
                 base_path: str,
                 train_file: str = "train.txt",
                 valid_file: str = "valid.txt",
                 test_file: str = "test.txt",
                 rules_file: str = "rules.pl", # Or rules.txt etc.
                 facts_file: str = "facts.txt", # Or train.txt if facts = train triples
                 n_eval_queries: Optional[int] = None,
                 n_test_queries: Optional[int] = None,
                 corruption_mode: Optional[str] = None, # Only 'dynamic' or None relevant now
                 filter_queries_by_rules: bool = True,
                 ):
        """
        Initialize the data handler.

        Args:
            dataset_name: Name of the dataset.
            base_path: Base path to the dataset directory.
            train_file: Filename for training queries/facts.
            valid_file: Filename for validation queries.
            test_file: Filename for test queries.
            rules_file: Filename for rules.
            facts_file: Filename for base facts (can be same as train_file).
            n_eval_queries: Max number of validation queries to load.
            n_test_queries: Max number of test queries to load.
            corruption_mode: Set to 'dynamic' if dynamic negative sampling is used.
            filter_queries_by_rules: If True, filter queries whose predicates don't appear in rule heads.
        """
        self.dataset_name = dataset_name
        self.base_path = join(base_path, dataset_name)
        self.corruption_mode = corruption_mode

        # --- Load Rules and Facts ---
        rules_path = join(self.base_path, rules_file)
        facts_path = join(self.base_path, facts_file)
        self.rules = get_rules_from_rules_file(rules_path)
        self.facts = get_queries(facts_path) # Assuming facts are simple atoms
        logging.info(f"Loaded {len(self.rules)} rules and {len(self.facts)} facts.")

        # --- Load Queries ---
        train_path = join(self.base_path, train_file)
        valid_path = join(self.base_path, valid_file)
        test_path = join(self.base_path, test_file)

        # Load raw queries
        self.train_queries_raw = get_queries(train_path)
        self.valid_queries_raw = get_queries(valid_path)
        self.test_queries_raw = get_queries(test_path)
        logging.info(f"Loaded raw queries: Train={len(self.train_queries_raw)}, Valid={len(self.valid_queries_raw)}, Test={len(self.test_queries_raw)}")

        # --- Extract Vocabulary ---
        self.predicates, self.predicates_arity, self.constants, self.variables = get_predicates_and_arguments(
            self.rules, self.facts
        )
        self.max_arity = get_max_arity(self.rules, self.facts)
        self.constant_no = len(self.constants)
        self.predicate_no = len(self.predicates)
        self.variable_no = len(self.variables)
        logging.info(f"Vocabulary: Preds={self.predicate_no}, Consts={self.constant_no}, Vars={self.variable_no}, MaxArity={self.max_arity}")


        # --- Filter and Finalize Queries ---
        rules_head_predicates = set(rule.head.predicate for rule in self.rules) if filter_queries_by_rules else None

        self.train_queries = self._filter_and_slice_queries(self.train_queries_raw, rules_head_predicates, None, "Train")
        self.valid_queries = self._filter_and_slice_queries(self.valid_queries_raw, rules_head_predicates, n_eval_queries, "Valid")
        self.test_queries = self._filter_and_slice_queries(self.test_queries_raw, rules_head_predicates, n_test_queries, "Test")

        # --- Setup for Dynamic Corruption (if needed) ---
        self.sampler = None
        self.triples_factory = None
        self.entity2domain = None
        self.domain2entity = None
        if self.corruption_mode == "dynamic":
            logging.info("Setting up for dynamic corruption (requires sampler/factory to be set externally).")
            # Potentially load domain info if needed by sampler
            if 'countries' in self.dataset_name: # Example specific logic
                 self._load_domain_info()


    def _filter_and_slice_queries(self, queries: List[Term],
                                  filter_preds: Optional[Set[str]],
                                  slice_n: Optional[int],
                                  set_name: str) -> List[Term]:
        """Helper to filter queries by predicate and slice to a max number."""
        original_count = len(queries)
        filtered_queries = queries
        if filter_preds:
            filtered_queries = [q for q in queries if q.predicate in filter_preds]
            excluded_count = original_count - len(filtered_queries)
            if excluded_count > 0:
                logging.info(f"{set_name} queries excluded by rule head filter: {excluded_count} ({excluded_count/original_count:.1%})")

        sliced_queries = filtered_queries[:slice_n] if slice_n is not None else filtered_queries
        if slice_n is not None and len(filtered_queries) > slice_n:
             logging.info(f"{set_name} queries sliced from {len(filtered_queries)} to {len(sliced_queries)}")

        logging.info(f"Final {set_name} query count: {len(sliced_queries)}")
        return sliced_queries

    def _load_domain_info(self):
        """Loads domain information (example for countries dataset)."""
        domain_file = join(self.base_path, "domain2constants.txt")
        try:
            self.entity2domain = {}
            self.domain2entity = defaultdict(list)
            with open(domain_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        domain, *entities = line.split()
                        for entity in entities:
                            self.entity2domain[entity] = domain
                            self.domain2entity[domain].append(entity)
            logging.info(f"Loaded domain info from {domain_file}")
        except FileNotFoundError:
            logging.warning(f"Domain info file {domain_file} not found. Domain-based sampling may fail.")
        except Exception as e:
            logging.error(f"Error loading domain info from {domain_file}: {e}", exc_info=True)

    # --- Methods to potentially set sampler/factory externally ---
    def set_sampler(self, sampler):
        self.sampler = sampler

    def set_triples_factory(self, factory):
        self.triples_factory = factory

