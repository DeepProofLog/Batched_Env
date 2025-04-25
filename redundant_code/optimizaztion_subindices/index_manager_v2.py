from typing import List, Optional, Tuple, Dict, Union
import random
from optimizaztion_subindices.utils_v1 import Term, is_variable, extract_var
import torch
from optimizaztion_subindices.utils_v1 import Rule


class IndexManager():
    def __init__(self, constants: set,
                 predicates: set,
                 variables: set, # Initial "global" variables
                 constant_no: int,
                 predicate_no: int,
                 variable_no: int, # Initial count of global variables
                 rules: List[Rule], # Rules are now needed here
                 constants_images: set = (),
                 constant_images_no: int = 0,
                 padding_atoms: int = 10, # Might be less relevant now
                 max_arity: int = 2,      # Might be less relevant now
                 device: Optional[torch.device] = None): # Made device optional

        self.device = device if device else torch.device("cpu") # Default to CPU
        self.constants = constants
        self.predicates = predicates
        # self.variables = variables # Store original set - Will be updated
        self.constant_no = constant_no
        # self.variable_no = variable_no # Will be updated
        self.predicate_no = predicate_no
        self.constants_images = constants_images
        self.constant_images_no = constant_images_no
        self.rules_as_terms = rules # Keep original rules

        # --- Collect all variables from rules ---
        all_rule_vars = set()
        if rules: # Check if rules list is provided and not empty
            for rule in rules:
                # Collect from head
                if rule.head and rule.head.args: # Check head exists and has args
                    for arg in rule.head.args:
                        if is_variable(arg):
                            all_rule_vars.add(arg)
                # Collect from body
                if rule.body: # Check body exists
                    for term in rule.body:
                         if term and term.args: # Check term exists and has args
                            for arg in term.args:
                                if is_variable(arg):
                                    all_rule_vars.add(arg)

        # Combine initial variables with rule variables
        self.variables = variables.union(all_rule_vars) # Update the stored set
        self.variable_no = len(self.variables) # Update the count based on combined set

        # --- Define Fixed Special IDs and Offset ---
        self.padding_id = 0
        self.true_pred_id = 1
        self.false_pred_id = 2
        self.id_offset = 2

        # --- ID Ranges (Shifted - Use updated self.variable_no) ---
        self.constant_start_idx = self.id_offset + 1
        self.constant_end_idx = self.id_offset + self.constant_no
        # Update variable range based on potentially new count
        self.variable_start_idx = self.constant_end_idx + 1
        self.variable_end_idx = self.variable_start_idx + self.variable_no - 1 # Use updated count
        self.temp_variable_start_idx = self.variable_end_idx + 1

        self.predicate_start_idx = self.id_offset + 1 # Predicate IDs can overlap with Constants now
        self.predicate_end_idx = self.id_offset + self.predicate_no

        # --- Mappings ---
        self.constant_str2idx: Dict[str, int] = {}
        self.constant_idx2str: Dict[int, str] = {}
        self.predicate_str2idx: Dict[str, int] = {}
        self.predicate_idx2str: Dict[int, str] = {}
        self.variable_str2idx: Dict[str, int] = {} # Now includes all rule vars
        self.variable_idx2str: Dict[int, str] = {} # Includes all rule vars + temp vars

        # --- Rule Representations ---
        self.rules_as_ids: List[Tuple[Tuple[int, ...], List[Tuple[int, ...]]]] = []
        self.facts_as_ids_set: Set[Tuple[int, ...]] = set() # Populated externally now

        # --- Create Mappings and Convert Rules ---
        self.create_global_idx()      # Create mappings using the updated self.variables
        self._convert_rules_to_ids() # Convert rules using the created indices

        self.reset_atom() # Keep for potential compatibility if needed

    def create_global_idx(self):
        """Creates global indices for constants, predicates, and variables."""
        # Constants (handle images if present - assuming no images for simplicity here)
        current_id = self.constant_start_idx
        # Map regular constants first
        for const in sorted(self.constants): # Sort for deterministic mapping
            if const not in self.constant_str2idx:
                self.constant_str2idx[const] = current_id
                current_id += 1

        # Check if the number of mapped constants matches the expected count
        # Adjust expected count if images are handled differently
        expected_const_count = self.constant_no # Assuming no image constants for now
        if len(self.constant_str2idx) != expected_const_count:
             print(f"Warning: Constant count mismatch. Expected {expected_const_count}, Got {len(self.constant_str2idx)}")
        # Check if the highest assigned ID matches the calculated end index *if* constants exist
        if self.constant_str2idx:
            max_const_id = max(self.constant_str2idx.values())
            # The expected end ID is start_id + count - 1
            expected_end_id = self.constant_start_idx + expected_const_count - 1
            if max_const_id != expected_end_id:
                 print(f"Warning: Constant mapping end mismatch. Max ID: {max_const_id}, Expected End: {expected_end_id} (Based on start {self.constant_start_idx} and count {expected_const_count})")
        # Check against the pre-calculated end index (should match if counts are correct)
        if current_id -1 != self.constant_end_idx and self.constant_str2idx: # Check if the *next* ID aligns
             print(f"Warning: Next constant ID {current_id} does not match expected end index + 1 ({self.constant_end_idx + 1})")


        self.constant_idx2str = {v: k for k, v in self.constant_str2idx.items()}


        # Predicates
        self.predicate_str2idx = {term: self.predicate_start_idx + i for i, term in enumerate(sorted(self.predicates))}
        self.predicate_idx2str = {v: k for k, v in self.predicate_str2idx.items()}
        # Add special predicates explicitly
        self.predicate_idx2str[self.true_pred_id] = 'True'
        self.predicate_idx2str[self.false_pred_id] = 'False'
        # Check if the number of mapped regular predicates matches the expected count
        regular_pred_count = len(self.predicate_str2idx)
        if regular_pred_count != self.predicate_no:
             print(f"Warning: Predicate count mismatch. Expected {self.predicate_no}, Got {regular_pred_count}")
        # Check if the highest assigned regular ID matches the calculated end index *if* predicates exist
        if self.predicate_str2idx:
            max_reg_pred_id = max(self.predicate_str2idx.values())
            if max_reg_pred_id != self.predicate_end_idx:
                print(f"Warning: Predicate mapping end mismatch. Max ID: {max_reg_pred_id}, Expected End: {self.predicate_end_idx}")


        # Global Variables (Now includes all rule variables)
        # Ensure self.variables is populated correctly before this runs
        self.variable_str2idx = {term: self.variable_start_idx + i for i, term in enumerate(sorted(self.variables))}
        self.variable_idx2str = {v: k for k, v in self.variable_str2idx.items()}
        # Check if the number of mapped variables matches the updated count
        if len(self.variable_str2idx) != self.variable_no:
             print(f"Warning: Variable count mismatch. Expected {self.variable_no}, Got {len(self.variable_str2idx)}")
        # Check if the highest assigned variable ID matches the calculated end index *if* variables exist
        if self.variable_str2idx:
            max_var_id = max(self.variable_str2idx.values())
            if max_var_id != self.variable_end_idx:
                 print(f"Warning: Variable mapping end mismatch. Max ID: {max_var_id}, Expected End: {self.variable_end_idx}")



    # --- _convert_rules_to_ids (No change needed in logic) ---
    # This function relies on term_to_ids finding all variables in the global map now.
    def _convert_rules_to_ids(self):
        """Converts stored Rule objects to ID representation."""
        self.rules_as_ids = []
        # Rule conversion uses global var map (now including all rule vars)
        # term_to_ids called with assign_new_vars=False should now find all rule vars.
        initial_var_map = self.variable_str2idx.copy() # Map with all global+rule vars
        initial_next_temp_id = self.temp_variable_start_idx # Not used when assign_new_vars=False

        for rule in self.rules_as_terms:
            if not rule or not rule.head: # Add check for invalid rule object
                print(f"Warning: Skipping invalid or empty rule: {rule}")
                continue
            try:
                # Use a consistent variable map *per rule* conversion (starts with full map)
                rule_var_map = initial_var_map.copy()
                rule_next_temp_id = initial_next_temp_id

                head_id_tuple, rule_var_map, rule_next_temp_id = self.term_to_ids(
                    rule.head, rule_var_map, rule_next_temp_id, assign_new_vars=False # Should find vars now
                )

                body_id_tuples = []
                if rule.body: # Check if body exists
                    for term in rule.body:
                         if not term: # Add check for invalid term in body
                             print(f"Warning: Skipping invalid term in body of rule: {rule}")
                             continue
                         body_term_id_tuple, rule_var_map, rule_next_temp_id = self.term_to_ids(
                            term, rule_var_map, rule_next_temp_id, assign_new_vars=False # Should find vars now
                         )
                         body_id_tuples.append(body_term_id_tuple)

                self.rules_as_ids.append((head_id_tuple, body_id_tuples))
            except ValueError as e:
                print(f"Warning: Skipping rule '{rule}' due to conversion error: {e}")
            except Exception as e: # Catch other potential errors during conversion
                 print(f"Warning: Skipping rule '{rule}' due to unexpected error: {e}")


    # --- term_to_ids (No change needed in logic) ---
    # The critical part is the check:
    # elif var_id is None: ... if assign_new_vars: ... else: raise ValueError(...)
    # When assign_new_vars is False (rule conversion), this 'else' branch should no
    # longer be hit for valid rule variables because they are now in variable_str2idx.
    def term_to_ids(self, term: Term,
                    current_var_map: Optional[Dict[str, int]] = None,
                    next_temp_var_id: Optional[int] = None,
                    assign_new_vars: bool = True # Control creation of new temp vars
                   ) -> Tuple[Tuple[int, ...], Dict[str, int], int]:
        """
        Converts a Term object to a tuple of integer IDs.
        Handles potentially new variables (e.g., in queries) by assigning temporary IDs.
        Returns the ID tuple, updated variable map, and next available temp var ID.
        """
        if not term or not term.predicate: # Add check for invalid term
             raise ValueError(f"Cannot convert invalid term: {term}")

        local_var_map = current_var_map.copy() if current_var_map is not None else self.variable_str2idx.copy()
        local_next_temp_id = next_temp_var_id if next_temp_var_id is not None else self.temp_variable_start_idx

        try:
            # Handle special predicates by name first
            if term.predicate == 'True': pred_id = self.true_pred_id
            elif term.predicate == 'False': pred_id = self.false_pred_id
            # elif term.predicate == 'End': pred_id = self.end_pred_id # If used
            else:
                pred_id = self.predicate_str2idx.get(term.predicate)
                if pred_id is None: raise ValueError(f"Predicate '{term.predicate}' not found.")

            arg_ids = []
            if term.args: # Check if args exist
                for arg in term.args:
                    if is_variable(arg): # Check if it looks like a variable string
                        var_id = local_var_map.get(arg)
                        if var_id is not None: # Variable found in current map (global/rule or temp)
                             arg_ids.append(var_id)
                        elif assign_new_vars:
                            # New variable encountered (e.g., in a query) -> assign temp ID
                            var_id = local_next_temp_id
                            local_var_map[arg] = var_id
                            # Add to reverse map (essential for ids_to_term)
                            if var_id not in self.variable_idx2str:
                                self.variable_idx2str[var_id] = arg
                            local_next_temp_id += 1
                            arg_ids.append(var_id)
                        else:
                            # Rule conversion should not create new variables *if* all rule vars were pre-collected.
                            # If this error occurs now, it likely means the pre-collection missed something or the rule is malformed.
                            raise ValueError(f"Variable '{arg}' in rule term '{term}' not found in global/rule variable map. Check rule definitions and IndexManager setup.")
                    else: # Constant
                        const_id = self.constant_str2idx.get(arg)
                        if const_id is None: raise ValueError(f"Constant '{arg}' not found in term '{term}'.")
                        arg_ids.append(const_id)
            # Handle 0-arity predicates (no args)
            # else: pass

            return (pred_id,) + tuple(arg_ids), local_var_map, local_next_temp_id
        # except KeyError as e:
        #     raise ValueError(f"Error converting term '{term}': Key {e} not found in mappings.") from e
        except ValueError as e: # Catch ValueErrors raised within or above
            # Add more context to the error
            raise ValueError(f"Error converting term '{term}' (Predicate: '{term.predicate}', Args: {term.args}): {e}") from e
        except Exception as e: # Catch unexpected errors
            raise RuntimeError(f"Unexpected error converting term '{term}': {e}") from e


    # --- Remaining methods unchanged ---
    def reset_atom(self):
        self.atom_to_index: Dict[Term, int] = {}
        self.atom_id_to_sub_id: Dict[int, torch.Tensor] = {}
        self.next_atom_index = 1

    def is_constant_id(self, arg_id: int) -> bool:
        return self.constant_start_idx <= arg_id <= self.constant_end_idx

    def is_global_variable_id(self, arg_id: int) -> bool:
        # Note: This now includes variables that were local to rules but added to the global map
        return self.variable_start_idx <= arg_id <= self.variable_end_idx

    def is_temp_variable_id(self, arg_id: int) -> bool:
        return arg_id >= self.temp_variable_start_idx

    def is_variable_id(self, arg_id: int) -> bool:
        # Checks if it's a global/rule variable OR a temporary variable
        return arg_id >= self.variable_start_idx

    def ids_to_term(self, term_ids: Tuple[int, ...]) -> Term:
        """Converts a tuple of integer IDs back to a Term object."""
        if not term_ids:
             raise ValueError("Cannot convert empty tuple to Term.")
        try:
            pred_id = term_ids[0]
            predicate = self.predicate_idx2str.get(pred_id)
            if predicate is None: raise ValueError(f"Predicate ID {pred_id} not found.")

            args = []
            for arg_id in term_ids[1:]:
                if self.is_variable_id(arg_id): # Check if it's any kind of variable ID
                    arg_str = self.variable_idx2str.get(arg_id)
                    if arg_str is None:
                        # Safety default if somehow missing from reverse map
                        arg_str = f"_Var{arg_id}"
                        self.variable_idx2str[arg_id] = arg_str # Store generated name
                    args.append(arg_str)
                elif self.is_constant_id(arg_id):
                    arg_str = self.constant_idx2str.get(arg_id)
                    if arg_str is None: raise ValueError(f"Constant ID {arg_id} not found.")
                    args.append(arg_str)
                else:
                    if arg_id == self.padding_id:
                        # Decide how to handle padding. Error? Special string?
                        # For now, let's raise an error as padding shouldn't be in final terms.
                        raise ValueError(f"Padding ID {arg_id} encountered during ids_to_term conversion.")
                    else:
                         raise ValueError(f"Argument ID {arg_id} is not a valid constant, variable, or known special ID.")

            return Term(predicate, args)
        except KeyError as e:
            raise ValueError(f"Error converting term IDs {term_ids}: ID {e} not found in idx2str map.") from e
        except IndexError:
            raise ValueError(f"Error converting term IDs {term_ids}: Malformed tuple (IndexError).")
        except ValueError as e: # Catch ValueErrors raised within
            raise ValueError(f"Error converting term IDs {term_ids}: {e}") from e
        except Exception as e: # Catch unexpected errors
             raise RuntimeError(f"Unexpected error converting term IDs {term_ids}: {e}") from e


    def state_to_ids(self, state: List[Term],
                     current_var_map: Optional[Dict[str, int]] = None,
                     next_temp_var_id: Optional[int] = None
                    ) -> Tuple[List[Tuple[int, ...]], Dict[str, int], int]:
        """Converts a list of Term objects (state) to ID tuples."""
        state_ids = []
        # Start with global map if none provided
        local_var_map = current_var_map if current_var_map is not None else self.variable_str2idx.copy()
        local_next_temp_id = next_temp_var_id if next_temp_var_id is not None else self.temp_variable_start_idx

        for term in state:
            # Allow creation of new temp vars for state terms (queries/goals)
            term_id_tuple, local_var_map, local_next_temp_id = self.term_to_ids(
                term, local_var_map, local_next_temp_id, assign_new_vars=True
            )
            state_ids.append(term_id_tuple)
        return state_ids, local_var_map, local_next_temp_id

    def ids_to_state(self, ids_state: List[Tuple[int, ...]]) -> List[Term]:
        """Converts a list of ID tuples back to a list of Term objects."""
        return [self.ids_to_term(ids) for ids in ids_state]
