import janus_swi as janus
from typing import List, Any, Dict, Union, Tuple, Optional
import traceback # For detailed error logging

# --- Term Class Definition ---
# Ensure you have a compatible Term class defined or imported.
# This is a basic example; adapt it if yours is different.
class Term:
    """Represents a logical term like predicate(arg1, arg2)."""
    def __init__(self, predicate: str, args: List[Union[str, 'Term']]):
        # Basic validation
        if not isinstance(predicate, str) or not predicate:
            raise ValueError("Term predicate must be a non-empty string.")
        if not isinstance(args, list):
            raise ValueError("Term args must be a list.")

        self.predicate = predicate
        # Store args, potentially converting dicts back to Terms if needed during init
        self.args = [self._ensure_arg_type(arg) for arg in args]

    def _ensure_arg_type(self, arg):
        """Helper to ensure arguments are strings or Term instances."""
        if isinstance(arg, (str, Term)):
            return arg
        # Add handling for other types if necessary (e.g., numbers)
        elif isinstance(arg, (int, float)):
             return str(arg) # Convert numbers to string for Prolog compatibility
        else:
            # Attempt to convert dict back to Term if structure matches
            if isinstance(arg, dict) and 'predicate' in arg and 'args' in arg:
                 try:
                      return Term(arg['predicate'], arg['args'])
                 except Exception as e:
                      raise TypeError(f"Failed to convert dict arg to Term: {arg}. Error: {e}")
            raise TypeError(f"Unsupported argument type in Term: {type(arg)}. Value: {arg}")


    def __repr__(self) -> str:
        """Provides a string representation useful for debugging."""
        # Represent arguments recursively
        args_str = ', '.join(map(repr, self.args))
        return f"{self.predicate}({args_str})"

    def __eq__(self, other) -> bool:
        """Checks for equality with another Term."""
        if not isinstance(other, Term):
            return NotImplemented
        return self.predicate == other.predicate and self.args == other.args

    def __hash__(self) -> int:
        """Computes a hash for the Term, allowing use in sets/dicts."""
        # Hash based on predicate and tuple of args (lists aren't hashable)
        try:
             # Recursively hash arguments if they are Terms
             hashed_args = tuple(hash(arg) if isinstance(arg, Term) else arg for arg in self.args)
             return hash((self.predicate, hashed_args))
        except TypeError as e:
             # Handle unhashable arguments if necessary
             print(f"Warning: Could not hash Term args: {self.args}. Error: {e}")
             # Fallback hash (less ideal, might cause collisions)
             return hash(self.predicate)


    def to_prolog_str(self) -> str:
        """Converts the Term object to a Prolog-compatible string."""
        args_str = ','.join(_py_arg_to_prolog_str(arg) for arg in self.args)
        # Basic predicate handling, might need quoting for complex/operator predicates
        predicate_str = self.predicate
        # Simple check if quoting might be needed (contains non-alphanumeric or starts uppercase)
        # This is a basic heuristic and might need refinement based on actual Prolog syntax rules.
        if not predicate_str.isidentifier() or predicate_str[0].isupper():
             # Basic single quoting, escape single quotes within
             predicate_str = f"'{predicate_str.replace('\'', '\\\'')}'"

        return f"{predicate_str}({args_str})"

# --- Helper Conversion Functions ---

def _py_arg_to_prolog_str(arg: Any) -> str:
    """Converts a Python argument (string, variable, number, or nested Term) to Prolog string."""
    if isinstance(arg, Term):
        return arg.to_prolog_str() # Use the Term's own conversion method
    elif isinstance(arg, str):
        # Assume strings starting with uppercase or '_' are variables
        if arg and (arg[0].isupper() or arg[0] == '_'):
            return arg
        else:
            # Quote atoms/strings. Escape single quotes and backslashes within.
            # Handles most cases, but complex atoms might need more specific quoting.
            escaped_arg = arg.replace('\\', '\\\\').replace('\'', '\\\'')
            return f"'{escaped_arg}'"
    elif isinstance(arg, (int, float)):
         # Numbers are generally represented directly
         return str(arg)
    else:
        # Fallback for other types - might raise errors in Prolog
        print(f"Warning: Converting unexpected type {type(arg)} to string: {arg}")
        return str(arg)

def _janus_result_to_python(result: Any) -> Union[Term, str, int, float, list, dict, None]:
    """
    Converts a result component from Janus back to Python types (Term, str, list, etc.).
    Relies on Janus returning standard Python types for basic Prolog types
    and potentially janus specific objects or tuples for compound terms/variables.
    """
    # Simple types pass through: strings, numbers
    if isinstance(result, (str, int, float)):
        return result
    # Check for janus specific Variable type (adjust if janus uses a different representation)
    # Assuming variables might be returned as strings like '_GXXX' or a specific object
    elif isinstance(result, str) and result.startswith('_') and result[1:].isalnum():
         # Heuristic: Treat strings like '_G123' as variables
         return result # Return as string
    # Check for compound term representation (assuming tuple: (functor, [arg1, arg2,...]))
    # Adjust this based on how janus actually returns compound terms.
    elif isinstance(result, tuple) and len(result) >= 1 and isinstance(result[0], str):
         # Potentially a compound term: ('predicate', [arg1, arg2]) or ('atom',)
         predicate = result[0]
         args_raw = result[1] if len(result) > 1 else []

         if isinstance(args_raw, list):
              # Recursively convert arguments
              args = [_janus_result_to_python(arg) for arg in args_raw]
              # Basic check to ensure args conversion didn't fail badly
              if any(arg is None and raw_arg is not None for arg, raw_arg in zip(args, args_raw)):
                   print(f"Warning: Null argument after conversion in term: {predicate}{args_raw}")
                   # Decide error handling: raise, return None, return partially converted?
                   return None # Indicate conversion failure for this term
              try:
                   return Term(predicate, args)
              except (ValueError, TypeError) as e:
                   print(f"Error creating Term from Janus result: Predicate='{predicate}', Args={args}. Error: {e}")
                   return None # Indicate failure
         else:
              # Handle cases like ('atom',) - maybe return just the atom string?
              if len(result) == 1 and not args_raw:
                   return predicate # Treat ('atom',) as just 'atom'
              else:
                   print(f"Warning: Unexpected structure for potential term from Janus: {result}")
                   return None # Indicate failure

    # Handle lists: recursively convert elements
    elif isinstance(result, list):
        converted_list = [_janus_result_to_python(item) for item in result]
        # Check for conversion failures within the list
        if any(item is None and raw_item is not None for item, raw_item in zip(converted_list, result)):
             print(f"Warning: Null item after conversion in list: {result}")
             # Return None or the partially converted list?
             return None # Indicate conversion failure for the list
        return converted_list

    # Handle dicts (if Prolog returns dicts, unlikely unless using specific libraries)
    elif isinstance(result, dict):
         converted_dict = {k: _janus_result_to_python(v) for k, v in result.items()}
         # Check for conversion failures
         if any(v is None and raw_v is not None for (k,v), (rk, raw_v) in zip(converted_dict.items(), result.items())):
              print(f"Warning: Null value after conversion in dict: {result}")
              return None
         return converted_dict

    elif result is None:
         return None # Pass None through

    else:
        # Handle other potential janus types (e.g., janus.Variable if it exists)
        # Add specific checks here based on janus documentation/behavior.
        print(f"Warning: Unexpected data type from Janus: {type(result)}. Value: {result}")
        # Fallback: return as is or return None to indicate inability to convert
        return None # Indicate conversion failure


# --- Main Function ---

def get_next_state_prolog(
    current_state: List[Term],
    janus_instance: Optional[Any] = None
) -> List[List[Term]]:
    """
    Performs one step of Prolog resolution using the loaded KB via Janus.

    Requires that the 'step_resolver.pl' module and the main knowledge
    base (e.g., 'kinship_family.pl') have already been consulted into
    the Janus environment.

    Args:
        current_state: A list of Term objects representing the current goals.
        janus_instance: The initialized janus_swi instance (optional,
                        defaults to using the global janus instance if initialized).

    Returns:
        A list of possible next states. Each state is a list of Term objects.
        - Returns `[[Term('True', [])]]` if a proof is found in this step
          (Prolog returns an empty list `[]` as a next state).
        - Returns `[]` if no resolution step is possible (Prolog predicate fails).
        - Returns `[]` and prints errors if conversion or Prolog errors occur.
    """
    # Use provided janus instance or the global one
    prolog = janus_instance if janus_instance is not None else janus

    if not prolog.is_initialized():
         print("Error: Janus-SWI is not initialized. Please initialize it and consult KB files.")
         return []

    # Handle the base case: empty goal list means success.
    if not current_state:
        # Represent success as a list containing one state: the empty goal list.
        # The caller should interpret an empty inner list as True.
        # Or, more explicitly:
        return [[Term('True', [])]]

    # Convert Python List[Term] to Prolog list string: "[g1, g2, ...]"
    try:
        # Filter out any None terms that might have crept in
        valid_terms = [term for term in current_state if isinstance(term, Term)]
        if len(valid_terms) != len(current_state):
             print(f"Warning: Filtered out non-Term objects from current_state: {current_state}")

        if not valid_terms: # If filtering left nothing
             if current_state: # But original state was not empty
                  print("Error: Current state contains no valid Term objects after filtering.")
                  return []
             else: # Original state was empty, handled above
                  return [[Term('True', [])]]


        prolog_goal_list_str = '[' + ','.join(term.to_prolog_str() for term in valid_terms) + ']'

    except Exception as e:
        print(f"Error converting Python state to Prolog string: {e}")
        print(f"State was: {current_state}")
        traceback.print_exc()
        return [] # Indicate conversion error

    # Construct the Prolog query to call the helper predicate
    query = f"step_resolver:resolve_one_step({prolog_goal_list_str}, ListOfNextStates)."
    # print(f"DEBUG: Executing Prolog query: {query}") # Optional debug print

    final_next_states = []
    try:
        # Use findall to get all possible next states in one go.
        # query_once would only give the first result.
        results = list(prolog.query(query)) # Use query to get all solutions

        # print(f"DEBUG: Prolog raw results: {results}") # Optional debug print

        if not results:
            # No solutions found by findall - means no next step possible
            return []

        # Process each solution found by findall
        for solution in results:
             if solution and 'ListOfNextStates' in solution:
                  raw_list_of_lists = solution['ListOfNextStates']
                  # raw_list_of_lists should be the list of *all* possible next states
                  # generated by findall within Prolog.

                  # Convert the Prolog structure back to Python List[List[Term]]
                  converted_outer_list = _janus_result_to_python(raw_list_of_lists)

                  if converted_outer_list is None:
                       print(f"Error: Failed to convert main result list from Prolog: {raw_list_of_lists}")
                       continue # Skip this potentially corrupted result

                  if not isinstance(converted_outer_list, list):
                       print(f"Error: Expected a list of lists from Prolog conversion, got {type(converted_outer_list)}")
                       continue # Skip malformed result

                  # Iterate through each potential next state list
                  for state_list_representation in converted_outer_list:
                       if state_list_representation == []: # Empty list means proof found for this branch
                            # Ensure we add the explicit True marker only once if multiple proofs found?
                            # The current structure collects all outcomes.
                            final_next_states.append([Term('True', [])])
                       elif isinstance(state_list_representation, list):
                            # Convert list of terms for this specific next state
                            py_state = []
                            valid_state = True
                            for item in state_list_representation:
                                 # We expect items to be Term objects or variable strings after conversion
                                 if isinstance(item, Term):
                                      py_state.append(item)
                                 elif isinstance(item, str): # Assume strings are variables or atoms
                                      # Decide how to handle raw strings - wrap in Term?
                                      # For now, let's assume they should have been Terms if compound.
                                      # If it's just an atom like 'true' or a variable, how to represent?
                                      # Let's skip raw strings for now, assuming conversion should yield Terms.
                                      print(f"Warning: Skipping raw string '{item}' in next state list {state_list_representation}. Expecting Term objects.")
                                      # Or potentially wrap simple atoms: py_state.append(Term(item, []))
                                 else:
                                      print(f"Error: Unexpected item type '{type(item)}' in converted state list: {state_list_representation}")
                                      valid_state = False
                                      break # Stop processing this malformed state list
                            if valid_state:
                                 final_next_states.append(py_state)
                       else:
                            print(f"Error: Expected list or empty list for a state, got {type(state_list_representation)}: {state_list_representation}")

             else:
                  # Should not happen if query syntax is correct and predicate exists
                  print(f"Warning: Prolog query ran but solution dict is empty or missing key: {solution}")


        # Deduplicate resulting states if necessary (Prolog might yield same state via different paths)
        # Using tuples of tuples for hashing requires Terms to be hashable and consistently ordered.
        unique_states = []
        seen_states = set()
        for state in final_next_states:
             # Create a hashable representation (e.g., tuple of Term hashes or reprs)
             # Sorting ensures order doesn't affect uniqueness check
             try:
                  state_repr = tuple(sorted(map(repr, state))) # Simple approach using repr
                  # Or use hash: state_repr = tuple(sorted(map(hash, state)))
                  if state_repr not in seen_states:
                       unique_states.append(state)
                       seen_states.add(state_repr)
             except Exception as e:
                  print(f"Warning: Could not create hashable representation for state {state}. Adding without uniqueness check. Error: {e}")
                  unique_states.append(state) # Add anyway if hashing fails


        return unique_states # Return the list of unique next states

    except janus.PrologError as e:
        print(f"Prolog error during resolution step: {e}")
        print(f"Query was: {query}")
        traceback.print_exc()
        return [] # Indicate failure
    except Exception as e:
        print(f"Unexpected Python/Janus error: {e}")
        print(f"Query was: {query}")
        traceback.print_exc()
        return [] # Indicate failure