import re
from typing import Dict, Union, List, Any, Tuple, Iterable, Optional
import datetime
import os
import random
import numpy as np
import ast
from collections import defaultdict
import dataclasses
from dataclasses import dataclass, field
from tensordict import TensorDict, TensorDictBase
import functools

import os
import numpy as np
import cProfile
import pstats
import io
import torch
import torch.profiler
from torch.profiler import ProfilerActivity
import torch.nn as nn
# Optional Weights & Biases integration. Avoid hard dependency during tests.
try:
    import wandb  # type: ignore
    try:
        from wandb.integration.sb3 import WandbCallback  # type: ignore
    except Exception:  # pragma: no cover - optional dependency path
        WandbCallback = object  # fallback stub
except Exception:  # pragma: no cover - optional dependency path
    wandb = None  # type: ignore
    WandbCallback = object  # fallback stub


# ----------------------
# Logic utils       
# ----------------------

@dataclass(frozen=True, order=True, slots=True)
class Term:
    """
    Represents a logical term (predicate and arguments) using a frozen dataclass.
    Instances are immutable, hashable, and ordered lexicographically
    by predicate, then by arguments.
    """
    predicate: str
    args: Tuple[str, ...] = field(compare=True)

    def __str__(self):
        """Custom string representation like 'predicate(arg1,arg2)'."""
        args_str = ', '.join(map(str, self.args))
        return f"{self.predicate}({args_str})"

    def __repr__(self):
        """Representation is the same as the string form for readability."""
        args_str = ', '.join(map(str, self.args))
        return f"{self.predicate}({args_str})"

    def prolog_str(self):
        """
        Prolog-style string representation, e.g., 'predicate("_VAR_1", "arg2")'
        1. Add a " at the beginning and end, except if a variable starts with 'Var'.
        2. Replace 'Var' with '_Var'.
        """
        args_str = ', '.join(
            [f'"{arg}"' if not arg.startswith('Var') else f'_{arg}' for arg in self.args]
        )
        return f'{self.predicate}({args_str})'
    # include a sorting method to sort terms alphabetically by predicate and then by args
    # def __lt__(self, other):
    #     """
    #     Less than comparison for sorting terms.
    #     Compares first by predicate, then by arguments.
    #     """
    #     if not isinstance(other, Term):
    #         return NotImplemented
    #     if self.predicate != other.predicate:
    #         return self.predicate < other.predicate
    #     return self.args < other.args

@dataclasses.dataclass
class Rule:
    """
    Represents a logical rule with a head term and a body of terms.
    """
    head: Term
    body: List[Term]

    def __str__(self):
        """String representation of the rule in the form 'head :- body'."""
        body_str = ", ".join(str(term) for term in self.body)
        return f"{self.head} :- {body_str}"

    def __repr__(self):
        """Representation is the same as the string form for readability."""
        body_str = ", ".join(str(term) for term in self.body)
        return f"{self.head} :- {body_str}"


@functools.lru_cache(maxsize=1000000)
def get_atom_from_string(atom_str: str) -> Term:
    """
    Optimized version using string methods instead of regex.
    Handles optional trailing '.' and atoms with no arguments.
    """
    # --- Basic String Cleaning ---
    s = atom_str.strip().removesuffix('.')

    # --- Find Predicate and Arguments ---
    try:
        paren_open_idx = s.index('(')
        # Ensure the string ends with ')' for valid atoms with args
        if not s.endswith(')'):
             raise ValueError(f"Malformed atom string (missing closing parenthesis): '{atom_str}'")

        predicate = s[:paren_open_idx]
        args_content = s[paren_open_idx + 1:-1] # Content between '(' and ')'

        if not args_content: # Handles "predicate()"
             args_tuple = tuple()
        else:
             # Split, strip, and filter empty strings in one go
             args_tuple = tuple(stripped
                                for arg in args_content.split(',')
                                if (stripped := arg.strip())) 
    except ValueError: # Handles cases where '(' is not found, e.g., "fact"
        predicate = s
        args_tuple = tuple()

    # --- Basic Validation (Optional but Recommended) ---
    # Ensure predicate is not empty
    if not predicate:
         raise ValueError(f"Empty predicate found in atom string: '{atom_str}'")
    # Could add checks for valid characters if needed, but often omitted for speed

    # --- Create Term ---
    # Assuming Term constructor takes predicate (str) and args (tuple)
    return Term(predicate, args_tuple)


def get_rule_from_string(rule_str: str) -> Rule:
    """
    Convert a rule string into a Rule object with head and body.

    Args:
        rule_str (str): Rule in the form 'head :- body' or a standalone fact.

    Returns:
        Rule: Structured rule containing head Term and list of body Terms.
    """
    if ":-" not in rule_str:
        head = rule_str.strip()
        rule = Rule(get_atom_from_string(head), [])
    else:
        head, body = rule_str.strip().split(":-")
        body = re.findall(r'\w+\(.*?\)', body)
        body = [get_atom_from_string(b) for b in body]

        head_atom = get_atom_from_string(head)
        rule = Rule(head_atom, body)
    return rule


def apply_substitution(term: Term, substitution: Dict[str, str]) -> Term:
    """
    Apply a variable substitution to a Term's arguments.

    Args:
        term (Term): Term object containing original arguments.
        substitution (Dict[str, str]): Mapping of variable names to replacement strings.

    Returns:
        Term: New Term with substituted argument values.
    """
    substituted_args_list = [substitution.get(arg, arg) for arg in term.args]
    return Term(term.predicate, tuple(substituted_args_list))


def is_variable(arg: str) -> bool:
    """
    Check if a string represents a logic variable.

    Args:
        arg (str): Argument name to evaluate.

    Returns:
        bool: True if name starts with an uppercase letter or underscore.
    """
    return arg[0].isupper() or arg[0] == '_'


def extract_var(state: str) -> list:
    """
    Extract unique variable names from a state string.

    Args:
        state (str): String containing terms with variables.

    Returns:
        list: Unique variable identifiers found.
    """
    pattern = r'\b[A-Z_][a-zA-Z0-9_]*\b'
    vars = re.findall(pattern, state)
    return list(dict.fromkeys(vars))


# ----------------------
# Environment utils       
# ----------------------


def simple_rollout(env, policy=None, batch_size: int=2, steps: int=10, tensordict: TensorDict = None, verbose: int=0) -> TensorDict:
    """
    Perform a basic rollout in an environment, collecting TensorDicts over time.

    Args:
        env: Environment providing reset, step, and action_spec.sample().
        policy: Optional policy with forward_dict method; if None, actions are random.
        batch_size (int): Number of parallel episodes.
        steps (int): Maximum number of steps per episode.
        tensordict (TensorDict, optional): Initial TensorDict for reset.
        verbose (int): Verbosity level (0=silent, >0=debug prints).

    Returns:
        TensorDict: Time-stacked TensorDict of observations, actions, rewards, etc.
    """
    data = []
    if tensordict is None:
        _data = env.reset(env.gen_params(batch_size=[batch_size]))
    else:
        _data = env.reset(tensordict)
    for i in range(steps):
        print('i', i, '------------------------------------') if verbose > 0 else None
        _data["action"] = env.action_spec.sample() if policy is None else policy.forward_dict(_data)["action"]
        _data = env.step(_data)

        if verbose > 0:
            for state, action, derived_states, reward, done in zip(_data['state'], _data['action'], _data['derived_states'], _data['reward'], _data['done']):
                print(*state, '-> action', action.item(), '/', len(derived_states)-1)
                print('reward', reward)
                print('Done', done)
                print('     Derived states:', *derived_states, '\n')

        print('actions', _data['action'], 'rewards', _data['reward'], 'dones', _data['done']) if verbose > 0 else None
        data.append(_data)  # We append it here because we want to keep the "next" data. Those will be datapoint samples
        if _data["done"].all():
            print('\nDONE', _data["done"]) if verbose > 0 else None
            break
        _data = step_mdp(_data, keep_other=True, exclude_reward=False, exclude_done=False, exclude_action=False)

    data = TensorDict.stack(data, dim=1)
    return data

def print_eval_info(split_name: str, metrics: Dict[str, float]):
    """
    Display evaluation metrics for a specified dataset split.

    Args:
        split_name (str): Name of the dataset split (e.g. 'train', 'valid', 'test').
        metrics (Dict[str, float]): Mapping of metric names to their values.
    """

    def _format_stat(mean: Optional[float], std: Optional[float], count: Optional[int]) -> str:
        if mean is None:
            return "N/A"
        if std is None and count is None:
            return f"{mean:.3f}"
        if std is None:
            return f"{mean:.3f} (n={count})" if count is not None else f"{mean:.3f}"
        if count is None:
            return f"{mean:.3f} +/- {std:.2f}"
        return f"{mean:.3f} +/- {std:.2f} ({count})"

    print(f'\n\n{split_name} set metrics:')
    grouped: Dict[str, Dict[str, Optional[float]]] = {}
    grouped_suffixes = {"mean", "std", "count"}
    handled_keys: set[str] = set()

    for key, value in metrics.items():
        if not isinstance(value, (int, float, np.integer, np.floating)):
            continue
        for suffix in grouped_suffixes:
            token = f"_{suffix}"
            if key.endswith(token):
                base = key[: -len(token)]
                if base not in grouped:
                    grouped[base] = {"mean": None, "std": None, "count": None}
                grouped[base][suffix] = float(value)
                handled_keys.add(key)
                break

    final_output = {}

    for base, stats in grouped.items():
        count_val = stats.get("count")
        display = _format_stat(stats.get("mean"), stats.get("std"), int(count_val) if count_val is not None else None)
        final_output[base] = display

    for key, value in metrics.items():
        if key in handled_keys:
            continue
        if isinstance(value, (float, np.floating)):
            final_output[key] = f"{value:.3f}"
        elif isinstance(value, (int, np.integer)):
            final_output[key] = f"{value}"
        else:
            final_output[key] = f"{value}"

    for key in sorted(final_output.keys()):
        print(f"{key}: {final_output[key]}")

def print_state_transition(state, derived_states, reward, done, action=None, truncated=None, label=None):
    """
    Print details of a single state transition.

    Args:
        state: Current state representation.
        derived_states: Possible next states list.
        reward: Received reward tensor.
        done: Done flag tensor.
        action: Executed action tensor (optional).
        truncated: Truncated flag (optional).
        label: Additional label for display (optional).
    """
    if action is not None:
        action_val = action.item() if hasattr(action, 'item') else action
        print('\nState', state, '( action', action_val, ')')
        print('Reward', reward.item(), 'Done', done.item())
        if truncated is not None or truncated!=False: print(f" Truncated {truncated}")
        print('     Derived states:', *derived_states[:100])
        if len(derived_states) > 100:
            print('     ... in total', len(derived_states),'\n')
    else:
        print('\nState', state, label) if label is not None else print(state)
        print('Reward', reward.item(), 'Done', done.item())
        if truncated is not None or truncated!=False: print(f" Truncated {truncated}")
        print('     Derived states:', *derived_states[:100])
        if len(derived_states) > 100:
            print('     ... in total', len(derived_states),'\n')

def print_rollout(data):
    """
    Print a rollout first by batch, then transposed by timestep.

    Args:
        data: Stacked TensorDict of rollout (batch x time).
    """
    # Print data by batch
    for i, batch_data in enumerate(data):
        print(f'Batch {i}')
        print_td(batch_data)
        print('\n')
    print('\n')

    # Print data by step after transposing
    data = data.transpose(0, 1)
    for i, step_data in enumerate(data):
        print(f'Step {i}:', [[str(atom) for atom in state] for state in step_data['state']])
        for j, state in enumerate(step_data['state']):
            print(f'     Step {i}, Batch {j}:', [str(atom) for atom in state])
    print('\n')


def print_td(td: TensorDictBase, next=False, exclude_states=False):
    """
    Pretty-print the contents of a TensorDict, showing nested next states if present.

    Args:
        td (TensorDictBase): TensorDict to display.
        next (bool): If True, label as 'Next TensorDict'.
        exclude_states (bool): Skip printing 'state' and 'derived_states' entries.
    """
    print_title = 'Next TensorDict' if next else 'TensorDict'
    print(f'{"="*10} {print_title} {"="*10}')
    
    for key, value in td.items():
        if (key == 'derived_states' and not exclude_states):
            value_data = value.data
            print(f'Key: {key}', value_data)
            for i, batch in enumerate(value_data):
                for j, next_state in enumerate(batch):
                    print(f'     {i}, {j} next_possible_state:', [str(atom) for atom in next_state])

        elif (key == 'state' and not exclude_states):
            value_data = value.data
            for i, state in enumerate(value_data):
                print(f'     {i} state:', [str(atom) for atom in state])

        elif key == 'next':
            print(f'Key: {key}')
            print_td(value, next=True, exclude_states=exclude_states)

        elif key not in {'state', 'derived_states'}:
            if isinstance(value, torch.Tensor):
                print(f'Key: {key} Shape: {value.shape} Values:\n{value}')
            elif isinstance(value, list):
                print(f'Key: {key} Length: {len(value)} Values:\n{value}')
            else:
                print(f'Key: {key} Value:\n{value}')
    
    print("="*30 if not next else "^"*30)


def get_device(device: Union[torch.device, str] = "auto") -> torch.device:
    """
    Resolve and return a torch.device, preferring CUDA if available.

    Args:
        device (Union[torch.device, str]): 'auto', 'cuda', 'cpu' or torch.device.

    Returns:
        torch.device: Resolved compute device (CPU or GPU).
    """
    # Cuda by default
    if device == "auto":
        device = "cuda"
    # Force conversion to torch.device
    device = torch.device(device)

    # Cuda not available
    if device.type == torch.device("cuda").type and not torch.cuda.is_available():
        return torch.device("cpu")

    return device




# ----------------------
# File logger
# ----------------------


class FileLogger:
    """A class for logging experiment results to files."""

    def __init__(self, base_folder: str = './log_folder'):
        """
        Initialize FileLogger and create directory structure.

        Args:
            base_folder (str): Root folder for logs and runs.
        """
        self.folder = base_folder
        self.folder_experiments = os.path.join(base_folder, 'averaged_runs')
        self.folder_run = os.path.join(base_folder, 'indiv_runs')
        self.date = self._get_formatted_date()

        self._create_directories()

    def _create_directories(self) -> None:
        """
        Ensure that base, averaged_runs, and indiv_runs directories exist.
        """
        for folder in [self.folder, self.folder_experiments, self.folder_run]:
            os.makedirs(folder, exist_ok=True)

    @staticmethod
    def _get_formatted_date() -> str:
        """
        Generate a timestamp string in 'YYYY_MM_DD_HH_MM_SS' format.

        Returns:
            str: Formatted current date and time.
        """
        return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    def _write_header(self, filename: str, headers: Iterable[str]) -> None:
        """
        Write a CSV header line to a new file.

        Args:
            filename (str): File path for the header.
            headers (Iterable[str]): Column names to write.
        """
        try:
            with open(filename, "w") as f:
                f.write(",".join(map(str, headers)))
        except IOError as e:
            print(f"Error writing header to file {filename}: {e}")

    def log(self, filename: str, args: Dict[str, Any], dicts: Dict[str, Dict[str, Any]]) -> None:
        """
        Append run arguments and associated metric dicts to a log file.

        Args:
            filename (str): Path to the log file.
            args (Dict[str, Any]): Key-value arguments for this run.
            dicts (Dict[str, Dict[str, Any]]): Named metric dictionaries.
        """
        header_filename = os.path.join(self.folder_run, "header.txt")
        
        if not os.path.exists(header_filename):
            self._write_header(header_filename, args.keys())

        try:
            with open(filename, "a") as f:
                f.write("\nAll data;")
                f.write(";".join(f'{k}:{v}' for k, v in args.items()))
                f.write('\n')
                for name, dictionary in dicts.items():
                    f.write(f'{name};')
                    f.write(";".join(f'{k}:{v}' for k, v in dictionary.items())) if dictionary else f.write("None")
                    f.write("\n")

        except IOError as e:
            print(f"Error writing to file {filename}: {e}")

    def finalize_log_file(self, tmp_filename: str, log_filename_run: str) -> None:
        """
        Rename a temporary log file to its final run-specific name.

        Args:
            tmp_filename (str): Temporary file path.
            log_filename_run (str): Destination file path.
        """
        try:
            os.rename(tmp_filename, log_filename_run)
        except OSError as e:
            print(f"Error renaming log file: {e}")

    def exists_run(self, run_signature: str, seed: int) -> bool:
        """
        Check whether a run file with a given signature and seed exists.

        Args:
            run_signature (str): Unique identifier in filenames.
            seed (int): Seed value included in filename.

        Returns:
            bool: True if corresponding run file is found.
        """
        files_with_signature = [file for file in os.listdir(self.folder_run) if run_signature in file]

        if len(files_with_signature) == 0:
            return False
        for file in files_with_signature:
            if f'seed_{seed}' in file:
                print("Seed number ", seed, 'already done')
                return True
        return False        

    def exists_experiment(self, args: Dict[str, Any]) -> bool:
        """
        Determine if averaged experiment results already exist.

        Args:
            args (Dict[str, Any]): Experiment parameters including 'run_signature'.

        Returns:
            bool: True if an experiment file with the signature is present.
        """
        experiment_file = os.path.join(self.folder_experiments, 'experiments.csv')
        if not os.path.exists(experiment_file):
            return False

        experiments_files = [f for f in os.listdir(self.folder_experiments) if f.startswith('experiments')]
        if len(experiments_files) == 0:
            return False

        for file in experiments_files:
            with open(os.path.join(self.folder_experiments, file), 'r') as f:
                lines = f.readlines()
                headers = None
                for j, line in enumerate(lines):
                    if 'run_signature' in line:
                        headers = line.split(';')
                        pos_run_signature = headers.index('run_signature')
                    if headers is not None:
                        try:
                            file_signature = line.split(';')[pos_run_signature]
                            if file_signature == args['run_signature']:
                                print("Skipping training, it has been already done for", args['run_signature'], "\n")
                                return True
                        except:
                            continue
        return False

    def _parse_line(self, line: str) -> Dict[str, Any]:
        """
        Parse a semicolon-delimited log line into a typed dictionary.

        Args:
            line (str): Raw log line starting with a label then key:value entries.

        Returns:
            Dict[str, Any]: Parsed data with appropriate Python types.
        """
        data = line.strip().split(';')[1:]
        data_dict = {}
        for el in data:
            if ':' not in el:
                continue
            d_key, raw_value = el.split(':', 1)
            parsed_value = self._coerce_metric_value(raw_value)
            if (
                isinstance(parsed_value, dict)
                and {'mean', 'std', 'count'} <= set(parsed_value.keys())
            ):
                for suffix, numeric_value in parsed_value.items():
                    data_dict[f"{d_key}_{suffix}"] = numeric_value
            else:
                data_dict[d_key] = parsed_value
        return data_dict

    @staticmethod
    def _coerce_metric_value(raw_value: str) -> Any:
        """Attempt to convert a raw string metric into a numeric type."""
        value_str = raw_value.strip()
        try:
            return ast.literal_eval(value_str)
        except Exception:
            pass

        metric_match = re.match(
            r"^([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\+/-\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\((\d+)\)$",
            value_str,
        )
        if metric_match:
            return {
                'mean': float(metric_match.group(1)),
                'std': float(metric_match.group(2)),
                'count': int(metric_match.group(3)),
            }

        try:
            return float(value_str)
        except ValueError:
            return value_str

    def log_avg_results(self, args_dict: Dict[str, Any], run_signature: str, seeds: List[int]) -> None:
        """
        Aggregate and average results across multiple run log files.

        Args:
            args_dict (Dict[str, Any]): Base arguments common to all runs.
            run_signature (str): Identifier present in each run filename.
            seeds (List[int]): List of seed integers used for runs.
        """
        all_files = os.listdir(self.folder_run)
        run_files = [file for file in all_files if run_signature in file]

        if len(run_files) < len(seeds):
            print(f'Number of files {len(run_files)} < number of seeds {len(seeds)}!')
            return None, None

        avg_results = defaultdict(list)
        seeds_found = set()

        for file in run_files:
            file_path = os.path.join(self.folder_run, file)
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith('All data'):
                        data = self._parse_line(line)
                        results_time = {k: v for k, v in data.items() if k in 
                                        ['time_train', 'time_inference', 'time_ground_train', 'time_ground_valid', 'time_ground_test']}
                        for name in results_time.keys():
                            avg_results[name].append(results_time[name])
                        seed = data['seed_run_i']
                        if seed in seeds:
                            seeds_found.add(seed)
                    
                    if line.split(';')[0] in {'train', 'valid', 'test'}:
                        data = self._parse_line(line)
                        dataset = line.split(';')[0]
                        data = {f'{dataset}_{k}': v for k, v in data.items()}
                        for name in data.keys():
                            avg_results[name].append(data[name])

        len_keys = [len(v) for v in avg_results.values()]
        assert all([l == len_keys[0] for l in len_keys]), f'Not all the keys in avg_results have the same length! {[(k, len(v)) for k, v in avg_results.items()], avg_results}'                  
        assert len(seeds_found) == len(seeds), f'Number of seeds {seeds_found} found in the experiments is different from the number of seeds {seeds}!'
        
        avg_results = {key: [np.mean(values), np.std(values)] for key, values in avg_results.items()}
        self.write_avg_results(args_dict, avg_results)

    def write_avg_results(self, args_dict, avg_results: Dict[str, List[List[float]]]) -> None:
        """
        Write averaged metrics and parameters to the experiments CSV.

        Args:
            args_dict (Dict[str, Any]): Experiment configuration parameters.
            avg_results (Dict[str, List[List[float]]]): Metric means and standard deviations.
        """
        file_csv = os.path.join(self.folder_experiments, 'experiments.csv')
        
        if 'contrastive_loss' in args_dict:
            args_dict.remove('contrastive_loss')

        column_names = list(args_dict.keys()) + list(avg_results.keys())
        column_names = ';'.join(column_names)

        values_args = [str(v) for k, v in args_dict.items()]
        values_avg_results = [str([float(np.round(v[0], 3)), float(np.round(v[1], 3))]) for k, v in avg_results.items()]
        combined_results = ';'.join(values_args + values_avg_results)

        print("Writing results to", file_csv)
        with open(file_csv, 'a') as f:
            empty = os.stat(file_csv).st_size == 0
            if empty:
                f.write('sep=;\n')
                f.write(column_names)
            f.write('\n')
            f.write(combined_results)




# ----------------------
# Train utils       
# ----------------------

def profile_code(profiler_type, function_to_profile, *args, **kwargs):
    """
    Profiles a function using either cProfile or torch.profiler.
    """
    if profiler_type == "cProfile":
        profiler = cProfile.Profile()
        profiler.enable()
        result = function_to_profile(*args, **kwargs)
        profiler.disable()

        s = io.StringIO()
        stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        stats.print_stats(30)
        print("\n--- cProfile Cumulative Time ---")
        print(s.getvalue())

        s = io.StringIO()
        stats = pstats.Stats(profiler, stream=s).sort_stats('tottime')
        stats.print_stats(30)
        print("\n--- cProfile Total Time ---")
        print(s.getvalue())

        return result

    elif profiler_type == "torch":
        prof_activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            prof_activities.append(ProfilerActivity.CUDA)
        
        trace_dir = "./profiler_traces"
        os.makedirs(trace_dir, exist_ok=True)

        with torch.profiler.profile(
            activities=prof_activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            result = function_to_profile(*args, **kwargs)
        
        sort_key = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
        print(prof.key_averages().table(sort_by=sort_key, row_limit=30))
        
        trace_file = f"{trace_dir}/{function_to_profile.__name__}_trace.json"
        try:
            prof.export_chrome_trace(trace_file)
            print(f"--- Trace exported to {trace_file}. ---")
        except Exception as e:
            print(f"--- Failed to export trace: {e} ---")

        return result

    else:
        # print("No valid profiler specified, running function without profiling.")
        return function_to_profile(*args, **kwargs)

def _set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def _freeze_dropout_layernorm(m: nn.Module):
    if isinstance(m, (nn.Dropout, nn.LayerNorm)):
        m.eval()          # freeze statistics
        m.training = False



def _warn_non_reproducible(args: Any) -> None:
    if args.restore_best_val_model is False:
        print(
            "Warning: This setting is not reproducible when creating 2 models from scratch, "
            "but it is when loading pretrained models. You can use\n"
            "  export CUBLAS_WORKSPACE_CONFIG=:16:8; export PYTHONHASHSEED=0\n"
            "to make runs reproducible."
        )

def _maybe_enable_wandb(use_WB: bool, args: Any, WB_path: str, model_name: str):
    if not use_WB:
        return None
    return wandb.init(
        project="RL-NeSy",
        group=args.run_signature,
        name=model_name,
        dir=WB_path,
        sync_tensorboard=True,
        config=vars(args),
    )
