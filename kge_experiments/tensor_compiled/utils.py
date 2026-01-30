"""Shared utilities for logic parsing, profiling, and lightweight logging helpers."""

import re
import copy
from typing import Dict, Union, List, Any, Tuple, Iterable, Optional, Sequence, Mapping
import datetime
import os
import random
import numpy as np
import ast
from collections import defaultdict
import dataclasses
from dataclasses import dataclass, field
from functools import lru_cache
from tensordict import TensorDict, TensorDictBase
import functools
from pathlib import Path


import os
import numpy as np
import cProfile
import pstats
import io
import torch
import torch.profiler
from torch.profiler import ProfilerActivity
import torch.nn as nn

# Lazy import to avoid loading heavy optional dependencies in worker processes.
def _get_wandb() -> Any:
    """Lazy import of wandb to keep worker initialization light."""
    import wandb
    return wandb


# ----------------------
# Logic utils       
# ----------------------

@dataclass(frozen=True, order=False, slots=True)
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

def atom_to_str(atom: torch.Tensor, idx2predicate: Sequence[str], idx2constant: Sequence[str], n_constants: int, padding_idx: int = 0) -> str:
    """Convert a single atom tensor to string."""
    pred = int(atom[0].item())
    args = [int(x.item()) for x in atom[1:]]
    
    if pred == padding_idx:
        return "PAD"
        
    if idx2predicate and 0 <= pred < len(idx2predicate):
        pred_str = idx2predicate[pred]
    else:
        pred_str = f"?p{pred}"
        
    arg_strs = []
    for arg in args:
        if arg == padding_idx:
            arg_strs.append("PAD")
        elif arg > n_constants:
            # Variable
            arg_strs.append(f"Var_{arg}")
        elif idx2constant and 0 <= arg < len(idx2constant):
            # Constant
            arg_strs.append(idx2constant[arg])
        else:
            arg_strs.append(f"?c{arg}")
            
    if not arg_strs and pred_str in ["True", "False", "Endf"]:
        return f"{pred_str}"
        
    return f"{pred_str}({','.join(arg_strs)})"

def state_to_str(state: torch.Tensor, idx2predicate: Sequence[str], idx2constant: Sequence[str], n_constants: int, padding_idx: int = 0, **kwargs) -> str:
    """Convert state tensor to string. Handles both [M, 3] and [3] shapes."""
    if state.dim() == 1:
        return atom_to_str(state, idx2predicate, idx2constant, n_constants, padding_idx)
        
    valid = state[:, 0] != padding_idx
    if not valid.any():
        return ""
        
    atoms = state[valid]
    atom_strs = [atom_to_str(atom, idx2predicate, idx2constant, n_constants, padding_idx) for atom in atoms]
    return "|".join(atom_strs)


@lru_cache(maxsize=10000)
def get_atom_from_string(atom_str: str) -> Term:
    """
    Optimized version using string methods instead of regex.
    Handles optional trailing '.' and atoms with no arguments.
    Strips quotes from arguments for consistency.
    
    OPTIMIZATION: Cached with lru_cache to avoid re-parsing common atoms.
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
             # Also strip quotes from each argument for consistency
             args_tuple = tuple(stripped.strip('"\'')
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
        rule_str (str): Rule in the form 'head :- body', 'body -> head', or a standalone fact.
                       Also handles format like 'r1:1:body -> head' with prefix.

    Returns:
        Rule: Structured rule containing head Term and list of body Terms.
    """
    rule_str = rule_str.strip()
    
    # Handle format with prefix like 'r1:1:body -> head'
    # Remove the prefix (e.g., 'r1:1:')
    if ':' in rule_str and not ':-' in rule_str:
        # Check if it starts with a rule ID like 'r1:1:'
        parts = rule_str.split(':', 2)  # Split at most twice
        if len(parts) >= 3 and parts[0].startswith('r') and parts[1].isdigit():
            # Remove the prefix 'r1:1:'
            rule_str = parts[2].strip()
    
    # Handle ':-' format: 'head :- body'
    if ":-" in rule_str:
        head, body = rule_str.split(":-", 1)
        body = re.findall(r'\w+\(.*?\)', body)
        body = [get_atom_from_string(b) for b in body]
        head_atom = get_atom_from_string(head)
        rule = Rule(head_atom, body)
    # Handle '->' format: 'body -> head'
    elif "->" in rule_str:
        body, head = rule_str.split("->", 1)
        body = re.findall(r'\w+\(.*?\)', body)
        body = [get_atom_from_string(b) for b in body]
        head_atom = get_atom_from_string(head)
        rule = Rule(head_atom, body)
    # Standalone fact (no body)
    else:
        head = rule_str.strip()
        rule = Rule(get_atom_from_string(head), [])
    
    return rule


# ----------------------
# File logger
# ----------------------


class FileLogger:
    """A class for logging experiment results to files."""

    def __init__(self, base_folder: str = None):
        """
        Initialize FileLogger and create directory structure.

        Args:
            base_folder (str): Root folder for logs and runs.
        """
        if base_folder is None:
             base_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'runs')
            
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

    def get_tmp_log_filename(self, run_signature: str, date: str, seed: int) -> str:
        """
        Generate temporary log filename.
        """
        return os.path.join(
            self.folder,
            f"_tmp_log-{run_signature}-{date}-seed_{seed}.csv",
        )

    def log_run(self, args, train_metrics, valid_metrics, test_metrics, log_filename_tmp, date, seed):
        """
        Log run results and finalize log file.
        """
        # Create a clean copy excluding non-serializable attributes (like _components with tensors)
        if hasattr(args, '__dict__'):
            clean_dict = {k: v for k, v in args.__dict__.items()
                         if not k.startswith('_') and not hasattr(v, 'parameters')}
            logged_data = type(args).__new__(type(args))
            logged_data.__dict__.update(copy.deepcopy(clean_dict))
        else:
            logged_data = copy.deepcopy(args)
        dicts_to_log = {
            'train': train_metrics,
            'valid': valid_metrics,
            'test': test_metrics,
        }
        self.log(log_filename_tmp, logged_data.__dict__, dicts_to_log)

        # Extract scalar values from metrics (handle both float and list cases)
        rewards_pos_mean_val = test_metrics.get('rewards_pos_mean', 0)
        if isinstance(rewards_pos_mean_val, (list, np.ndarray)):
            rewards_pos_mean_val = np.mean(rewards_pos_mean_val)
        rewards_pos_mean = np.round(float(rewards_pos_mean_val), 3)
        
        mrr_val = test_metrics.get('mrr_mean', 0)
        if isinstance(mrr_val, (list, np.ndarray)):
            mrr_val = np.mean(mrr_val)
        mrr = np.round(float(mrr_val), 3)
        
        metrics = f"{rewards_pos_mean:.3f}_{mrr:.3f}"
        log_filename_run_name = os.path.join(
            self.folder_run,
            f"_ind_log-{args.run_signature}-{date}-{metrics}-seed_{seed}.csv",
        )
        self.finalize_log_file(log_filename_tmp, log_filename_run_name)

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

    def _flatten_metric_dict(self, prefix: str, value: Any):
        """Recursively flatten nested metric dictionaries."""
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                new_prefix = f"{prefix}_{sub_key}"
                yield from self._flatten_metric_dict(new_prefix, sub_value)
        else:
            yield prefix, value

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
            if isinstance(parsed_value, dict):
                for flat_key, flat_val in self._flatten_metric_dict(d_key, parsed_value):
                    data_dict[flat_key] = flat_val
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
        if not os.path.exists(self.folder_run):
            os.makedirs(self.folder_run)
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
                        for name, value in results_time.items():
                            if isinstance(value, (int, float, np.number)):
                                avg_results[name].append(value)
                        seed = data['seed_run_i']
                        if seed in seeds:
                            seeds_found.add(seed)
                    
                    if line.split(';')[0] in {'train', 'valid', 'test'}:
                        data = self._parse_line(line)
                        dataset = line.split(';')[0]
                        data = {f'{dataset}_{k}': v for k, v in data.items()}
                        for name, value in data.items():
                            if isinstance(value, (int, float, np.number)):
                                avg_results[name].append(value)

        # Filter out metrics with incomplete data (e.g., depth-specific metrics not present in all seeds)
        expected_len = len(seeds)
        avg_results = {k: v for k, v in avg_results.items() if len(v) == expected_len}

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
# Seeding       
# ----------------------

def seed_all(
    seed: int,
    deterministic: bool = False,
    deterministic_cudnn: bool = False,
    warn: bool = False
) -> None:
    """
    Set seeds for ALL random number generators globally.
    
    This is the CENTRAL seeding function - call this ONCE at the start
    of your script/training run (typically from runner.py).
    
    Args:
        seed: The seed value to use
        deterministic: If True, enables strict deterministic operations.
            - Sets torch.use_deterministic_algorithms(True)
            - May impact performance but ensures exact reproducibility
            - Set to False for production (faster, but non-reproducible)
        deterministic_cudnn: If True AND deterministic=True, sets CUDNN
            to deterministic mode. Ignored if deterministic=False.
        warn: If True, print a warning about deterministic mode
    
    Example:
        # For reproducible parity testing:
        seed_all(42, deterministic=True)
        
        # For production (faster):
        seed_all(42, deterministic=False)
    """
    import os
    
    # Core seeding - always done
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Deterministic mode - optional for performance
    if deterministic:
        # Set CUBLAS_WORKSPACE_CONFIG for deterministic CUDA matmul operations
        if torch.cuda.is_available():
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        
        # Enable deterministic algorithms for exact reproducibility
        torch.use_deterministic_algorithms(True, warn_only=False)
        print('ensuring determinism in the torch algorithm')
        
        if deterministic_cudnn and torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            if warn:
                print(
                    "Warning: This setting is not reproducible when creating "
                    "2 models from scratch, but it is when loading pretrained models."
                )
    else:
        # Non-deterministic mode - faster for production
        torch.use_deterministic_algorithms(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True  # Enable cuDNN autotuner

# ----------------------
# Config utils     
# ----------------------



BOOLEAN_TRUE = {'true', 't', 'yes', 'y', 'on', '1'}
BOOLEAN_FALSE = {'false', 'f', 'no', 'n', 'off', '0'}


def load_experiment_configs(config_path: str) -> Sequence[Dict[str, Any]]:
    """Load experiments from a YAML file containing an 'experiments' list."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "PyYAML is required to read experiment configuration files. "
            "Install it with `pip install pyyaml`."
        ) from exc

    data = yaml.safe_load(path.read_text())
    if data is None:
        raise ValueError("Configuration file is empty.")

    if not isinstance(data, Mapping) or "experiments" not in data:
        raise ValueError(
            "Configuration file must be a YAML mapping with an 'experiments' list."
        )

    experiments = data["experiments"]
    if not isinstance(experiments, Sequence):
        raise ValueError("The 'experiments' entry must be a list.")

    for idx, experiment in enumerate(experiments):
        if not isinstance(experiment, Mapping):
            raise ValueError(f"Experiment entry at index {idx} must be a dictionary.")

    return list(experiments)


def parse_scalar(
    text: str,
    *,
    boolean_true: Iterable[str] = BOOLEAN_TRUE,
    boolean_false: Iterable[str] = BOOLEAN_FALSE,
) -> Any:
    """Best-effort conversion of a string literal to Python types."""
    text = text.strip()
    if not text:
        return ''
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        lowered = text.lower()
        if lowered in boolean_true:
            return True
        if lowered in boolean_false:
            return False
        if lowered in {'none', 'null'}:
            return None
    return text


def coerce_config_value(
    key: str,
    value: Any,
    defaults: Mapping[str, Any],
    *,
    boolean_true: Iterable[str] = BOOLEAN_TRUE,
    boolean_false: Iterable[str] = BOOLEAN_FALSE,
) -> Any:
    """Match override type to default config entry."""
    if key not in defaults:
        raise ValueError(f"Unknown configuration key '{key}'.")

    default = defaults[key]

    if isinstance(default, list):
        if isinstance(value, (list, tuple)):
            return [copy.deepcopy(v) for v in value]
        return [value]

    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.lower()
            if lowered in boolean_true:
                return True
            if lowered in boolean_false:
                return False
            raise ValueError(f"Cannot parse boolean for '{key}': {value}")
        return bool(value)

    if isinstance(default, int) and not isinstance(default, bool):
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            return int(value)

    if isinstance(default, float):
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            return float(value)

    if isinstance(default, str):
        if value is None:
            return None
        return str(value)

    return copy.deepcopy(value)


def update_config_value(
    config: Dict[str, Any],
    key: str,
    value: Any,
    defaults: Mapping[str, Any],
    *,
    prevalidated: bool = False,
    boolean_true: Iterable[str] = BOOLEAN_TRUE,
    boolean_false: Iterable[str] = BOOLEAN_FALSE,
) -> None:
    """Update configuration dict, coercing value types unless already validated."""
    if key not in defaults:
        raise ValueError(f"Unknown configuration key '{key}'.")
    if not prevalidated:
        value = coerce_config_value(
            key,
            value,
            defaults,
            boolean_true=boolean_true,
            boolean_false=boolean_false,
        )
    if isinstance(value, (list, dict)):
        config[key] = copy.deepcopy(value)
    else:
        config[key] = value


def parse_assignment(entry: str) -> tuple[str, str]:
    if '=' not in entry:
        raise ValueError(f"Assignments must be in key=value format, got '{entry}'.")
    key, raw = entry.split('=', 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Invalid assignment '{entry}'.")
    return key, raw.strip()


def select_best_gpu(min_free_gb: float = 1.0) -> Optional[int]:
    """
    Automatically select the GPU with the most free memory.
    
    Args:
        min_free_gb: Minimum free memory in GB required to consider a GPU
    
    Returns:
        GPU index with most free memory, or None if no suitable GPU found
    """
    if not torch.cuda.is_available():
        return None
    
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        return None
    
    # Get free memory for each GPU
    max_free_memory = 0
    best_gpu = None
    min_free_bytes = min_free_gb * 1e9
    
    for gpu_id in range(num_gpus):
        try:
            # Query memory without setting device (to avoid OOM on full GPUs)
            free_memory, total_memory = torch.cuda.mem_get_info(gpu_id)
            used_memory = total_memory - free_memory
            
            print(f"GPU {gpu_id}: {free_memory / 1e9:.2f} GB free / {total_memory / 1e9:.2f} GB total "
                  f"({used_memory / 1e9:.2f} GB used, {100 * used_memory / total_memory:.1f}% utilized)")
            
            # Only consider GPUs with sufficient free memory
            if free_memory >= min_free_bytes and free_memory > max_free_memory:
                max_free_memory = free_memory
                best_gpu = gpu_id
        except Exception as e:
            print(f"Warning: Could not query GPU {gpu_id}: {e}")
            continue
    
    if best_gpu is not None:
        print(f"Selected GPU {best_gpu} with {max_free_memory / 1e9:.2f} GB free memory")
    else:
        print(f"No GPU found with at least {min_free_gb:.1f} GB free memory")
    
    return best_gpu


def get_available_gpus(min_free_gb: float = 1.0) -> List[int]:
    """
    Get list of all GPUs with sufficient free memory.
    
    Args:
        min_free_gb: Minimum free memory in GB required
    
    Returns:
        List of GPU indices with sufficient free memory
    """
    if not torch.cuda.is_available():
        return []
    
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        return []
    
    available_gpus = []
    min_free_bytes = min_free_gb * 1e9
    
    for gpu_id in range(num_gpus):
        try:
            free_memory, total_memory = torch.cuda.mem_get_info(gpu_id)
            used_memory = total_memory - free_memory
            
            print(f"GPU {gpu_id}: {free_memory / 1e9:.2f} GB free / {total_memory / 1e9:.2f} GB total "
                  f"({used_memory / 1e9:.2f} GB used, {100 * used_memory / total_memory:.1f}% utilized)")
            
            if free_memory >= min_free_bytes:
                available_gpus.append(gpu_id)
        except Exception as e:
            print(f"Warning: Could not query GPU {gpu_id}: {e}")
            continue
    
    return available_gpus



def select_device_with_min_memory(min_memory_gb: float = 2.0, use_multiple_gpus: bool = True) -> str:
    """
    Select device using GPUs with sufficient free memory.
    
    If use_multiple_gpus is False, selects the best single GPU.
    If use_multiple_gpus is True and multiple GPUs available, selects all of them
    and returns "cuda:0" (which maps to the GPU with most free memory).
    Sets CUDA_VISIBLE_DEVICES to restrict to selected GPU(s), ordered by free memory.
    
    Args:
        min_memory_gb: Minimum free memory in GB required
        use_multiple_gpus: If True, use all available GPUs instead of just the best one
    
    Returns:
        Device string ("cpu", "cuda:X", or "cuda:0" for multi-GPU with best GPU as primary)
    """
    available_gpus = get_available_gpus(min_free_gb=min_memory_gb)
    
    if len(available_gpus) == 0:
        print(f"No GPU with at least {min_memory_gb} GB free memory found.")
        print("Falling back to CPU")
        return "cpu"
    elif len(available_gpus) == 1 or not use_multiple_gpus:
        # Select single GPU (best if multiple available)
        if len(available_gpus) == 1:
            gpu_id = available_gpus[0]
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            print(f"Selected GPU {gpu_id}")
            return f"cuda:{gpu_id}"
        else:
            # Select the GPU with most free memory
            max_free_memory = 0
            best_gpu = None
            
            for gpu_id in available_gpus:
                free_memory, _ = torch.cuda.mem_get_info(gpu_id)
                if free_memory > max_free_memory:
                    max_free_memory = free_memory
                    best_gpu = gpu_id
            
            if best_gpu is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
                print(f"Selected best GPU {best_gpu} from {available_gpus} with {max_free_memory / 1e9:.2f} GB free")
                return f"cuda:{best_gpu}"
            else:
                print("Unexpected error selecting best GPU")
                return "cpu"
    else:
        # Use multiple GPUs - sort by free memory (highest first)
        gpu_memory_pairs = []
        for gpu_id in available_gpus:
            free_memory, _ = torch.cuda.mem_get_info(gpu_id)
            gpu_memory_pairs.append((gpu_id, free_memory))
        
        # Sort by free memory descending
        gpu_memory_pairs.sort(key=lambda x: x[1], reverse=True)
        sorted_gpus = [gpu_id for gpu_id, _ in gpu_memory_pairs]
        
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, sorted_gpus))
        best_gpu = sorted_gpus[0]
        best_memory = gpu_memory_pairs[0][1]
        print(f"Selected multiple GPUs: {sorted_gpus}")
        print(f"Primary GPU (cuda:0) is GPU {best_gpu} with {best_memory / 1e9:.2f} GB free memory")
        return f"cuda:0"  # PyTorch will see them as cuda:0, cuda:1, etc. with cuda:0 having most memory


__all__ = [
    'BOOLEAN_TRUE',
    'BOOLEAN_FALSE',
    'load_experiment_configs',
    'parse_scalar',
    'coerce_config_value',
    'update_config_value',
    'parse_assignment',
    'select_device_with_min_memory',
]


# ----------------------
# Other utils     
# ----------------------


def save_profile_results(profiler: cProfile.Profile, args: Any, device: Any, output_path: str = 'profile_results.txt', n_functions: int = 30):
    """
    Save profiling results to a file.
    
    Args:
        profiler (cProfile.Profile): The profiler object containing stats.
        args (Any): Configuration namespace (used for metadata).
        device (Any): Device used for training.
        output_path (str): Path to save the profile results.
        n_functions (int): Number of top functions to display.
    """
    with open(output_path, 'w') as f:
        f.write(f"Device: {device}\n")
        f.write(f"Total timesteps: {args.timesteps_train}\n")
        f.write(f"Dataset: {args.dataset_name}\n\n")
        
        ps = pstats.Stats(profiler, stream=f)
        ps.strip_dirs()
        f.write("="*80 + "\n")
        f.write("Top by Cumulative Time\n")
        f.write("="*80 + "\n")
        ps.sort_stats('cumulative')
        ps.print_stats(n_functions)
    
        f.write("\n\n" + "="*80 + "\n")
        f.write("Top by Total Time\n")
        f.write("="*80 + "\n")
        ps.sort_stats('tottime')
        ps.print_stats(n_functions)
    
    print(f"\nResults saved to {output_path}")



"""
Utility functions for KGE experiments.
"""

def print_results(results):
    """
    Prints experiment results in a structured, readable format using tabulate.
    """
    try:
        from tabulate import tabulate
    except ImportError:
        # Fallback if tabulate is not available
        print(f"\nResults summary:")
        for k, v in results.items():
            if k != 'per_mode':
                print(f"  {k}: {v}")
        return

    print("\n" + "═" * 80)
    print(f"║ {'EXPERIMENT RESULTS REPORT':^76} ║")
    print("═" * 80)

    res = results.copy()
    
    # --- 1. CORE PERFORMANCE METRICS ---
    core_keys = ['MRR', 'Hits@1', 'Hits@3', 'Hits@10', 'AUC_PR', 'success_rate']
    core_data = []
    for k in core_keys:
        if k in res:
            val = res.pop(k)
            formatted_val = f"{val:.4f}" if isinstance(val, (int, float)) else str(val)
            core_data.append([k, formatted_val])
    
    if core_data:
        print(f"\n[ Core Performance ]")
        print(tabulate(core_data, headers=["Metric", "Value"], tablefmt="fancy_grid"))

    # --- 2. TRAINING & ROLLOUT SUMMARY ---
    train_keys = ['ep_rew_mean', 'ep_len_mean', 'reward', 'len']
    train_data = []
    for k in train_keys:
        if k in res:
            val = res.pop(k)
            if isinstance(val, (int, float)):
                formatted_val = f"{val:.4f}"
            else:
                formatted_val = str(val)
            train_data.append([k, formatted_val])
            
    if train_data:
        print(f"\n[ Training & Rollout Summary ]")
        print(tabulate(train_data, headers=["Metric", "Value"], tablefmt="fancy_grid"))

    # --- 3. PER-MODE RANK METRICS ---
    if 'per_mode' in res:
        per_mode = res.pop('per_mode')
        print(f"\n[ Per-Mode Rank breakdown ]")
        # Get headers from first mode entry
        try:
            first_mode = next(iter(per_mode.values()))
            mode_headers = ["Mode"] + list(first_mode.keys())
            mode_rows = []
            for mode, metrics in per_mode.items():
                row = [mode]
                for m_key in mode_headers[1:]:
                    mv = metrics.get(m_key, "-")
                    row.append(f"{mv:.4f}" if isinstance(mv, (int, float)) else str(mv))
                mode_rows.append(row)
            print(tabulate(mode_rows, headers=mode_headers, tablefmt="fancy_grid"))
        except (StopIteration, AttributeError):
            pass

    # --- 4. DETAILED CATEGORY BREAKDOWN ---
    detailed_map = {} # tag -> {prefix -> value}
    prefixes = ('len_', 'reward_', 'proven_')
    
    for k in list(res.keys()):
        for p in prefixes:
            if k.startswith(p):
                val = res.pop(k)
                tag = k[len(p):]
                if tag not in detailed_map:
                    detailed_map[tag] = {}
                detailed_map[tag][p[:-1]] = val
                break
                
    if detailed_map:
        print(f"\n[ Detailed Result breakdown ]")
        det_headers = ["Category"] + [p[:-1] for p in prefixes]
        det_rows = []
        sorted_tags = sorted(detailed_map.keys(), key=lambda x: (x not in ['pos', 'neg'], x))
        
        for tag in sorted_tags:
            row = [tag]
            for p_key in det_headers[1:]:
                val = detailed_map[tag].get(p_key, "-")
                row.append(str(val))
            det_rows.append(row)
        print(tabulate(det_rows, headers=det_headers, tablefmt="fancy_grid"))

    # --- 5. MISCELLANEOUS ---
    if res:
        print(f"\n[ Other Metrics ]")
        misc_data = []
        for k in sorted(res.keys()):
            val = res[k]
            formatted_val = f"{val:.4f}" if isinstance(val, (int, float)) else str(val)
            misc_data.append([k, formatted_val])
        print(tabulate(misc_data, headers=["Metric", "Value"], tablefmt="grid"))

    print("\n" + "═" * 80 + "\n")
