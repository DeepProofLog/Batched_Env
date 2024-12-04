import re
from typing import Dict, Union, List, Any, Tuple, Iterable, Optional
import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.envs.utils import step_mdp
import datetime
import os
import numpy as np
import ast
from collections import defaultdict


class Term:
    def __init__(self, predicate: str, args: List[str]):
        self.predicate = predicate  # Predicate name
        self.args = args  # List of arguments (constants or variables)

    def __str__(self):
        return f"{self.predicate}({', '.join(self.args)})"

    def __repr__(self):
        return f"{self.predicate}({', '.join(self.args)})"

def simple_rollout(env, policy = None, batch_size: int=2, steps: int=10, tensordict: TensorDict = None, verbose: int=0) -> TensorDict:
    ''' CAREFUL!!! pytroch doesnt stack the keys that are lists properly (for the tensors it should be fine). OR maybe it is because of the data_spec. Check it out'''
    data = []
    if tensordict is None:
        _data = env.reset(env.gen_params(batch_size=[batch_size]))
    else:
        _data = env.reset(tensordict)
    for i in range(steps):
        print('i', i,'------------------------------------') if verbose > 0 else None
        _data["action"] = env.action_spec.sample() if policy is None else policy.forward_dict(_data)["action"]
        _data = env.step(_data)

        if verbose > 0:
            for state, action, derived_states,reward,done in zip(_data['state'], _data['action'],_data['derived_states'],_data['reward'], _data['done']):
                print(*state, '-> action', action.item(),'/', len(derived_states)-1)
                print('reward',reward)
                print('Done',done)
                print('     Derived states:',*derived_states,'\n')
        
        print('actions',_data['action'],'rewards',_data['reward'],'dones',_data['done']) if verbose > 0 else None
        data.append(_data) # We append it here because we want to keep the "next" data. Those will be datapoint samples
        if _data["done"].all():
            print('\nDONE',_data["done"]) if verbose > 0 else None
            break
        _data = step_mdp(_data, keep_other=True,exclude_reward=False,exclude_done=False,exclude_action=False)

    data = TensorDict.stack(data, dim=1)
    return data

def apply_substitution(term: Term, substitution: Dict[str, str]) -> Term:
    """Apply the substitution to a given term."""
    substituted_args = [substitution.get(arg, arg) for arg in term.args]
    return Term(term.predicate, substituted_args)

def is_variable(arg: str) -> bool:
    """Check if an argument is a variable."""
    return arg[0].isupper() or arg[0] == '_'

def extract_var(state: str)-> list:
    '''Extract unique variables from a state: start with uppercase letter or underscore'''
    pattern = r'\b[A-Z_][a-zA-Z0-9_]*\b'
    vars = re.findall(pattern, state)
    return list(dict.fromkeys(vars))


def print_state_transition(state, derived_states, reward, done, action=None, truncated=None):
    if action is not None:
        for state_, action_, derived_states_, reward_, done_ in zip(state, action, derived_states, reward, done):
            print(*state_, '( action', action_.item(),')')
            # print(*state_, '-> action', action_.item(),'/', len(derived_states_)-1)
            print('Reward',reward_)
            print('Done',done_)
            print('Truncated',truncated) if truncated is not None else None
            print('     Derived states:',*derived_states_,'\n')
    else:
        print('Reset-----------')
        for state_, derived_states_, reward_, done_ in zip(state, derived_states, reward, done):
            print(*state_)
            print('Reward',reward_)
            print('Done',done_)
            print('Truncated',truncated) if truncated is not None else None
            print('     Derived states:',*derived_states_,'\n')

def print_rollout(data):
    """Prints each batch and then transposes to print each step in rollout data."""
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
    """Prints keys and values of a TensorDict, with optional flags for next states and excluding states."""
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
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.

    :param device: One for 'auto', 'cuda', 'cpu'
    :return: Supported Pytorch device
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


class FileLogger:
    """A class for logging experiment results to files."""

    def __init__(self, base_folder: str = './log_folder'):
        """
        Initialize the FileLogger.

        Args:
            base_folder (str): The base folder for all logs.
        """
        self.folder = base_folder
        self.folder_experiments = os.path.join(base_folder, 'averaged_runs')
        self.folder_run = os.path.join(base_folder, 'indiv_runs')
        self.date = self._get_formatted_date()

        self._create_directories()

    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        for folder in [self.folder, self.folder_experiments, self.folder_run]:
            os.makedirs(folder, exist_ok=True)

    @staticmethod
    def _get_formatted_date() -> str:
        """Get the current date and time in a formatted string."""
        return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    def _write_header(self, filename: str, headers: Iterable[str]) -> None:
        """
        Write the header to a file.

        Args:
            filename (str): The name of the file to write the header to.
            headers (Iterable[str]): The headers to write.
        """
        try:
            with open(filename, "w") as f:
                f.write(",".join(map(str, headers)))
        except IOError as e:
            print(f"Error writing header to file {filename}: {e}")

    def log(self, filename: str, args: Dict[str, Any], dicts: Dict[str, Dict[str, Any]]) -> None:
        """
        Append the results as the last line of a file. Each element should be appended as Name;key1:value1;key2:value2;...

        Args:
            filename (str): The name of the file to log to.
            args (Dict[str, Any]): A dictionary of arguments to log.
            kwargs (Dict[str, Any]): A dictionary of keyword arguments to log.
            
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
        Rename the temporary log file to its final name.

        Args:
            tmp_filename (str): The temporary filename to be renamed.
            log_filename_run (str): The final filename for the log file.
        """
        try:
            os.rename(tmp_filename, log_filename_run)
            # print(f"Log file renamed to: {log_filename_run}")
        except OSError as e:
            print(f"Error renaming log file: {e}")


    def exists_run(self, run_signature: str, seed: int) -> bool:
        """
        Check if a run with the given signature and seed already exists.

        Args:
            run_signature (str): Signature.
            seed (int): The seed used in the run.

        Returns:
            bool: True if the run exists, False otherwise.
        """
        # filter the files that contain the run_signature in self.folder_run
        files_with_signature = [file for file in os.listdir(self.folder_run) if run_signature in file]

        # If there are no files with the run_signature, return False, if there are files, check if the seed is in the filename
        if len(files_with_signature) == 0:
            return False
        for file in files_with_signature:
            if f'seed_{seed}' in file:
                print("Seed number ", seed,'already done')
                return True
        return False        

    def exists_experiment(self, args: Dict[str, Any]) -> bool:
        """
        Check if an experiment (avg of runs) with the given arguments already exists.

        Args:
            args (Dict[str, Any]): The arguments of the experiment to check.

        Returns:
            bool: True if the experiment exists, False otherwise.
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
                            if file_signature in args['run_signature']:
                                print("Skipping training, it has been already done for", args['run_signature'],"\n")
                                return True
                        except:
                            continue
        return False


    def _parse_line(self,line: str) -> Dict[str, Any]:
                                                                                                                                                                          
        """
        Parse a line of experiment data to extract metrics, times, metric names, and seed.

        Args:
            data (List[str]): List of data elements from a line.

        Returns:
            Dict[str, Any]: Dictionary containing metrics, times, metric names, and seed.
        """
        # Parse the data line to extract metrics, times, and seed
        data = line.strip().split(';')[1:]
        data_dict  = {}
        for el in data:
            [d_key, d_value] = el.split(':')
            try: # Try to convert it to a list or a number, otherwise it is a string
                d_value = ast.literal_eval(d_value)
            except:
                pass
            data_dict[d_key] = d_value
        return data_dict
    

    def log_avg_results(self, args_dict: Dict[str, Any], run_signature: str, seeds: List[int]) -> None:
        """
        Calculate the average results from multiple experiment runs with different seeds.

        Args:
            run_signature (str): Unique identifier present in the filenames of experiment result files.
            seeds (List[int]): List of seeds used in the experiments.

        Returns:
            Tuple[Dict[str, List[List[float]]], List[str]]: A tuple containing the average results dictionary and list of metric names.
        """
        # List all files in the run folder
        all_files = os.listdir(self.folder_run)
        
        # Filter files that contain the run_signature
        run_files = [file for file in all_files if run_signature in file]
        
        # Check if the number of filtered files matches the number of seeds
        if len(run_files) < len(seeds):
            print(f'Number of files {len(run_files)} < number of seeds {len(seeds)}!')
            return None, None
        
        # Initialize dictionaries and lists to store results
        avg_results = defaultdict(list)
        seeds_found = set()

        # Process each file
        for file in run_files:
            file_path = os.path.join(self.folder_run, file)
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    # Process lines starting with 'All data'
                    if line.startswith('All data'):
                        data = self._parse_line(line)

                        # Extract time info
                        results_time = {k: v for k, v in data.items() if k in 
                                        ['time_train', 'time_inference', 'time_ground_train', 'time_ground_valid', 'time_ground_test']}
                        for name in results_time.keys():
                            avg_results[name].append(results_time[name])
                        
                        # Extract seed info
                        seed = data['seed_run_i']
                        if seed in seeds:
                            seeds_found.add(seed)
                    
                    # Process lines starting with 'train', 'valid', or 'test'
                    if line.split(';')[0]=='train' or line.split(';')[0]=='valid' or line.split(';')[0]=='test':
                        data = self._parse_line(line)
                        dataset = line.split(';')[0]
                        # Append the dataset name (e.g. train,valid,test) to all the metrics, to differenciate train from valid from test
                        data = {f'{dataset}_{k}': v for k, v in data.items()}
                        for name in data.keys():
                            avg_results[name].append(data[name])

        # Check that all the kays in avg_results have the same length
        len_keys = [len(v) for v in avg_results.values()]
        assert all([l == len_keys[0] for l in len_keys]), 'Not all the keys in avg_results have the same length!'                  
        assert len(seeds_found) == len(seeds), f'Number of seeds {seeds_found} found in the experiments is different from the number of seeds {seeds}!'
        
        # Calculate average and standard deviation for each metric
        avg_results = {key: [np.mean(values), np.std(values)] for key, values in avg_results.items()}
        self.write_avg_results(args_dict,avg_results)
        
    

    def write_avg_results(self, args_dict, avg_results: Dict[str, List[List[float]]],) -> None:
        """
        Write average results to a CSV file along with experiment parameters.

        Args:
            args_dict (Dict[str, Any]): Dictionary containing experiment parameters.
            avg_results (Dict[str, List[List[float]]]): Dictionary containing average results and standard deviations.
            metrics_name (List[str]): List of metric names used in the experiment.
        """
        file_csv = os.path.join(self.folder_experiments, 'experiments.csv')
        
        if 'contrastive_loss' in args_dict:
            args_dict.remove('contrastive_loss')

        column_names = list(args_dict.keys()) + list(avg_results.keys())
        column_names = ';'.join(column_names)

        values_args = [str(v) for k, v in args_dict.items()]
        values_avg_results = [ str([np.round(v[0], 3), np.round(v[1], 3)]) for k, v in avg_results.items()]
        combined_results = ';'.join(values_args + values_avg_results)

        print("Writing results to", file_csv)
        with open(file_csv, 'a') as f:
            empty = os.stat(file_csv).st_size == 0
            if empty:
                f.write('sep=;\n')
                f.write(column_names)
            f.write('\n')
            f.write(combined_results)
