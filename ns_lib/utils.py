
import os.path
import sys
import ast
import pandas as pd
from collections.abc import Iterable
import tensorflow as tf
import argparse
import numpy as np
import sys
from typing import Dict, Any, Optional
import datetime
# from tensorflow_ranking.python.utils import sort_by_scores, ragged_to_dense
from ns_lib.logic.commons import Atom, Domain, Rule, RuleLoader
from collections import defaultdict
from collections import defaultdict
import pickle
import json

def nested_dict(n, type):
    """
    Creates a nested defaultdict with n levels.

    Parameters:
    n (int): The number of nested levels for the defaultdict.
    type: The default data type for the defaultdict.

    Returns:
    defaultdict: A nested defaultdict with n levels.
    """
    if n == 1:
        return defaultdict(type)
    else:
        return defaultdict(lambda: nested_dict(n-1, type))

def ragged_to_dense(labels, predictions, weights):
  """Converts given inputs from ragged tensors to dense tensors.

  Args:
    labels: A `tf.RaggedTensor` of the same shape as `predictions` representing
      relevance.
    predictions: A `tf.RaggedTensor` with shape [batch_size, (list_size)]. Each
      value is the ranking score of the corresponding example.
    weights: An optional `tf.RaggedTensor` of the same shape of predictions or a
      `tf.Tensor` of shape [batch_size, 1]. The former case is per-example and
      the latter case is per-list.

  Returns:
    A tuple (labels, predictions, weights, mask) of dense `tf.Tensor`s.
  """
  _PADDING_LABEL = -1.
  _PADDING_PREDICTION = -1e6
  _PADDING_WEIGHT = 0.
  # TODO: Add checks to validate (ragged) shapes of input tensors.
  mask = tf.cast(tf.ones_like(labels).to_tensor(0.), dtype=tf.bool)
  labels = labels.to_tensor(_PADDING_LABEL)
  if predictions is not None:
    predictions = predictions.to_tensor(_PADDING_PREDICTION)
  if isinstance(weights, tf.RaggedTensor):
    weights = weights.to_tensor(_PADDING_WEIGHT)
  return labels, predictions, weights, mask


def _get_shuffle_indices(shape, mask=None, shuffle_ties=True, seed=None):
  """Gets indices which would shuffle a tensor.

  Args:
    shape: The shape of the indices to generate.
    mask: An optional mask that indicates which entries to place first. Its
      shape should be equal to given shape.
    shuffle_ties: Whether to randomly shuffle ties.
    seed: The ops-level random seed.

  Returns:
    An int32 `Tensor` with given `shape`. Its entries are indices that would
    (randomly) shuffle the values of a `Tensor` of given `shape` along the last
    axis while placing masked items first.
  """
  # Generate random values when shuffling ties or all zeros when not.
  if shuffle_ties:
    shuffle_values = tf.random.uniform(shape, seed=seed)
  else:
    shuffle_values = tf.zeros(shape, dtype=tf.float32)

  # Since shuffle_values is always in [0, 1), we can safely increase entries
  # where mask=False with 2.0 to make sure those are placed last during the
  # argsort op.
  if mask is not None:
    shuffle_values = tf.where(mask, shuffle_values, shuffle_values + 2.0)

  # Generate indices by sorting the shuffle values.
  return tf.argsort(shuffle_values, stable=True)

def sort_by_scores(scores,
                   features_list,
                   topn=None,
                   shuffle_ties=True,
                   seed=None,
                   mask=None):
  """Sorts list of features according to per-example scores.

  Args:
    scores: A `Tensor` of shape [batch_size, list_size] representing the
      per-example scores.
    features_list: A list of `Tensor`s to be sorted. The shape of the `Tensor`
      can be [batch_size, list_size] or [batch_size, list_size, feature_dims].
      The latter is applicable for example features.
    topn: An integer as the cutoff of examples in the sorted list.
    shuffle_ties: A boolean. If True, randomly shuffle before the sorting.
    seed: The ops-level random seed used when `shuffle_ties` is True.
    mask: An optional `Tensor` of shape [batch_size, list_size] representing
      which entries are valid for sorting. Invalid entries will be pushed to the
      end.

  Returns:
    A list of `Tensor`s as the list of sorted features by `scores`.
  """
  with tf.compat.v1.name_scope(name='sort_by_scores'):
    scores = tf.cast(scores, tf.float32)
    scores.get_shape().assert_has_rank(2)
    list_size = tf.shape(input=scores)[1]
    if topn is None:
      topn = list_size
    topn = tf.minimum(topn, list_size)

    # Set invalid entries (those whose mask value is False) to the minimal value
    # of scores so they will be placed last during sort ops.
    if mask is not None:
      scores = tf.where(mask, scores, tf.reduce_min(scores))

    # Shuffle scores to break ties and/or push invalid entries (according to
    # mask) to the end.
    shuffle_ind = None
    if shuffle_ties or mask is not None:
      shuffle_ind = _get_shuffle_indices(
          tf.shape(input=scores), mask, shuffle_ties=shuffle_ties, seed=seed)
      scores = tf.gather(scores, shuffle_ind, batch_dims=1, axis=1)

    # Perform sort and return sorted feature_list entries.
    _, indices = tf.math.top_k(scores, topn, sorted=True)
    if shuffle_ind is not None:
      indices = tf.gather(shuffle_ind, indices, batch_dims=1, axis=1)
    return [tf.gather(f, indices, batch_dims=1, axis=1) for f in features_list]

#######################################
# Utils.

def get_arg(args, name: str, default=None, assert_defined=False):
    value = getattr(args, name) if hasattr(args, name) else default
    if assert_defined:
        assert value is not None, 'Arg %s is not defined: %s' % (name, str(args))
    return value

def read_rules(path,args):
    rules = []
    with open(path, 'r') as f:
        for line in f:
            # if len(rules) < 11:
            # split by :
            line = line.split(':')
            # first element is the name of the rule
            rule_name = line[0]
            # second element is the weight of the rule
            rule_weight = float(line[1].replace(',', '.'))
            # third element is the rule itself. Split by ->
            rule = line[2].split('->')
            # second element is the head of the rule
            rule_head = rule[1]
            # remove the \n from the head and the space
            rule_head = [rule_head[1:-1]]
            # first element is the body of the rule
            rule_body = rule[0]
            # split the body by ,
            rule_body = rule_body.split(', ')
            # for every body element, if the last character is a " ", remove it
            for i in range(len(rule_body)):
                if rule_body[i][-1] == " ":
                    rule_body[i] = rule_body[i][:-1]
            # Take the vars of the body and head and put them in a dictionary
            all_vars = rule_body + rule_head
            var_names = {}
            for i in range(len(all_vars)):
                # split the element of the body by (
                open_parenthesis = all_vars[i].split('(')
                # Split the second element by )
                variables = open_parenthesis[1].split(')')
                # divide the variables by ,
                variables = variables[0].split(',')
                # Create a dictionary with the variables as keys and the value "countries" as values
                if 'nations' in args.dataset_name:
                    for var in variables:
                        var_names[var] = "countries"
                elif ('countries' in args.dataset_name) or ('ablation' in args.dataset_name):
                        var_names = {"X": "countries", "W": "subregions", "Z": "regions", "Y": "countries", "K": "countries"}
                        if 'nodomain' in args.dataset_name:
                            for var in variables:
                                var_names[var] = "cte" 
                elif 'kinship' in args.dataset_name:
                    # var_names = {"x": "people", "y": "people", "z": "people","a": "people", "b": "people","c": "people","d": "people"}
                    for var in variables:
                        var_names[var] = "cte" 
                else: 
                    for var in variables:
                        var_names[var] = "cte" 

            # print all the info
            # if len(rules) < 1001:
            # print('rule name: ', rule_name, 'rule weight: ', rule_weight, 'rule head: ', rule_head, 
            #         'rule body: ', rule_body, 'var_names: ', var_names)
            rules.append(Rule(name=rule_name,var2domain=var_names,body=rule_body,head=rule_head))
    print('number of rules: ', len(rules))
    return rules

# Recursive flattening to a 1D Iterable. TODO(check if this works).
# Example usages:
# Flatten([[1,2,3], 2, ['a','b']], tuple) -> (1,2,3,2,'a','b')
# Flatten([[1,2,3], 2, ['a','b']], list) -> [1,2,3,2,'a','b']
# Flatten(([1,2,3], 2, ('a','b')), list) -> [1,2,3,2,'a','b']
# Flatten(([1,2,3], 2, ('a','b')), np.array) -> np.array(1,2,3,2,'a','b')
def Flatten(lst: Iterable, flattening_function=tuple) -> Iterable:
    return flattening_function(Flatten(i, flattening_function)
                               if (not isinstance(elem, str) and
                                   isinstance(i, Iterable))
                               else i for i in lst)

# Can we just use the one above? Or this one? TODO cleanup.
def to_flat(nestedList):
    ''' Converts a nested list to a flat list '''
    flatList = []
    # Iterate over all the elements in given list
    for elem in nestedList:
        # Check if type of element is list
        if not isinstance(elem, str) and isinstance(elem, list):
            # Extend the flat list by adding contents of this element (list)
            flatList.extend(to_flat(elem))
        else:
            # Append the element to the list
            flatList.append(elem)
    return flatList

# Skips lines starting with comment_start, otherwise all lines are kept.
def read_file_as_lines(file, allow_empty=True, comment_start='#'):
    try:
        with open(file, 'r') as f:
            if comment_start:
                return [line.rstrip() for line in f.readlines()
                        if line[:len(comment_start)] != comment_start]
            else:
                return [line.rstrip() for line in f.readlines()]
    except IOError as e:
        if allow_empty:
            return []
        raise IOError("Couldn't open file (%s)" % file)

# Used by operation, TODO move to logic/commons.py or remove as there are already methods to do this.
def parse_atom(atom):
    spls = atom.split("(")
    atom_str = spls[0].strip()
    constant_str = spls[1].split(")")[0].split(",")

    return [atom_str] + [c.strip() for c in constant_str]


################################
# Callbacks.
class ActivateFlagAt(tf.keras.callbacks.Callback):
    """Activate a boolean tf.Variable at the beginning of a specific epoch"""

    def __init__(self, flag: tf.Variable, at_epoch : int):
        super().__init__()
        self.flag  = flag
        self.at_epoch = at_epoch

    def on_epoch_begin(self, epoch, logs=None):
        if epoch  == self.at_epoch - 1:
            print("Activating flag %s at epoch %d" % (self.flag.name, epoch + 1))
            self.flag.assign(True)

class PrintEachEpochCallback(tf.keras.callbacks.Callback):

    def __init__(self, model, fun):
        super().__init__()
        self.model  = model
        self.fun = fun

    def on_epoch_end(self, epoch, logs=None):
        print(self.fun(self.model))



def load_model_weights(model, ckpt_filepath, verbose=True):
    """
    Load the weights of a model from a checkpoint file.
    
    Args:
        model: Keras model instance
        ckpt_filepath: Path to the checkpoint file
        verbose: Print information about the loading process
    
    Returns:
        success: Boolean indicating whether the weights were loaded successfully
    """
    success = False

    # # .ckpt format
    # if os.path.exists(ckpt_filepath + '.ckpt' + '.index'):
    #     ckpt_filepath = ckpt_filepath + '.ckpt' + '.index'
    # else:
    #     files = os.listdir(os.path.dirname(ckpt_filepath))
    #     ckpt_filepaths = [f for f in files if '.ckpt' in f and '.index' in f and 'kge' not in f]
    #     if len(ckpt_filepaths) != 1:
    #         print('THERE IS NOT EXACTLY ONE FILE:', os.path.dirname(ckpt_filepath),'\n')
    #         return False
    #     ckpt_filepath = os.path.join(os.path.dirname(ckpt_filepath), ckpt_filepaths[0])
    
    # # .keras format
    # if os.path.exists(ckpt_filepath + '.keras'):
    #     ckpt_filepath = ckpt_filepath + '.keras'
    # else:
    #     print('THERE IS NO FILE:', ckpt_filepath + '.keras','\n')
    #     return False

    # .h5 format
    if os.path.exists(ckpt_filepath + '.weights.h5'):
        ckpt_filepath = ckpt_filepath + '.weights.h5'
    else:
        print('\nTHERE IS NO FILE:', ckpt_filepath + '.weights.h5','\n')
        return False
    

    print('ckpt_filepath', ckpt_filepath)
    if os.path.exists(ckpt_filepath):
        model.load_weights(ckpt_filepath)
        # checkpoint = tf.train.Checkpoint(model=model)
        # checkpoint.restore(ckpt_filepath)
        success = True
        if verbose:
            print('\nWeights loaded from', ckpt_filepath,'\n')
    else:
        if verbose:
            print('\nWeights not found in', ckpt_filepath,'\n')
    return success


def load_kge_weights(model, ckpt_filepath, verbose=True):
    """
    Load the weights of a model from a checkpoint file.
    
    Args:
        model: Keras model instance
        ckpt_filepath: Path to the checkpoint file
        verbose: Print information about the loading process
    
    Returns:
        success: Boolean indicating whether the weights were loaded successfully
    """
    success = False

    # # .ckpt format
    # if os.path.exists(ckpt_filepath + '.ckpt' + '.index'):
    #     ckpt_filepath = ckpt_filepath + '.ckpt' + '.index'
    # else:
    #     files = os.listdir(os.path.dirname(ckpt_filepath))
    #     ckpt_filepaths = [f for f in files if '.ckpt' in f and '.index' in f and 'kge' not in f]
    #     if len(ckpt_filepaths) != 1:
    #         print('THERE IS NOT EXACTLY ONE FILE:', os.path.dirname(ckpt_filepath),'\n')
    #         return False
    #     ckpt_filepath = os.path.join(os.path.dirname(ckpt_filepath), ckpt_filepaths[0])
    
    # # .keras format
    # if os.path.exists(ckpt_filepath + '.keras'):
    #     ckpt_filepath = ckpt_filepath + '.keras'
    # else:
    #     print('THERE IS NO FILE:', ckpt_filepath + '.keras','\n')
    #     return False

    # .h5 format
    if os.path.exists(ckpt_filepath + '.weights.h5'):
        ckpt_filepath = ckpt_filepath + '.weights.h5'
    else:
        print('THERE IS NO FILE:', ckpt_filepath + '.weights.h5','\n')
        return False
    
    if os.path.exists(ckpt_filepath):
        model.kge_model.load_weights(ckpt_filepath)
        # checkpoint = tf.train.Checkpoint(model=model.kge_model)
        # checkpoint.restore(ckpt_filepath)
        success = True
        if verbose:
            print('Weights loaded from', ckpt_filepath)
    else:
        if verbose:
            print('Weights not found in', ckpt_filepath)
    return success


# Checkpointer that allows both in-memory and filename checkpointing.
# This extends the functionalities of tf.keras.callbacks.ModelCheckpoint,
# which can save only to file and it is much slower for small
# non-persistent tests.

class MMapModelCheckpoint(tf.keras.callbacks.Callback):
    """Callback to save model weights when a metric improves."""
    
    def __init__(
        self,
        model: tf.keras.Model,
        monitor: str = 'val_loss',
        filepath: Optional[str] = None,
        save_weights_only: bool = True,
        save_best_only: bool = True,
        maximize: bool = True,
        verbose: bool = True,
        frequency: int = 1,
        name: str = None
    ):
        """Initialize checkpoint callback.
        
        Args:
            model: Model to monitor and save
            monitor: Metric name to monitor
            filepath: Path to save checkpoints. If None, weights are only kept in memory
            save_weights_only: If True, save only weights not full model
            save_best_only: If True, only save when metric improves
            maximize: If True, maximize metric, otherwise minimize
            verbose: Print messages when saving
            frequency: Frequency of checking for improvement in epochs
        """
        super().__init__()
        
        self.model_ = model
        self.monitor = monitor
        self.filepath = filepath
        self.save_weights_only = save_weights_only
        self.save_best_only = save_best_only
        self.maximize = maximize
        self.verbose = verbose
        self.frequency = frequency
        self.name = name

        # Initialize tracking variables
        self.best_value = -sys.float_info.max if maximize else sys.float_info.max
        self.best_weights = None
        self.best_epoch = None
        self.last_save_path = None

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        """Check for improvement on epoch end."""
        logs = logs or {}
        
        # Skip if not checking this epoch
        if (epoch + 1) % self.frequency != 0:
            return

        # Validate monitor metric exists
        if self.monitor not in logs:
            raise KeyError(f'Monitor metric "{self.monitor}" not found in logs. '
                         f'Available metrics: {",".join(logs.keys())}')

        current_value = logs[self.monitor]
        improved = (self.maximize and current_value > self.best_value) or \
                  (not self.maximize and current_value < self.best_value)
        if improved:
            if self.verbose:
                print(f'\nBest {self.name} {self.monitor}: {current_value:.5f}. '
                      f'Delta = {self.best_value-current_value:.5f}. Epoch {epoch+1}') if self.best_epoch is not None else None
            self.best_value = current_value
            self.best_epoch = epoch+1
            self.best_weights = self.model_.get_weights()

            # Save checkpoint and info
            if self.filepath:
                # FOR KERAS 2
                # save_path = f'{self.filepath}.ckpt'
                # checkpoint = tf.train.Checkpoint(model=self.model_)
                # checkpoint.save(save_path)

                # save_path = self.filepath
                # tf.saved_model.save(self.model_, save_path)

                # FOR KERAS 3
                # save_path = self.filepath
                # self.model_.save(f'{save_path}.keras')

                save_path = f'{self.filepath}.weights.h5'
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                if self.save_weights_only:
                    # Save only weights
                    self.model_.save_weights(save_path)
                else:
                    # Save full model
                    self.model_.save(save_path)

                self.last_save_path = save_path
                self.write_info(epoch+1, current_value)
                
                if self.verbose:
                    print(f'{self.name} weights saved') # to {save_path}')

    def restore_weights(self):
        """Restore the best weights."""
        if self.best_weights is not None:
            self.model_.set_weights(self.best_weights)
            if self.verbose:
                print(f'Restored best weights from epoch {self.best_epoch + 1}')
        else:
            print(f'No best weights to restore from {self.name}')

    def write_info(self, best_epoch: int, current_value: float):
        """Write checkpoint metadata to JSON file."""
        import json
        import datetime
        
        if self.filepath is None:
            return
            
        info = {
            'best_epoch': best_epoch,
            'best_value': float(current_value),  # Convert numpy float to Python float
            'metric': self.monitor,
            'timestamp': datetime.datetime.now().isoformat(),
            'maximize': self.maximize
        }
        
        # Save info next to checkpoint file
        info_path = f'{self.filepath}_info.json'
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
    
    def write_train_time(self,train_time: float):
        """Write checkpoint metadata to JSON file."""
        import json
        import datetime
        
        if self.filepath is None:
            return
            
        info = {
            'train_time': train_time,
        }
        
        # open the info file and append the train time
        info_path = f'{self.filepath}_info.json'
        with open(info_path, 'r') as f:
            data = json.load(f)
            data.update(info)
        with open(info_path, 'w') as f:
            json.dump(data, f, indent=2)
            




#############################################

from keras.callbacks import CSVLogger
import csv

class CustomCSVLogger(CSVLogger):
    def __init__(self, filename, separator=';', append=False):
        super().__init__(filename, separator, append)
        self.writer = None
        self.append_header = False
        self.separator = separator  # Explicitly store the separator

    def on_train_begin(self, logs=None):
        """Initialize the CSV file at the start of training."""
        logs = logs or {}
        mode = "a" if self.append else "w"

        # Check if we need to append the header
        if self.append and tf.io.gfile.exists(self.filename):
            with tf.io.gfile.GFile(self.filename, "r") as f:
                self.append_header = not bool(f.readline().strip())

        self.csv_file = tf.io.gfile.GFile(self.filename, mode)

    def on_train_end(self, logs=None):
        """Clean up and close the CSV file at the end of training."""
        if logs:
            header = ['epoch'] + list(logs.keys())
            self.csv_file.write(self.separator.join(header) + '\n')
        self.csv_file.close()
        self.writer = None

    def on_epoch_end(self, epoch, logs=None):
        """Write the metrics values to the CSV file at the end of each epoch."""
        logs = logs or {}
        
        if self.writer is None:
            fieldnames = ['epoch'] + list(logs.keys())
            self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames, delimiter=self.separator)
            if self.append_header:
                self.writer.writeheader()
                self.append_header = False

        row_dict = {'epoch': epoch, **logs}
        self.writer.writerow(row_dict)
        self.csv_file.flush()






import os
import datetime
import ast
import numpy as np
from typing import Dict, Iterable, List, Tuple, Any

class FileLogger:
    """A class for logging experiment results to files."""

    def __init__(self, base_folder: str = './log_folder'):
        """
        Initialize the FileLogger.

        Args:
            base_folder (str): The base folder for all logs.
        """
        self.folder = base_folder
        self.folder_experiments = os.path.join(base_folder, 'experiments')
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

    def _read_last_lines(self, filename: str, num_lines: int = 2) -> List[str]:
        """
        Read the last n lines of a file.

        Args:
            filename (str): The path to the file.
            num_lines (int): The number of lines to read from the end.

        Returns:
            List[str]: The last n lines of the file.
        """
        try:
            with open(filename, 'r') as file:
                lines = file.readlines()
                return lines[-num_lines:]
        except IOError as e:
            print(f"Error reading file {filename}: {e}")
            return []

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

                # f.write("\nAll data;")
                # f.write(";".join(f'{k}:{v}' for k, v in args.items()))
                # f.write(f'\nSignature;{args["run_signature"]}')

        except IOError as e:
            print(f"Error writing to file {filename}: {e}")

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

    def get_date(self) -> str:
        """
        Get the current formatted date.

        Returns:
            str: The current date in the format YYYY_MM_DD_HH_MM_SS.
        """
        self.date = self._get_formatted_date()
        return self.date

    def finalize_log_file(self, tmp_filename: str, log_filename_run: str) -> None:
        """
        Rename the temporary log file to its final name.

        Args:
            tmp_filename (str): The temporary filename to be renamed.
            log_filename_run (str): The final filename for the log file.
        """
        try:
            os.rename(tmp_filename, log_filename_run)
            print(f"Log file renamed to: {log_filename_run}")
        except OSError as e:
            print(f"Error renaming log file: {e}")

    def exists_experiment(self, args: Dict[str, Any]) -> bool:
        """
        Check if an experiment with the given arguments already exists.

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
                    # print('j, line', j, line)
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

    def write_to_csv(self, to_write: str) -> None:
        """
        Write log data to a CSV file.

        Args:
            to_write (str): The name of the CSV file to write to.
        """
        lines = []
        header = None
        for filename in os.listdir(self.folder):
            if filename.startswith("log"):
                last_line = self._read_last_lines(os.path.join(self.folder, filename), 1)
                lines.append(last_line)
            if filename.startswith("header"):
                header = self._read_last_lines(os.path.join(self.folder, filename), 1)
        
        with open(os.path.join(self.folder, to_write), "w") as f:
            if header:
                f.write(header[0] + "\n")
            for line in lines:
                f.write(line[0] + "\n")

    def _parse_line(self,line):
                                                                                                                                                                          
        """
        Parse a line of experiment data to extract metrics, times, metric names, and seed.

        Args:
            data (List[str]): List of data elements from a line.

        Returns:
            Tuple[Dict[str, float], Dict[str, float], List[str], int]: Parsed metrics, times, metric names, and seed.
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
    

    def get_avg_results(self, args_dict: Dict[str, Any], run_signature: str, seeds: List[int]) -> None:
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
        # avg_results_metrics = nested_dict(2, list)
        # avg_results_time = {}
        avg_results = {}
        seeds_found = set()


        # DO SOMETHING MODULAR TO GET INFO FROM EACH DICT FOR CERTAIN KEYS, AND CALCULATE THE AVG
        # Process each file
        for file in run_files:
            file_path = os.path.join(self.folder_run, file)
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    # Process lines starting with 'All data'
                    if line.startswith('All data'):
                        data = self._parse_line(line)

                        time_names = ['time_train', 'time_inference', 'time_ground_train', 'time_ground_valid', 'time_ground_test']
                        results_time = {k: v for k, v in data.items() if k in time_names}
                        # Append the results to the average results
                        for name in results_time.keys():
                            if name not in avg_results:
                                avg_results[name] = []
                            avg_results[name].append(results_time[name])

                        seed = data['seed_run_i']
                        if seed in seeds:
                            seeds_found.add(seed)
                        
                        # Include compatibility with older results, where the train\val\test metrics were not separated
                        if ('train_acc' in data) and ('valid_acc' in data) and ('test_acc' in data):
                            datasets = ['train', 'valid', 'test']
                            metrics_names = [metric for metric in data['metrics'] if 'val' not in metric  and 'lr' not in metric]
                            for dataset in datasets:
                                for i,name in enumerate(metrics_names):
                                    if dataset+'_'+name not in avg_results:
                                        avg_results[dataset+'_'+name] = [] 
                                    avg_results[dataset+'_'+name].append(data[dataset+'_acc'][i])

                    if line.split(';')[0]=='train' or line.split(';')[0]=='valid' or line.split(';')[0]=='test':
                        data = self._parse_line(line)
                        dataset = line.split(';')[0]
                        # Append the dataset name to all the metrics, to differenciate train from valid from test
                        data = {f'{dataset}_{k}': v for k, v in data.items()}
                        for name in data.keys():
                            if name not in avg_results:
                                avg_results[name] = []  
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
        values_avg_results = [ str([np.round(v[0], 3).item(), np.round(v[1], 3).item()]) for k, v in avg_results.items()]
        combined_results = ';'.join(values_args + values_avg_results)

        print("Writing results to", file_csv)
        with open(file_csv, 'a') as f:
            empty = os.stat(file_csv).st_size == 0
            if empty:
                f.write('sep=;\n')
                f.write(column_names)
            f.write('\n')
            f.write(combined_results)



class BinaryCrossEntropyRagged(tf.keras.losses.Loss):
    def __init__(self, balance_negatives=False, from_logits=False):
        super().__init__()
        self.balance_negatives = balance_negatives
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        if (isinstance(y_true, tf.RaggedTensor) or
            isinstance(y_pred, tf.RaggedTensor)):
            y_true = y_true.to_tensor()
            y_pred = y_pred.to_tensor()
        loss = tf.keras.losses.binary_crossentropy(
            y_true, y_pred, from_logits=self.from_logits)

        if self.balance_negatives:
            num_positives = tf.reduce_sum(tf.where(y_true == 1, 1.0, 0.0),
                                          axis=-1, keepdims=True)
            num_negatives = tf.reduce_sum(tf.where(y_true == 0, 1.0, 0.0),
                                          axis=-1, keepdims=True)
            loss_positive = tf.math.divide_no_nan(
                tf.where(y_true == 1, loss, 0.0),
                tf.expand_dims(num_positives, axis=-1))
            loss_negative = tf.math.divide_no_nan(
                tf.where(y_true == 0, loss, 0.0),
                tf.expand_dims(num_negatives, axis=-1))
            loss = loss_positive + loss_negative
        return loss

class PairwiseCrossEntropyRagged(tf.keras.losses.Loss):
    def __init__(self, balance_negatives=False, from_logits=False):
        super().__init__()
        self.balance_negatives = balance_negatives
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        if (isinstance(y_true, tf.RaggedTensor) or
            isinstance(y_pred, tf.RaggedTensor)):
            y_true = y_true.to_tensor()
            y_pred = y_pred.to_tensor()
        pos_loss = -tf.reduce_sum(tf.where(y_true == 1, tf.math.log(1e-7 + tf.nn.sigmoid(y_pred)), 0.0),
                                  axis=-1, keepdims=True)
        neg_loss = -tf.reduce_sum(tf.where(y_true == 0, tf.math.log(1e-7 + tf.nn.sigmoid(-y_pred)), 0.0),
                                  axis=-1, keepdims=True)
        if self.balance_negatives:
            num_positives = tf.reduce_sum(tf.where(y_true == 1, 1.0, 0.0),
                                          axis=-1, keepdims=True)
            num_negatives = tf.reduce_sum(tf.where(y_true == 0, 1.0, 0.0),
                                          axis=-1, keepdims=True)

        loss = tf.squeeze(pos_loss + neg_loss)
        return loss


class CategoricalCrossEntropyRagged(tf.keras.losses.Loss):
    def __init__(self, from_logits=False):
        super().__init__()
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        if (isinstance(y_true, tf.RaggedTensor) or
            isinstance(y_pred, tf.RaggedTensor)):
            y_true = y_true.to_tensor()
            y_pred = y_pred.to_tensor()
        return tf.keras.losses.categorical_crossentropy(
            y_true, y_pred, from_logits=self.from_logits)

######################################
# KGE utils and metrics, this is application dependent, TODO move to kge.py?
def KgeLossFactory(name: str) -> tf.keras.losses.Loss:
    if name == 'hinge':
        return HingeLossRagged(gamma=1.0)
    elif name == 'l2':
        return L2LossRagged()
    elif name == 'categorical_crossentropy':
        return CategoricalCrossEntropyRagged()
    elif name == 'binary_crossentropy':
        return BinaryCrossEntropyRagged()
    # include binary_crossentropy that converts the tensors to dense 
    # and then applies the loss, and then converts back to ragged
    elif name == 'balanced_binary_crossentropy':
        return BinaryCrossEntropyRagged(balance_negatives=True)
    elif name == 'balanced_pairwise_crossentropy':
        return PairwiseCrossEntropyRagged(balance_negatives=True)
    else:
        assert False, 'Unknown loss %s'% name


class EvalOnlyMetric(tf.keras.metrics.Metric):
    def __init__(self, name='eval_only_metric', **kwargs):
        super(EvalOnlyMetric, self).__init__(name=name, **kwargs)
        self.sum_of_values = self.add_weight(name='sum_of_values', initializer='zeros')
        self.num_updates = self.add_weight(name='num_updates', initializer='zeros', dtype=tf.int32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        learning_phase = tf.keras.backend.learning_phase()
        if learning_phase == 0:  # Evaluation phase
            # Example: Mean Absolute Error (MAE) - you can customize this
            values = tf.abs(y_true - y_pred)
            # Ensure values are not NaN or Inf. Otherwise, the metric will not update
            values = tf.where(tf.math.is_finite(values), values, tf.zeros_like(values))
            self.sum_of_values.assign_add(tf.reduce_sum(values))
            self.num_updates.assign_add(tf.cast(tf.shape(y_true)[0], tf.int32))

    def result(self):
        return tf.math.divide_no_nan(self.sum_of_values, tf.cast(self.num_updates, tf.float32))

    def reset_state(self):
        self.sum_of_values.assign(0.0)
        self.num_updates.assign(0)


class AUCPRMetric(tf.keras.metrics.AUC):
    """Implements AUC-PR metric with ragged tensor support."""
    def __init__(self, name='auc-pr', dtype=None, **kwargs):
        super().__init__(curve='PR', name=name, dtype=dtype, **kwargs)  # Add **kwargs

    def _compute(self, labels, predictions):
        if any(isinstance(tensor, tf.RaggedTensor)
               for tensor in [labels, predictions]):
            labels, predictions, _, _ = ragged_to_dense(labels, predictions, None)
        return super()._compute(labels, predictions)


class HitsMetric(tf.keras.metrics.Metric):
    """Implements HITS@N metric with ragged tensor support."""
    def __init__(self, n, name='hits', dtype=None, **kwargs):
        super().__init__(name=f'{name}@{n}', dtype=dtype, **kwargs)  # Keyword args for name/dtype
        self._n = n
        self.hits = self.add_weight(name="total", initializer="zeros")  # Explicit name
        self._count = self.add_weight(name="count", initializer="zeros")
        self.reset_state()

    def reset_state(self):
        self.hits.assign(0.)
        self._count.assign(0.)

    def result(self):
        return tf.math.divide_no_nan(self.hits, self._count)

    def update_state(self, y_true, y_pred, sample_weight=None):
        hits = self._compute(y_true, y_pred)
        self.hits.assign_add(tf.reduce_sum(hits))
        self._count.assign_add(tf.reduce_sum(tf.ones_like(hits)))

    def _compute(self, labels, predictions):
        if any(isinstance(tensor, tf.RaggedTensor)
               for tensor in [labels, predictions]):
            labels, predictions, _, _ = ragged_to_dense(labels, predictions, None)

        topn = tf.shape(predictions)[1]
        sorted_labels, = sort_by_scores(predictions, [labels], topn=topn, mask=None)
        relevance = tf.cast(tf.greater_equal(sorted_labels, 1.0), dtype=tf.float32)
        top_relevance = relevance[:, :self._n]
        hits = tf.reduce_sum(top_relevance, axis=1, keepdims=True)
        return hits

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'n': self._n}


class MRRMetric(tf.keras.metrics.Metric):
    """Implements mean reciprocal rank (MRR). It uses the same implementation
       of tensorflow_ranking MRRMetric but with an online check for ragged
       tensors."""
    def __init__(self, name='mrr', dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.mrr = self.add_weight(name="total", initializer="zeros")  # Explicitly name parameter
        self._count = self.add_weight(name="count", initializer="zeros")
        self.reset_state()

    def reset_state(self):
        self.mrr.assign(0.)
        self._count.assign(0.)

    def result(self):
        return tf.math.divide_no_nan(self.mrr, self._count)

    def update_state(self, y_true, y_pred, sample_weight=None):
        mrrs = self._compute(y_true, y_pred)
        self.mrr.assign_add(tf.reduce_sum(mrrs))
        self._count.assign_add(tf.reduce_sum(tf.ones_like(mrrs)))

    def _compute(self, labels, predictions):
        if any(isinstance(tensor, tf.RaggedTensor)
               for tensor in [labels, predictions]):
            labels, predictions, _, _ = ragged_to_dense(labels, predictions, None)

        topn = tf.shape(predictions)[1]
        sorted_labels, = sort_by_scores(predictions, [labels], topn=topn, mask=None)
        sorted_list_size = tf.shape(input=sorted_labels)[1]
        relevance = tf.cast(tf.greater_equal(sorted_labels, 1.0), dtype=tf.float32)
        reciprocal_rank = 1.0 / tf.cast(
            tf.range(1, sorted_list_size + 1), dtype=tf.float32)
        mrr = tf.reduce_max(
            input_tensor=relevance * reciprocal_rank, axis=1, keepdims=True)
        return mrr



def save_embeddings_from_model(model, fol, serializer, save_dir="embeddings"):
    """
    Extract and save embeddings from a trained KGEModel. Get the emb with the global indices for
    constants and predicates, and map them to their names.
    
    Args:
        model: Trained KGEModel instance 
        save_dir: Directory to save embedding dictionaries
        
    Returns:
        constant_embeddings_dict: Dict mapping constant names to embeddings for each domain
        predicate_embeddings_dict: Dict mapping predicate names to embeddings
    """

    constant_to_global_index = serializer.constant_to_global_index  # Dict[domain][constant] = index
    # print('constant_to_global_index', [(k,v) for k,v in constant_to_global_index.items()])

    print('domain.constants', [(domain.name, len(domain.constants)) for domain in fol.domains])
    embedder = model.kge_model.constant_embedder.embedder # there's one per domain

    embeddings_c = {domain.name: embedder[domain.name](tf.range(len(domain.constants))) for domain in fol.domains}
    print('embeddings_c', [(name, embeddings.shape) for name, embeddings in embeddings_c.items()])

    # create a dictionary with the constant str as key and the embedding as value
    constant_embeddings_dict = defaultdict(dict)

    for domain in fol.domains:
        for (c_str, idx) in constant_to_global_index[domain.name].items():
            print('c_str', c_str, 'idx', idx, 'embeddings_c[domain.name]', embeddings_c[domain.name].shape) if idx < 15 else None
            constant_embeddings_dict[domain.name][c_str] = embeddings_c[domain.name][idx]
    print('constant_embeddings_dict', [(name, len(embeddings)) for name, embeddings in constant_embeddings_dict.items()])

    predicate_to_global_index = {p: i for i, p in enumerate(fol.predicates)}
    predicate_embedder = model.kge_model.predicate_embedder.embedder
    predicate_embeddings = predicate_embedder(tf.range(len(fol.predicates)))
    predicate_embeddings_dict = {p: predicate_embeddings[i] for p, i in predicate_to_global_index.items()}
    print('predicate_embeddings_dict', [(name, embeddings.shape) for name, embeddings in predicate_embeddings_dict.items()])
    
    # DO A TEST TO SEE IF THE EMBEDDINGS ARE CORRECT. print 5 elements of the first embedding of each domain
    print('CONSTANTS')
    for domain in fol.domains:
        print('\ndomain', domain.name,'first embedding')
        print('saved', constant_embeddings_dict[domain.name][list(constant_embeddings_dict[domain.name].keys())[0]][:5])
        print('from embedders', embeddings_c[domain.name][0,:5])
        print('they are equal',constant_embeddings_dict[domain.name][list(constant_embeddings_dict[domain.name].keys())[0]][:5] == embeddings_c[domain.name][0,:5])
    print('\nPREDICATES')
    print('saved', predicate_embeddings_dict[list(predicate_embeddings_dict.keys())[0]][:5])
    print('from embedders', predicate_embeddings[0,:5])
    print('they are equal',predicate_embeddings_dict[list(predicate_embeddings_dict.keys())[0]][:5] == predicate_embeddings[0,:5])
    
    # convert the embeddings to numpy arrays
    constant_embeddings_dict = {str(k): {str(kk): v.numpy() for kk, v in vv.items()} for k, vv in constant_embeddings_dict.items()}
    predicate_embeddings_dict = {str(k): v.numpy() for k, v in predicate_embeddings_dict.items()}

    # Save embeddings
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "constant_embeddings.pkl"), "wb") as f:
        pickle.dump(constant_embeddings_dict, f)
    with open(os.path.join(save_dir, "predicate_embeddings.pkl"), "wb") as f:
        pickle.dump(predicate_embeddings_dict, f)

    # with open(os.path.join(save_dir, "constant_embeddings.json"), "w") as f:
    #     json.dump(constant_embeddings_dict, f)
    # with open(os.path.join(save_dir, "predicate_embeddings.json"), "w") as f:
    #     json.dump(predicate_embeddings_dict, f)
        
    print(f"Saved embeddings to {save_dir}/")

    # load them back
    
    with open(os.path.join(save_dir, "constant_embeddings.pkl"), "rb") as f:
        constant_embeddings_dict = pickle.load(f)
    with open(os.path.join(save_dir, "predicate_embeddings.pkl"), "rb") as f:
        predicate_embeddings_dict = pickle.load(f)

    # with open(os.path.join(save_dir, "constant_embeddings.json"), "r") as f:
    #     constant_embeddings_dict = json.load(f)
    # with open(os.path.join(save_dir, "predicate_embeddings.json"), "r") as f:
    #     predicate_embeddings_dict = json.load(f)

    print('predicate_embeddings_dict', [(k, v.shape) for k, v in predicate_embeddings_dict.items()])
    print('constant_embeddings_dict', [(domain, {c: emb.shape for c, emb in dict_.items()}) for domain, dict_ in constant_embeddings_dict.items()])
    # print('constant_embeddings_dict', constant_embeddings_dict)
    # print('predicate_embeddings_dict',  predicate_embeddings_dict)
    return constant_embeddings_dict,  predicate_embeddings_dict



def evaluate_and_store_ranks(model, data_gen_test, seed, args, metric, batch_size=32):
    """Evaluates the model in batches and stores query ranks to a file incrementally."""
    print('Evaluating model and storing ranks...')
    if args.distill: 
        os.makedirs('./experiments/ranks_distill/ranking', exist_ok=True)
        output_file = f'./experiments/ranks_distill/ranking/ranks_{args.run_signature}-seed_{seed}-{round(metric, 3)}.txt'
    elif args.xkge:
        os.makedirs('./experiments/ranks_xkge/ranking', exist_ok=True)
        output_file = f'./experiments/ranks_xkge/ranking/ranks_{args.run_signature}-seed_{seed}-{round(metric, 3)}.txt'
    else:
        os.makedirs('./experiments/ranks/ranking', exist_ok=True)
        output_file = f'./experiments/ranks/ranking/ranks_{args.run_signature}-seed_{seed}-{round(metric, 3)}.txt'
    dataset_size = len(data_gen_test.dataset)*2 # positives and negatives
    
    # Open the file in write mode initially to clear any existing content
    with open(output_file, 'w') as f:
        f.write('')


    for start_idx in range(0, dataset_size, batch_size):
        queries, x, y_true = data_gen_test._get_batch_with_queries(start_idx, batch_size)
        y_true = y_true['task']
        y_pred = model(x)['task']  # Get predictions

        batch_ranks = {}  # Store batch results temporarily
        query_ids = []
        best_ranked_queries = []

        for i in range(y_pred.shape[0]):
            # if len(queries[i]) == 0 or len(queries[i]) == 1:
            #     continue  # Skip queries with no or one query

            print(f"\rProcessed query {start_idx + i + 1}/{dataset_size} ({(start_idx + i + 1) / dataset_size * 100:.2f}%)", end="")

            predictions_for_query = y_pred[i]
            sorted_indices = tf.argsort(predictions_for_query, direction='DESCENDING').numpy()
            best_output = sorted_indices[0]
            best_ranked_query = queries[i][best_output]
            query_id = queries[i][0]  # Get query name
            
            # batch_ranks[query_id] = best_ranked_query  # Store in batch dictionary
            query_ids.append(query_id)
            best_ranked_queries.append(best_ranked_query)

        # Write results for this batch to file immediately
        with open(output_file, 'a') as f:
            for query, best_ranked in zip(query_ids, best_ranked_queries):
                f.write(f"{query}:{best_ranked}\n")

    print(f"\nRanks written to {output_file}")
    return output_file  # Return file path for reference

