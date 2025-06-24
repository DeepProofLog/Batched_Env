import torch
from train import main
import argparse
import ast
from itertools import product
import copy
import os
import numpy as np
from utils import FileLogger
import datetime
import sys

if __name__ == "__main__":

    class Tee:
        def __init__(self, file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            self.file = open(file_path, "w")
            self.stdout = sys.stdout

        def write(self, message):
            self.file.write(message)
            self.stdout.write(message)

        def flush(self):
            self.file.flush()
            self.stdout.flush()

    FALSE_RULES = [False] 
    MEMORY_PRUNING = [True] # filter prolog outputs to cut loop; False: stop at proven subgoal to cut loop
    END_PROOF_ACTION = [False]
    SKIP_UNARY_ACTIONS = [True]
    ENT_COEF = [0.5]
    CLIP_RANGE = [0.2]
    ENGINE = ['python']
    # reward_type = 1

    # Dataset settings 
    DATASET_NAME =  ["wn18rr"] #["countries_s2", "countries_s3", 'family', 'wn18rr']
    TRAIN_DEPTH = [None] # [{-1,3,2}]
    VALID_DEPTH = [None]
    TEST_DEPTH = [None]
    SEED = [[0]]
    LEARN_EMBEDDINGS = [True]
    ATOM_EMBEDDER = ['transe'] #['complex','rotate','transe','attention','rnn']
    STATE_EMBEDDER = ['mean']
    PADDING_ATOMS = [5]
    PADDING_STATES = [-1] # -1 sets the max padding size to a preset value (check below)
    ATOM_EMBEDDING_SIZE = [64]
    CORRUPTION_MODE =  ['dynamic']

    RESTORE_BEST_VAL_MODEL = [True]
    load_model = True
    save_model = True

    # Loggin settings 
    use_logger = True
    use_WB = False
    WB_path = "./../wandb/"
    logger_path = "./runs/"

    # Paths    
    data_path = "./data/"
    models_path = "models/"
    rules_file = "rules.txt"
    facts_file = "train.txt"
    train_file = "train.txt"
    valid_file = "valid.txt"
    test_file = "test.txt"

    # Training parameters
    TIMESTEPS_TRAIN = [1]
    MODEL_NAME = ["PPO"]
    MAX_DEPTH = [20]
    TRAIN_NEG_POS_RATIO = [1] # corruptions in train
    valid_negatives = None # corruptions in validation set (test)
    test_negatives = 500 # corruptions in test set (test)
    n_eval_queries = 500 
    n_test_queries = 5
    # Rollout-> train. in rollout, each env does n_steps steps, and n_envs envs are run in parallel.
    # The total number of steps in each rollout is n_steps*n_envs.
    n_envs = 16
    n_steps = 128
    n_eval_envs = 100
    eval_freq = n_steps*n_envs
    n_epochs = 10 # number of epochs to train the model with the collected rollout
    batch_size = 128 # Ensure batch size is a factor of n_steps (for the buffer).
    lr = 3e-4

    max_total_vars = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # If take inputs from the command line, overwrite the default values
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Experiment Runner')
    parser.add_argument("--d", nargs='+', help="Datasets")
    parser.add_argument("--s", help="Seeds")
    parser.add_argument("--epochs", default = None, help="epochs")

    parser.add_argument("--eval", default = None, action='store_const', const=True)
    parser.add_argument("--test_depth", default = None, help="test_depth")
    parser.add_argument("--test_negatives", default = None, help="test_negatives")

    args = parser.parse_args()

    # Update configuration with command line arguments
    if args.d: DATASET_NAME = args.d
    if args.s: SEED = [ast.literal_eval(args.s)]
    if args.epochs: epochs = [int(args.epochs)]

    if args.eval: load_model = True; TIMESTEPS_TRAIN = [0]
    if args.test_depth: TEST_DEPTH = [str(args.test_depth)]
    if args.test_negatives: test_negatives = int(args.test_negatives)


    print('Running experiments for the following parameters:','DATASET_NAME:'\
          ,DATASET_NAME,'MODEL_NAME:',MODEL_NAME,'SEED:',SEED)
    
    # Create a dictionary of all parameters
    param_dict = {
        'dataset_name': DATASET_NAME,
        'model_name': MODEL_NAME,
        'learn_embeddings': LEARN_EMBEDDINGS,
        'atom_embedder': ATOM_EMBEDDER,
        'state_embedder': STATE_EMBEDDER,
        'atom_embedding_size': ATOM_EMBEDDING_SIZE,
        'seed': SEED,
        'max_depth': MAX_DEPTH,
        'timesteps_train': TIMESTEPS_TRAIN,
        'restore_best_val_model': RESTORE_BEST_VAL_MODEL,
        'memory_pruning': MEMORY_PRUNING,
        'corruption_mode': CORRUPTION_MODE,
        'train_neg_pos_ratio': TRAIN_NEG_POS_RATIO,
        'false_rules': FALSE_RULES,
        'end_proof_action': END_PROOF_ACTION,
        'skip_unary_actions': SKIP_UNARY_ACTIONS,
        'ent_coef': ENT_COEF,
        'clip_range': CLIP_RANGE,
        'engine': ENGINE,
        'train_depth': TRAIN_DEPTH,
        'valid_depth': VALID_DEPTH,
        'test_depth': TEST_DEPTH,
        'padding_atoms': PADDING_ATOMS,
        'padding_states': PADDING_STATES,
    }

    # Generate all combinations using product
    param_combinations = list(product(*param_dict.values()))
    param_names = param_dict.keys()
    total_experiments = len(param_combinations)

    # Iterate over combinations with named parameters
    all_args = []
    for exp_idx, params in enumerate(param_combinations):
        args = argparse.Namespace(**dict(zip(param_names, params)))

        if not save_model and args.restore_best_val_model:
            print("\n\nERROR: restore_best_val_model is True and save_model is False.\
                   To restore the best val model you need to save it\n\n")
            continue
        
        if args.padding_states == -1:
            if args.dataset_name == "countries_s3": args.padding_states = 20
            elif args.dataset_name == "family": args.padding_states = 130
            elif args.dataset_name == "wn18rr": args.padding_states = 262 #1770

        constant_embedding_size = predicate_embedding_size = args.atom_embedding_size
        if args.atom_embedder == "complex":
            constant_embedding_size = 2*args.atom_embedding_size
            predicate_embedding_size = 2*args.atom_embedding_size
        if args.atom_embedder == "rotate":
            constant_embedding_size = 2*args.atom_embedding_size

        
        args.corruption_scheme = ['head','tail']
        if 'countries' in args.dataset_name or 'ablation in dataset_name' in args.dataset_name:
            args.corruption_scheme = ['tail']
        
        if args.false_rules:
            args.janus_file = "train_false_rules.pl"
        elif args.engine == 'python':
            args.janus_file = None
        else:
            args.janus_file = f"{args.dataset_name}.pl"
            print("Using prolog file:", args.janus_file)


        if args.dataset_name == "mnist_addition":
            args.corruption_mode = None
    
        if args.corruption_mode == "static":
            train_file = "train_label_corruptions.json"
            valid_file = "valid_label_corruptions.json"
            test_file  = "test_label_corruptions.json"

        if args.train_depth is not None:
            train_file = train_file.replace('.txt', f'_depths.txt')
        if args.valid_depth is not None:
            valid_file = valid_file.replace('.txt', f'_depths.txt')
        if args.test_depth is not None:
            test_file = test_file.replace('.txt', f'_depths.txt')

        args.data_path = data_path
        args.train_file = train_file
        args.valid_file = valid_file
        args.test_file = test_file
        args.rules_file = rules_file
        args.facts_file = facts_file
        

        args.atom_embedding_size = args.atom_embedding_size #if args.atom_embedder != "concat" else it is (pred+c1+c2+...+cn)*atom_embedding_size = (1+max_arity)*atom_embedding_size
        args.state_embedding_size = args.atom_embedding_size if args.state_embedder != "concat" \
            else args.atom_embedding_size*args.padding_atoms
        args.constant_embedding_size = constant_embedding_size
        args.predicate_embedding_size = predicate_embedding_size
        args.max_total_vars = max_total_vars
        args.device = device
        
        if args.restore_best_val_model and load_model=='last_epoch':
            print("\n\nWARNING: restore_best_val_model is True and load_model is 'last_epoch', \
                  instead of best_eval. You may not get the same eval results\n\n")
        args.load_model = load_model
        args.save_model = save_model
        args.models_path = models_path+args.dataset_name
        args.n_eval_queries = n_eval_queries
        args.n_test_queries = n_test_queries
        args.valid_negatives = valid_negatives
        args.test_negatives = test_negatives
        args.eval_freq = eval_freq
        args.n_envs = n_envs
        args.n_eval_envs = n_eval_envs
        args.n_steps = n_steps
        args.n_epochs = n_epochs
        args.batch_size = batch_size
        args.lr = lr

        run_vars = (args.dataset_name,args.atom_embedder,args.state_embedder,args.atom_embedding_size,
                    args.padding_atoms,args.padding_states,args.false_rules,args.end_proof_action, 
                    args.skip_unary_actions,args.memory_pruning,args.max_depth,
                    args.ent_coef,args.clip_range,args.engine,args.train_neg_pos_ratio
                    )
        
        args.run_signature = '-'.join(f'{v}' for v in run_vars)
        # # Redirect stdout to the Tee class
        # if use_logger:
        #     sys.stdout = Tee(f"output/output-{args.run_signature}.log")
        all_args.append(copy.deepcopy(args)) # append a hard copy of the args to the list of all_args

        

    def main_wrapper(args):

        if use_logger:
            logger = FileLogger(base_folder=logger_path)
            # if logger.exists_experiment(args.__dict__):
            #     return

        for seed in args.seed:
            date = str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
            args.seed_run_i = seed
            print("Seed", seed, " in ", args.seed)
            # order the args to have a consistent order
            dict_ordered = {k: args.__dict__[k] for k in sorted(args.__dict__.keys())}
            print("\nRun vars:", args.run_signature, '\n',dict_ordered,'\n')
            if use_logger:
                log_filename_tmp = os.path.join(logger_path,'_tmp_log-{}-{}-seed_{}.csv'.format(args.run_signature,date,seed))
                # if logger.exists_run(args.run_signature,seed):
                #     continue
                # else:
                #     print("Seed number ", seed, " not done. Exit")
                #     continue
            else:   
                log_filename_tmp = None

            train_metrics, valid_metrics, test_metrics = main(args,log_filename_tmp,use_logger,use_WB,WB_path,date)

            if use_logger:
                # Include the results in the logger
                logged_data = copy.deepcopy(args)
                dicts_to_log = {'train':train_metrics,'valid':valid_metrics,'test':test_metrics}
                # write the info about the results in the tmp file 
                logger.log(log_filename_tmp,logged_data.__dict__,dicts_to_log)
                # Rename to not be temporal anymore
                rewards_pos_mean = np.round(np.mean(test_metrics['rewards_pos_mean']),3)
                mrr = np.round(np.mean(test_metrics['mrr_mean']),3)
                metrics = "{:.3f}_{:.3f}".format(rewards_pos_mean,mrr)
                # print the mean reward with 3 decimals 
                log_filename_run_name = os.path.join(logger_path,'indiv_runs', '_ind_log-{}-{}-{}-seed_{}.csv'.format(
                                                            args.run_signature,date,metrics,seed))
                logger.finalize_log_file(log_filename_tmp,log_filename_run_name)

        # If we have done all the seeds in args.seed, we can get the average results
        logger.log_avg_results(args.__dict__, args.run_signature,args.seed) if use_logger else None

    for args in all_args:
        print('Experiment number ', all_args.index(args), ' out of ', len(all_args), ' experiments.')
        main_wrapper(args)
