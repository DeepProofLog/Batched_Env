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

    RULE_DEPEND_VAR = [False] # [True, False] # the way to define variable embedding, True: depend on rules, False: indexed based on appearance order
    DYNAMIC_CONSULT = [True] # [True, False]
    FALSE_RULES = [False] 
    MEMORY_PRUNING = [True] # True: filter prolog outputs to cut loop; False: stop at proven subgoal to cut loop
    END_PROOF_ACTION = [False]
    SKIP_UNARY_ACTIONS = [True]
    TRUNCATE_ATOMS = [True] # if more atoms in a state than pad_atoms, truncate the state to false
    TRUNCATE_STATES = [True] # if more states in next states than pad_states, truncate the state to false
    ENT_COEF = [0.1]
    CLIP_RANGE = [0.2]
    ENGINE = ['python']
    # reward_type = 1

    # Dataset settings 
    # for countries_s3, we use non provable corruptions and provable queries, run_signature='countries_s3-transe-sum-50-5-20-dynamic-False-True-1-True-False-False-True-False-20-True'
    DATASET_NAME =  ["kinship_family"] #["ablation_d1","ablation_d2","ablation_d3","countries_s2", "countries_s3", 'kinship_family']
    TRAIN_DEPTH = [None] # [None, 1, 2, 3]
    VALID_DEPTH = [None] # [None, 1, 2, 3]
    TEST_DEPTH = [None] # [None, 1, 2, 3]
    SEED = [[0]] # [[0,1,2,3,4]]
    LEARN_EMBEDDINGS = [True]
    ATOM_EMBEDDER = ['transe'] #['complex','rotate','transe','attention','rnn']
    STATE_EMBEDDER = ['mean'] #'mean'
    PADDING_ATOMS = [3]
    PADDING_STATES = [10]
    ATOM_EMBEDDING_SIZE = [64]
    '''Attention: if we use static corruptions, include non provable corruptions
    In pararell eval, we evaluate based on a fixed number of corruptions'''
    CORRUPTION_MODE =  ['dynamic'] # ["dynamic","static"] # TAKE INTO ACCOUNT THE DYNAMIC INCLUDES NON PROVABLE NEGATIVES
    NON_PROVABLE_QUERIES = [True]
    NON_PROVABLE_CORRUPTIONS = [True]

    RESTORE_BEST_VAL_MODEL = [True] #[True,False]
    load_model = True #['best_eval', 'last_epoch', False]
    save_model = True #['best_eval', 'last_epoch', False]

    # Loggin settings 
    use_logger = True
    use_WB = False
    WB_path = "./../wandb/"
    logger_path = "./runs/"

    # Paths    
    data_path = "./data/"
    models_path = "models/"
    janus_file = "train.pl"
    rules_file = "rules.txt"
    facts_file = "train.txt"
    train_file = "train.txt"
    valid_file = "valid.txt"    
    test_file = "test.txt"

    # Training parameters
    TIMESTEPS_TRAIN = [3000000]
    MODEL_NAME = ["PPO"]
    MAX_DEPTH = [20] # [20,100]
    TRAIN_NEG_POS_RATIO = [1] # corruptions in train
    valid_negatives = None # corruptions in validation set (test)
    test_negatives = 100 # corruptions in test set (test)
    n_eval_queries = 200 
    n_test_queries = None
    # Rollout->train. in rollout, each env does n_steps steps, and n_envs envs are run in parallel.
    # The total number of steps in each rollout is n_steps*n_envs.
    n_envs = 128
    n_steps = 128 # 2048
    n_eval_envs = 128
    n_callback_envs = 1
    eval_freq = n_steps*n_envs
    n_epochs = 10 # number of epochs to train the model with the collected rollout
    batch_size = 128 # Ensure batch size is a factor of n_steps (for the buffer).
    lr = 3e-4
    # if 'countries_s3' in DATASET_NAME:
    #     # use simple linear layers
    #     n_envs = 128 
    #     n_steps = 128 #2048
    #     n_eval_envs = 1
    #     eval_freq = n_steps*n_envs
    #     n_epochs = 10 # number of epochs to train the model with the collected rollout
    #     batch_size = 128 # Ensure batch size is a factor of n_steps (for the buffer).
    #     lr = 3e-4


    # number of variables in the index manager to create embeddings for. if RULE_DEPEND_VAR is True, 
    # this is ignored and the number of variables is determined by the number of variables in the rules
    variable_no = 500
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # If take inputs from the command line, overwrite the default values
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Experiment Runner')
    parser.add_argument("--d", nargs='+', help="Datasets")
    parser.add_argument("--s", help="Seeds")
    parser.add_argument("--epochs", default = None, help="epochs")

    parser.add_argument("--corr_mode", default = None, help="corruption mode")
    parser.add_argument("--non_provable_queries", default = None, help="non_provable_queries")
    parser.add_argument("--non_provable_corruptions", default = None, help="non_provable_corruptions")
    parser.add_argument("--train_neg_pos_ratio", default = None, help="train_neg_pos_ratio")

    parser.add_argument("--memory_pruning", default = None, help="memory_pruning")
    parser.add_argument("--rule_depend_var", default = None, help="rule_depend_var")
    parser.add_argument("--dynamic_consult", default = None, help="dynamic_consult")
    parser.add_argument("--false_rules", default = None, help="false_rules")
    parser.add_argument("--end_proof_action", default = None, help="end_proof_action")
    parser.add_argument("--skip_unary_actions", default = None, help="skip_unary_actions")
    parser.add_argument("--truncate_atoms", default = None, help="truncate_atoms")
    parser.add_argument("--truncate_states", default = None, help="truncate_states")

    parser.add_argument("--padding_atoms", default = None, help="padding_atoms")
    parser.add_argument("--padding_states", default = None, help="padding_states")
    parser.add_argument("--atom_embedding_size", default = None, help="atom_embedding_size")
    parser.add_argument("--test_file", default = None, help="test_file")


    args = parser.parse_args()

    # Update configuration with command line arguments
    if args.d: DATASET_NAME = args.d
    if args.s: SEED = [ast.literal_eval(args.s)]
    if args.epochs: epochs = [int(args.epochs)]

    if args.corr_mode: CORRUPTION_MODE = [args.corr_mode]
    if args.non_provable_queries: NON_PROVABLE_QUERIES = [ast.literal_eval(args.non_provable_queries)]
    if args.non_provable_corruptions: NON_PROVABLE_CORRUPTIONS = [ast.literal_eval(args.non_provable_corruptions)]
    if args.train_neg_pos_ratio: TRAIN_NEG_POS_RATIO = [int(args.train_neg_pos_ratio)]

    if args.memory_pruning: MEMORY_PRUNING = [ast.literal_eval(args.memory_pruning)]
    if args.rule_depend_var: RULE_DEPEND_VAR = [ast.literal_eval(args.rule_depend_var)]
    if args.dynamic_consult: DYNAMIC_CONSULT = [ast.literal_eval(args.dynamic_consult)]
    if args.false_rules: FALSE_RULES = [ast.literal_eval(args.false_rules)]
    if args.end_proof_action: END_PROOF_ACTION = [ast.literal_eval(args.end_proof_action)]
    if args.skip_unary_actions: SKIP_UNARY_ACTIONS = [ast.literal_eval(args.skip_unary_actions)]
    if args.truncate_atoms: TRUNCATE_ATOMS = [ast.literal_eval(args.truncate_atoms)]
    if args.truncate_states: TRUNCATE_STATES = [ast.literal_eval(args.truncate_states)]

    if args.padding_atoms: PADDING_ATOMS = [int(args.padding_atoms)]
    if args.padding_states: PADDING_STATES = [int(args.padding_states)]
    if args.atom_embedding_size: ATOM_EMBEDDING_SIZE = [int(args.atom_embedding_size)]
    if args.test_file: test_file = args.test_file


    print('Running experiments for the following parameters:','DATASET_NAME:',DATASET_NAME,'MODEL_NAME:',MODEL_NAME,'SEED:',SEED)
    
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
        'rule_depend_var': RULE_DEPEND_VAR,
        'dynamic_consult': DYNAMIC_CONSULT,
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
        'truncate_atoms': TRUNCATE_ATOMS,
        'truncate_states': TRUNCATE_STATES,
        'padding_atoms': PADDING_ATOMS,
        'padding_states': PADDING_STATES,
        'non_provable_queries': NON_PROVABLE_QUERIES,
        'non_provable_corruptions': NON_PROVABLE_CORRUPTIONS
    }

    # Generate all combinations using product
    param_combinations = list(product(*param_dict.values()))
    param_names = param_dict.keys()
    total_experiments = len(param_combinations)

    # Iterate over combinations with named parameters
    all_args = []
    for exp_idx, params in enumerate(param_combinations):
        args = argparse.Namespace(**dict(zip(param_names, params)))

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
            args.janus_file = janus_file

        if not args.non_provable_corruptions and args.corruption_mode == "dynamic":
            print("\n\nSKIPPING EXPERIMENT: non_provable_corruptions with dynamic corruptions is not supported\n\n")
            continue
        if args.non_provable_queries and args.corruption_mode == "static":
            print("\n\nSKIPPING EXPERIMENT: non_provable_queries with static corruptions is not yet supported\n\n")
            continue

        if args.dataset_name == "mnist_addition":
            args.corruption_mode = None
    
        if args.corruption_mode == "static":
            train_file = "train_label_corruptions.json"
            valid_file = "valid_label_corruptions.json"
            test_file  = "test_label_corruptions.json"
        elif args.corruption_mode == "dynamic" and not args.non_provable_queries:
            train_file = "train_label.txt"
            valid_file = "valid_label.txt"
            test_file = "test_label.txt"

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
        args.state_embedding_size = args.atom_embedding_size if args.state_embedder != "concat" else args.atom_embedding_size*args.padding_atoms
        args.constant_embedding_size = constant_embedding_size
        args.predicate_embedding_size = predicate_embedding_size
        args.variable_no = variable_no
        args.device = device
        
        if args.restore_best_val_model and load_model=='last_epoch':
            print("\n\nWARNING: restore_best_val_model is True and load_model is 'last_epoch', instead of best_eval. You may not get the same eval results\n\n")
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
        args.n_callback_envs = n_callback_envs
        args.n_steps = n_steps
        args.n_epochs = n_epochs
        args.batch_size = batch_size
        args.lr = lr

        run_vars = (args.dataset_name,args.atom_embedder,args.state_embedder,args.atom_embedding_size,args.padding_atoms,args.padding_states,
                    args.corruption_mode, args.non_provable_queries, args.non_provable_corruptions,args.train_neg_pos_ratio, 
                    args.dynamic_consult, args.false_rules, args.end_proof_action, args.skip_unary_actions, args.truncate_atoms,
                    args.truncate_states, args.memory_pruning, args.rule_depend_var, args.max_depth,args.restore_best_val_model,args.ent_coef,args.clip_range,
                    args.engine, args.train_neg_pos_ratio)
        
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
            print("\nRun vars:", args.run_signature, '\n',args,'\n')
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
