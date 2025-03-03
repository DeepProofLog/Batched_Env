from pydantic.v1.validators import validate_json

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

    LIMIT_SPACE = [True] # True: filter prolog outputs to cut loop; False: stop at proven subgoal to cut loop
    RULE_DEPEND_VAR = [False] # [True, False] # the way to define variable embedding, True: depend on rules, False: indexed based on appearance order
    DYNAMIC_CONSULT = [False] # [True, False]
    FALSE_RULES = [False] 
    END_PROOF_ACTION = [False]
    # reward_type = 1

    # Dataset settings
    DATASET_NAME =  ["kinship_family"] #["ablation_d1","ablation_d2","ablation_d3","countries_s2", "countries_s3", 'kinship_family']
    SEED = [[0]] # [[0,1,2,3,4]]
    LEARN_EMBEDDINGS = [True]
    ATOM_EMBEDDER = ['transe'] #['complex','rotate','transe']
    STATE_EMBEDDER = ['sum'] 
    PADDING_ATOMS = [40]
    PADDING_STATES = [3000]
    ATOM_EMBEDDING_SIZE = [50]
    CORRUPTION_MODE =  ['dynamic'] # ["dynamic","static"] # TAKE INTO ACCOUNT THE DYNAMIC INCLUDES NON PROVABLE NEGATIVES
    TRAIN_NEG_POS_RATIO = [1] # in validation and test, we use all corruptions

    RESTORE_BEST_VAL_MODEL = [True] #[True,False]
    load_model = False #['best_eval', 'last_epoch', False]
    save_model = False #['best_eval', 'last_epoch', False]

    # Loggin settings 
    use_logger = False
    use_WB = False
    WB_path = "./../wandb/"
    logger_path = "./runs/"

    # Paths    
    data_path = "./data/"
    models_path = "models/"
    janus_file = "train.pl"

    # Training parameters
    TIMESTEP_TRAIN = [5001]
    MODEL_NAME = ["PPO"]
    MAX_DEPTH = [20] # [20,100]
    eval_freq = 5000
    n_eval_episodes = 100
    n_epochs = 10
    n_steps = 2048 # number of steps to collect in each rollout
    batch_size = 64
    lr = 3e-4

    # number of variables in the index manager to create embeddings for. if RULE_DEPEND_VAR is True, 
    # this is ignored and the number of variables is determined by the number of variables in the rules
    variable_no = 500
    device = "cpu"

    # If take inputs from the command line, overwrite the default values
    parser = argparse.ArgumentParser(description='Description of your script')  
    parser.add_argument("--d", default = None, help="dataset",nargs='+')
    parser.add_argument("--m", default = None, help="model",nargs='+')
    parser.add_argument("--s", default = None, help="seed")

    args = parser.parse_args()
    if args.s: SEED = [ast.literal_eval(args.s)]
    if args.m: MODEL_NAME = args.m
    if args.d: DATASET_NAME = args.d

    del args.s, args.m, args.d

    print('Running experiments for the following parameters:','DATASET_NAME:',DATASET_NAME,'MODEL_NAME:',MODEL_NAME,'SEED:',SEED)
    
    # Do the hparam search
    all_args = []
    for dataset_name,learn_embeddings,atom_embedder,state_embedder,model_name,atom_embedding_size,seed,max_depth,timestep_train,restore_best_val_model,\
    limit_space,rule_depend_var,dynamic_consult,corruption_mode,train_neg_pos_ratio,false_rules,end_proof_action,padding_atoms,padding_states in product(DATASET_NAME,
        LEARN_EMBEDDINGS,ATOM_EMBEDDER,STATE_EMBEDDER,MODEL_NAME,ATOM_EMBEDDING_SIZE,SEED,MAX_DEPTH,TIMESTEP_TRAIN,RESTORE_BEST_VAL_MODEL,LIMIT_SPACE,
        RULE_DEPEND_VAR,DYNAMIC_CONSULT,CORRUPTION_MODE,TRAIN_NEG_POS_RATIO,FALSE_RULES,END_PROOF_ACTION,PADDING_ATOMS,PADDING_STATES):

        constant_emb_file = data_path+dataset_name+"/constant_embeddings.pkl"
        predicate_emb_file = data_path+dataset_name+"/predicate_embeddings.pkl"
        constant_embedding_size = predicate_embedding_size = atom_embedding_size
        if atom_embedder == "complex":
            constant_embedding_size = 2*atom_embedding_size
            predicate_embedding_size = 2*atom_embedding_size
        if atom_embedder == "rotate":
            constant_embedding_size = 2*atom_embedding_size

        args.end_proof_action = False
        args.train_neg_pos_ratio = train_neg_pos_ratio
        args.limit_space = limit_space
        args.corruption_mode = corruption_mode
        args.non_provable_queries = False
        args.non_provable_corruptions = True
        
        args.corruption_scheme = ['head','tail']
        if 'countries' in dataset_name or 'ablation in dataset_name' in dataset_name:
            args.corruption_scheme = ['tail']
        
        args.false_rules = false_rules
        if args.false_rules:
            args.janus_file = "train_false_rules.pl"

        if not args.non_provable_corruptions and args.corruption_mode == "dynamic":
            print("\n\nSKIPPING EXPERIMENT: non_provable_corruptions with dynamic corruptions is not supported\n\n")
            continue
        if args.non_provable_queries and args.corruption_mode == "static":
            print("\n\nSKIPPING EXPERIMENT: non_provable_queries with static corruptions is not yet supported\n\n")
            continue

        args.dataset_name = dataset_name
        if dataset_name == "mnist_addition":
            args.corruption_mode = None
    
        if args.corruption_mode == "static":
            train_file = "train_label_corruptions.json"
            valid_file = "valid_label_corruptions.json"
            test_file  = "test_label_corruptions.json"
        elif args.corruption_mode == "dynamic" and not args.non_provable_queries:
            train_file = "train_label.txt"
            valid_file = "valid_label.txt"
            test_file = "test_label.txt"
        elif args.corruption_mode == "dynamic" and args.non_provable_queries:
            train_file = "train.txt"
            valid_file = "valid.txt"
            test_file = "test.txt"

        args.data_path = data_path
        args.janus_file = janus_file
        args.train_file = train_file
        args.valid_file = valid_file
        args.test_file = test_file
        args.dynamic_consult = dynamic_consult
        
        args.learn_embeddings = learn_embeddings
        args.atom_embedder = atom_embedder
        args.state_embedder = state_embedder
        args.model_name = model_name
        args.padding_atoms = padding_atoms
        args.padding_states = padding_states
        args.atom_embedding_size = atom_embedding_size #if atom_embedder != "concat" else it is (pred+c1+c2+...+cn)*atom_embedding_size = (1+max_arity)*atom_embedding_size
        # args.state_embedding_size = atom_embedding_size #if state_embedder != "concat" else it is atom_embedding_size*padding_atoms
        args.constant_embedding_size = constant_embedding_size
        args.predicate_embedding_size = predicate_embedding_size
        args.constant_emb_file = constant_emb_file
        args.predicate_emb_file = predicate_emb_file
        args.rule_depend_var = rule_depend_var
        args.variable_no = variable_no
        args.device = device
        args.seed = seed
        
        args.restore_best_val_model = restore_best_val_model 
        # raise a warning if restore_best_val_model is true and load_model=='last_epoch'
        if restore_best_val_model and load_model=='last_epoch':
            print("\n\nWARNING: restore_best_val_model is True and load_model is 'last_epoch', instead of best_eval. You may not get the same eval results\n\n")
        args.load_model = load_model
        args.save_model = save_model
        args.models_path = models_path+dataset_name
        args.timesteps_train = timestep_train
        args.n_epochs = n_epochs
        args.n_steps = n_steps
        args.eval_freq = eval_freq
        args.n_eval_episodes = n_eval_episodes
        args.batch_size = batch_size
        args.lr = lr
        args.max_depth = max_depth

        run_vars = (args.dataset_name, args.atom_embedder, args.model_name, args.atom_embedding_size,args.max_depth,
                    args.learn_embeddings,args.timesteps_train,args.restore_best_val_model, args.corruption_mode, args.train_neg_pos_ratio, args.limit_space, args.rule_depend_var, args.dynamic_consult)
        args.run_signature = '-'.join(f'{v}' for v in run_vars)
        # # Redirect stdout to the Tee class
        if use_logger:
            sys.stdout = Tee(f"output/output-{args.run_signature}.log")

        all_args.append(copy.deepcopy(args)) # append a hard copy of the args to the list of all_args


    def main_wrapper(args):

        if use_logger:
            logger = FileLogger(base_folder=logger_path)
            # if logger.exists_experiment(args.__dict__):
            #     return

        for seed in args.seed:
            date = '_'+str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
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
