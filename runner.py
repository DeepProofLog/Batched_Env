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
            self.file = open(file_path, "w")
            self.stdout = sys.stdout

        def write(self, message):
            self.file.write(message)
            self.stdout.write(message)

        def flush(self):
            self.file.flush()
            self.stdout.flush()


    RESTORE_BEST_VAL_MODEL = [False] #[True,False]
    TIMESTEP_TRAIN = [2000]
    LIMIT_SPACE = [True] # [True, False]  # True: filter prolog outputs to cut loop; False: stop at proven subgoal to cut loop
    LOAD_MODEL = [False] #['best_eval', 'last_epoch', False]
    save_model = False
    train_neg_pos_ratio = 1 # in validation and test, we use all provable corruptions
   
    use_logger = True
    use_WB = False
    WB_path = "./../wandb/"
    logger_path = "./runs/"

    DATASET_NAME =  ["mnist_addition"] #["ablation_d1","ablation_d2","ablation_d3","countries_s2", "countries_s3"]
    LEARN_EMBEDDINGS = [True]
    KGE = ['transe']
    MODEL_NAME = ["PPO"]
    ATOM_EMBEDDING_SIZE = [200] #[50,200]
    SEED = [[0]] # [[0,1,2,3,4]]
    MAX_DEPTH = [20] # [20,100]
    RULE_DEPEND_VAR = [False] # [True, False] # the way to define variable embedding, True: depend on rules, False: indexed based on appearance order
    DYNAMIC_CONSULT = [False] # [True, False]

    # path to the data    
    data_path = "./data/"
    domain_file = None
    janus_file = "train.pl"
    train_txt = "train_queries.txt"
    train_json = "train_label_corruptions.json"
    valid_txt = "valid_queries.txt"
    test_txt = "test_queries.txt"

    models_path = "models/"
    # number of variables in the index manager to create embeddings for. if RULE_DEPEND_VAR is True, 
    # this is ignored and the number of variables is determined by the number of variables in the rules
    variable_no = 500 
    device = "cpu"

    # Training parameters
    n_epochs = 10
    n_steps = 2048 # number of steps to collect in each rollout
    batch_size = 64
    lr = 3e-4
    eval_freq = 1000


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
    for dataset_name, learn_embeddings, kge, model_name, atom_embedding_size, seed, max_depth,timestep_train,restore_best_val_model, limit_space, load_model, rule_depend_var, dynamic_consult in product(DATASET_NAME,
            LEARN_EMBEDDINGS, KGE, MODEL_NAME, ATOM_EMBEDDING_SIZE, SEED, MAX_DEPTH,TIMESTEP_TRAIN,RESTORE_BEST_VAL_MODEL, LIMIT_SPACE, LOAD_MODEL, RULE_DEPEND_VAR, DYNAMIC_CONSULT):

        constant_emb_file = data_path+dataset_name+"/constant_embeddings.pkl"
        predicate_emb_file = data_path+dataset_name+"/predicate_embeddings.pkl"
        constant_embedding_size = predicate_embedding_size = atom_embedding_size

        args.train_neg_pos_ratio = train_neg_pos_ratio
        args.limit_space = limit_space

        args.dynamic_neg = True
        args.standard_corruptions = False

        args.dataset_name = dataset_name
        if dataset_name == "mnist_addition":
            args.dynamic_neg = args.standard_corruptions = False
    
        if args.dynamic_neg:
            train_file = train_json
        else:
            train_file = train_txt
        valid_file = valid_txt
        test_file = test_txt

        args.data_path = data_path
        args.domain_file = domain_file
        args.janus_file = janus_file
        args.train_file = train_file
        args.valid_file = valid_file
        args.test_file = test_file
        args.dynamic_consult = dynamic_consult
        
        args.learn_embeddings = learn_embeddings
        args.kge = kge
        args.model_name = model_name
        args.atom_embedding_size = atom_embedding_size
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
        args.batch_size = batch_size
        args.lr = lr
        args.max_depth = max_depth

        run_vars = (args.dataset_name, args.kge, args.model_name, args.atom_embedding_size,args.max_depth,
                    args.learn_embeddings,args.timesteps_train,args.restore_best_val_model, args.dynamic_neg, args.train_neg_pos_ratio, args.limit_space, args.rule_depend_var, args.dynamic_consult)
        args.run_signature = '-'.join(f'{v}' for v in run_vars)
        print(args.run_signature)
        # # Redirect stdout to the Tee class
        sys.stdout = Tee(f"output-{args.run_signature}.log")

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
                mean_rwd = np.round(np.mean(test_metrics['rewards_mean']),3)
                mean_rwd = "{:.3f}".format(mean_rwd)
                # print the mean reward with 3 decimals 
                log_filename_run_name = os.path.join(logger_path,'indiv_runs', '_ind_log-{}-{}-{}-seed_{}.csv'.format(
                                                            args.run_signature,date,mean_rwd,seed))
                logger.finalize_log_file(log_filename_tmp,log_filename_run_name)

        # If we have done all the seeds in args.seed, we can get the average results
        logger.log_avg_results(args.__dict__, args.run_signature,args.seed) if use_logger else None

    for args in all_args:
        print('Experiment number ', all_args.index(args), ' out of ', len(all_args), ' experiments.')
        main_wrapper(args)