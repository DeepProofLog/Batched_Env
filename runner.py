from train import main
import argparse
import ast
from itertools import product
import copy
import os
import numpy as np
from utils import FileLogger

if __name__ == "__main__":

    RESTORE_BEST_MODEL = [True] #[True,False]
    TIMESTEP_TRAIN = [10000]
    load_model = True
    save_model = True
    
    use_logger = False
    use_WB = False
    WB_path = "./../wandb/"
    logger_path = "./experiments/runs/"

    DATASET_NAME =  ["ablation_d2",] #["ablation_d1","ablation_d2","ablation_d3","countries_s1", "countries_s2", "countries_s3"]
    LEARN_EMBEDDINGS = [True]
    KGE = ['transe']
    MODEL_NAME = ["PPO"]
    ATOM_EMBEDDING_SIZE = [200]#[50,200]
    SEED = [[0]] #[[0,1,2,3,4]]
    MAX_DEPTH = [20] #[20,100]

    # path to the data    
    data_path = "./data/"
    domain_file = None
    janus_file = "train.pl"
    # facts_file = "train.txt"
    train_file = "train_queries.txt"
    valid_file = "valid_queries.txt"
    test_file = "test_queries.txt"

    models_path = "./../models/"
    variable_no = 500
    device = "cpu"

    # Training parameters
    n_epochs = 10
    n_steps = 2048 # number of steps to collect in each rollout
    batch_size = 64
    lr = 3e-4


    
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
    for dataset_name, learn_embeddings, kge, model_name, atom_embedding_size, seed, max_depth,timestep_train,restore_best_model in product(DATASET_NAME, 
            LEARN_EMBEDDINGS, KGE, MODEL_NAME, ATOM_EMBEDDING_SIZE, SEED, MAX_DEPTH,TIMESTEP_TRAIN,RESTORE_BEST_MODEL):

        constant_emb_file = data_path+dataset_name+"/constant_embeddings.pkl"
        predicate_emb_file = data_path+dataset_name+"/predicate_embeddings.pkl"
        constant_embedding_size = predicate_embedding_size = atom_embedding_size
        
        args.dataset_name = dataset_name
        args.data_path = data_path
        args.domain_file = domain_file
        args.janus_file = janus_file
        args.train_file = train_file
        args.valid_file = valid_file
        args.test_file = test_file
        
        args.learn_embeddings = learn_embeddings
        args.kge = kge
        args.model_name = model_name
        args.atom_embedding_size = atom_embedding_size
        args.constant_embedding_size = constant_embedding_size
        args.predicate_embedding_size = predicate_embedding_size
        args.constant_emb_file = constant_emb_file
        args.predicate_emb_file = predicate_emb_file
        args.variable_no = variable_no
        args.device = device
        args.seed = seed
        
        args.restore_best_model = restore_best_model
        args.load_model = load_model
        args.save_model = save_model
        args.models_path = models_path+dataset_name
        args.timesteps_train = timestep_train
        args.n_epochs = n_epochs
        args.n_steps = n_steps
        args.batch_size = batch_size
        args.lr = lr
        args.max_depth = max_depth

        run_vars = (args.dataset_name, args.kge, args.model_name, args.atom_embedding_size,args.max_depth,
                    args.learn_embeddings,args.timesteps_train,args.restore_best_model)
        args.run_signature = '-'.join(f'{v}' for v in run_vars) 

        all_args.append(copy.deepcopy(args)) # append a hard copy of the args to the list of all_args


    def main_wrapper(args):

        if use_logger:
            logger = FileLogger(base_folder=logger_path)
            if logger.exists_experiment(args.__dict__):
                return
            
        for seed in args.seed:
            args.seed_run_i = seed
            print("Seed", seed, " in ", args.seed)
            print("\nRun vars:", args.run_signature, '\n',args,'\n')
            if use_logger:
                date = logger.date
                log_filename_tmp = os.path.join(logger_path,'_tmp_log-{}-{}-seed_{}.csv'.format(args.run_signature,date,seed))
                if logger.exists_run(args.run_signature,seed):   
                    continue
                # else:
                #     print("Seed number ", seed, " not done. Exit")
                #     continue
            else:   
                log_filename_tmp = None

            valid_metrics, test_metrics = main(args,log_filename_tmp,use_logger,use_WB,WB_path)

            if use_logger:
                # Include the results in the logger
                logged_data = copy.deepcopy(args)
                dicts_to_log = {'valid':valid_metrics,'test':test_metrics}
                # write the info about the results in the tmp file 
                logger.log(log_filename_tmp,logged_data.__dict__,dicts_to_log)
                # Rename to not be temporal anymore
                mean_rwd = np.round(np.mean(test_metrics['reward']),3)
                log_filename_run_name = os.path.join(logger_path,'indiv_runs', '_ind_log-{}-{}-{}-seed_{}.csv'.format(
                                                            args.run_signature,date,mean_rwd,seed))
                logger.finalize_log_file(log_filename_tmp,log_filename_run_name)

        # If we have done all the seeds in args.seed, we can get the average results
        logger.log_avg_results(args.__dict__, args.run_signature,args.seed) if use_logger else None

    for args in all_args:
        print('Experiment number ', all_args.index(args), ' out of ', len(all_args), ' experiments.')
        main_wrapper(args)