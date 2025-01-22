# from environments.env_gym_wrapper import TorchRLToGymWrapper
# from environments.env_logic import  BatchLogicProofEnv
from environments.env_logic_gym import LogicEnv_gym, IndexManager

import numpy as np
import datetime
import os
import random
import torch
from typing import Tuple

from utils import get_device,simple_rollout, print_state_transition
from my_callbacks import SB3ModelCheckpoint, EvalCallback, EpochTimingCallback
from dataset import DataHandler
from model_SB3 import CustomActorCriticPolicy, CustomCombinedExtractor
from kge import read_embeddings, create_embed_tables, KGEModel, EmbeddingFunction

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    StopTrainingOnMaxEpisodes,
    StopTrainingOnRewardThreshold,
    CallbackList,
    StopTrainingOnNoModelImprovement,
)
from wandb.integration.sb3 import WandbCallback
import wandb

from dataset import BasicNegativeSamplerDomain
from pykeen.triples import TriplesFactory
from typing import Dict



def main(args,log_filename,use_logger,use_WB,WB_path,date):

    torch.manual_seed(args.seed_run_i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed_run_i)
    random.seed(args.seed_run_i)
    np.random.seed(args.seed_run_i)

    device = get_device(args.device)

    data_handler = DataHandler(
        dataset_name=args.dataset_name,
        base_path=args.data_path,
        janus_file=args.janus_file,
        train_file= args.train_file,
        valid_file=args.valid_file,
        test_file= args.test_file,
        dynamic_neg=args.dynamic_neg,
        standard_corruptions=args.standard_corruptions,
        train_neg_pos_ratio=args.train_neg_pos_ratio,
        name=args.dataset_name)

    index_manager = IndexManager(data_handler.constants, 
                                 data_handler.predicates,
                                 data_handler.constant_no, 
                                 data_handler.predicate_no,
                                 args.variable_no,
                                 max_arity=data_handler.max_arity, 
                                 device=device)
    
    if args.standard_corruptions:
        np_facts = np.array([[f.args[0], f.predicate, f.args[1]] for f in data_handler.facts],dtype=str)
        triples_factory = TriplesFactory.from_labeled_triples(triples=np_facts,
                                                            entity_to_id=index_manager.constant_str2idx,
                                                            relation_to_id=index_manager.predicate_str2idx,
                                                            compact_id=False,
                                                            create_inverse_triples=False)
        from dataset import get_sampler
        data_handler.sampler = get_sampler(data_handler=data_handler, index_manager=index_manager, triples_factory=triples_factory)
        data_handler.triples_factory = triples_factory

    if args.learn_embeddings:
        embedder = KGEModel(data_handler.constant_no, 
                        data_handler.predicate_no,
                        args.variable_no,
                        args.kge,
                        constant_embedding_size=args.constant_embedding_size,
                        predicate_embedding_size=args.predicate_embedding_size,
                        atom_embedding_size=args.atom_embedding_size,
                        device=device)
    else:
        constant_str2idx, predicate_str2idx = index_manager.constant_str2idx, index_manager.predicate_str2idx
        constant_idx2emb, predicate_idx2emb = read_embeddings(args.constant_emb_file, args.predicate_emb_file, constant_str2idx, predicate_str2idx)
        constant_idx2emb, predicate_idx2emb = create_embed_tables(constant_idx2emb, predicate_idx2emb, args.variable_no)
        embedder = EmbeddingFunction(constant_idx2emb, predicate_idx2emb, device=device)


    
    # INIT ENV
    env = LogicEnv_gym(max_depth=args.max_depth,
                        device=device, 
                        index_manager=index_manager,
                        data_handler=data_handler,
                        seed=args.seed_run_i,
                        dynamic_neg=args.dynamic_neg,
                        train_neg_pos_ratio=args.train_neg_pos_ratio,
                        limit_space=args.limit_space)
    
    eval_env = LogicEnv_gym(max_depth=args.max_depth,
                            device=device, 
                            index_manager=index_manager,
                            data_handler=data_handler,
                            seed=args.seed_run_i,
                            dynamic_neg=args.dynamic_neg,
                            train_neg_pos_ratio=args.train_neg_pos_ratio,
                            limit_space=args.limit_space,
                            eval=True) 

    # INIT MODEL
    if args.model_name == "PPO":
        from model_SB3 import PPO_custom as PPO
        model = PPO(CustomActorCriticPolicy, 
                    env,
                    learning_rate=args.lr,
                    n_steps=args.n_steps,
                    batch_size=args.batch_size,
                    n_epochs=args.n_epochs,
                    verbose=1, 
                    device=device,
                    policy_kwargs={'features_extractor_class':CustomCombinedExtractor,
                                    'features_extractor_kwargs':{'features_dim':embedder.embed_dim,
                                                                    'embedder': embedder}})
    else:
        raise ValueError("Model not supported")
    

    # TRAIN
    model_path = os.path.join(args.models_path,args.run_signature, args.run_signature + f'_seed-{args.seed_run_i}')
    model_name = args.run_signature+date+'-seed_{}'.format(args.seed_run_i)
    if args.load_model:
        try:
            models = sorted(
                [m for m in os.listdir(model_path) if 'zip' in m and str(args.load_model) in m and f'seed_{args.seed_run_i}' in m])
            if models:
                print(f"Loading model from {os.path.join(model_path, models[-1])}")
                model = PPO.load(os.path.join(model_path, models[-1]), env=eval_env, device=device)
            else:
                raise FileNotFoundError(f"No suitable model found in {model_path}")
        except (FileNotFoundError, ValueError) as e:
            print(e)
            args.load_model = False
        
    if not args.load_model and args.timesteps_train > 0:

        timing_callback = EpochTimingCallback(verbose=1)

        callbacks = [timing_callback]

        # no_improvement_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, verbose=1)
        reward_threshold_callback = StopTrainingOnRewardThreshold(reward_threshold=1, verbose=1)

        eval_callback = EvalCallback(eval_env=eval_env, 
                                    model_path=model_path if args.save_model else None,
                                    save_model=args.save_model,
                                    log_path=log_filename if use_logger else None,
                                    eval_freq=10000,
                                    n_eval_episodes=len(data_handler.valid_queries),
                                    deterministic=True,
                                    render=False,
                                    name=model_name,
                                    callback_on_new_best=reward_threshold_callback if args.restore_best_model else None,
                                    # callback_after_eval=no_improvement_callback,
                                    )

        callbacks.append(eval_callback)
    
        checkpoint_callback = SB3ModelCheckpoint(model,monitor='train/loss', frequency=5000, 
                                                    total_steps=args.timesteps_train, 
                                                    model_path=model_path if args.save_model else None,
                                                    log_path = log_filename if use_logger else None,
                                                    name=model_name,)
        callbacks.append(checkpoint_callback)

        # Initialize a W&B run
        if use_WB:
            run = wandb.init(project = "RL-NeSy", 
                            group= args.run_signature,
                            name=model_name,
                            dir=WB_path,  
                            sync_tensorboard=True,  # Sync SB3's TensorBoard logs to W&B
                            config = dict(
                                seed = args.seed_run_i,
                                shuffle_buffer = 1024,
                                batch_size = args.batch_size,
                                learning_rate = args.lr,
                                epochs = args.n_epochs)) 
            
            callbacks.append(WandbCallback(
                                            # gradient_save_freq=100,
                                            # model_save_path=WB_path,
                                            # verbose=2,
                                            ))


        callbacks = CallbackList(callbacks)
        model.learn(total_timesteps=args.timesteps_train, callback=callbacks)
        # if args.restore_best_model:
        #     eval_callback.restore_best_ckpt()

        if use_WB:
            run.finish()   

    # TEST
    from model_eval import eval_test_corruptions
    def print_eval_info(metrics: Dict[str, float]):
        print(*[f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()], sep='\n')

    print('\nTesting train set...')
    if not args.dynamic_neg:
        metrics_train = eval_test_corruptions(eval_env.train_queries,eval_env.train_labels,data_handler.train_corruptions,eval_env,model,verbose=1,consult_janus=True)
    else:
        metrics_train = eval_test_corruptions(eval_env.pos_train_queries,[1]*len(eval_env.pos_train_queries), data_handler.train_corruptions, eval_env,model,verbose=1,consult_janus=True)
    print_eval_info(metrics_train)
    print('\nTesting val set...')
    metrics_valid = eval_test_corruptions(eval_env.valid_queries,eval_env.valid_labels,data_handler.valid_corruptions,eval_env,model,verbose=1)
    print_eval_info(metrics_valid)
    print('\nTesting test set...')
    metrics_test = eval_test_corruptions(eval_env.test_queries,eval_env.test_labels,data_handler.test_corruptions,eval_env,model,verbose=1)
    print_eval_info(metrics_test)

    return metrics_train, metrics_valid, metrics_test

