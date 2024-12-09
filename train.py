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
from my_callbacks import SB3ModelCheckpoint, EvalCallback
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



def main(args,log_filename,use_logger,use_WB,WB_path):

    torch.manual_seed(args.seed_run_i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed_run_i)
    random.seed(args.seed_run_i)
    np.random.seed(args.seed_run_i)

    date = '_'+str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    device = get_device(args.device)

    data_handler = DataHandler(
        dataset_name=args.dataset_name,
        base_path=args.data_path,
        janus_file=args.janus_file,
        train_file= args.train_file,
        valid_file=args.valid_file,
        test_file= args.test_file,
        use_only_positives=False,
        use_validation_as_train=False)

    index_manager = IndexManager(data_handler.constants, 
                                 data_handler.predicates,
                                 data_handler.constant_no, 
                                 data_handler.predicate_no,
                                 args.variable_no,
                                 max_arity=data_handler.max_arity, 
                                 device=device)

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
                        seed=args.seed_run_i)   
    
    eval_env = LogicEnv_gym(max_depth=args.max_depth,
                            device=device, 
                            index_manager=index_manager,
                            data_handler=data_handler,
                            seed=args.seed_run_i,
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
    if args.load_model:
        model_path = os.path.join(args.models_path, args.run_signature)
        if not os.path.exists(model_path):
            print("No model found in", model_path,'!!!')
            args.load_model = False
        elif args.model_name == "PPO":
            models = [m for m in os.listdir(model_path) if 'zip' in m and 'best_eval' in m]
            models.sort()
            if len(models) == 0:
                print("No model found in", model_path,'!!!')
                args.load_model = False
            else:
                print("Loading model from", os.path.join(model_path,models[-1]))
                model = PPO.load(os.path.join(model_path,models[-1]), env=None, device=device)
        else:
            print("Model not supported !!!!")
            args.load_model = False
        
    if not args.load_model and args.timesteps_train > 0:
        
        # no_improvement_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, verbose=1)
        reward_threshold_callback = StopTrainingOnRewardThreshold(reward_threshold=1, verbose=1)

        eval_callback = EvalCallback(eval_env=eval_env, 
                                    model_path=os.path.join(args.models_path, args.run_signature),
                                    save_model=args.save_model,
                                    log_path=log_filename if use_logger else None,
                                    eval_freq=10000,
                                    n_eval_episodes=len(data_handler.valid_queries),
                                    deterministic=True,
                                    render=False,
                                    name=args.run_signature+date,
                                    callback_on_new_best=reward_threshold_callback if args.restore_best_model else None,
                                    # callback_after_eval=no_improvement_callback,
                                    )

        callbacks = [eval_callback]
    
        checkpoint_callback = SB3ModelCheckpoint(model,monitor='train/loss', frequency=5000, 
                                                    total_steps=args.timesteps_train, 
                                                    model_path=os.path.join(args.models_path, args.run_signature) if args.save_model else None,
                                                    log_path = log_filename if use_logger else None,
                                                    name=args.run_signature+date)
        callbacks.append(checkpoint_callback)

        # Initialize a W&B run
        if use_WB:
            run = wandb.init(project = "RL-NeSy", 
                            group= args.run_signature,
                            name=args.run_signature+'-seed_{}'.format(args.seed_run_i),
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
        if args.restore_best_model:
            eval_callback.restore_best_ckpt()

        if use_WB:
            run.finish()   

    # TEST
    from model_SB3 import eval_test_corruptions as eval_test

    # from stable_baselines3.common.evaluation import evaluate_policy
    # print('Testing train set...')
    # eval_env.eval_dataset,eval_env.eval_len = 'train', len(data_handler.train_queries)
    # rewards_train, episode_len_train = evaluate_policy(model,eval_env,n_eval_episodes=len(data_handler.train_queries),deterministic=True,return_episode_rewards=True)
    # print('Testing val set...')
    # eval_env.eval_dataset,eval_env.eval_len = 'validation', len(data_handler.valid_queries)
    # rewards_valid, episode_len_valid = evaluate_policy(model,eval_env,n_eval_episodes=len(data_handler.valid_queries),deterministic=True,return_episode_rewards=True)
    # print('Testing test set...')
    # eval_env.eval_dataset,eval_env.eval_len = 'test', len(data_handler.test_queries)
    # rewards_test, episode_len_test = evaluate_policy(model,eval_env,n_eval_episodes=len(data_handler.test_queries),deterministic=True,return_episode_rewards=True)

    print('\nTesting train set...')
    rewards_train, episode_len_train = eval_test(eval_env.train_queries,eval_env.train_labels,eval_env,model,deterministic=True)
    print('\nTesting val set...')
    rewards_valid, episode_len_valid = eval_test(eval_env.valid_queries,eval_env.valid_labels,eval_env,model,deterministic=True)
    print('\nTesting test set...')
    rewards_test, episode_len_test = eval_test(eval_env.test_queries,eval_env.test_labels,eval_env,model,deterministic=True)

    print('\nTRAIN: rewards avg',np.round(np.mean(rewards_train),3), 'std', np.round(np.std(rewards_train),3), 'episode len avg', np.round(np.mean(episode_len_train),3), 'std', np.round(np.std(episode_len_train),3))
    print('VALID: rewards avg',np.round(np.mean(rewards_valid),3), 'std', np.round(np.std(rewards_valid),3), 'episode len avg', np.round(np.mean(episode_len_valid),3), 'std', np.round(np.std(episode_len_valid),3))
    print('TEST: rewards avg',np.round(np.mean(rewards_test),3), 'std', np.round(np.std(rewards_test),3), 'episode len avg', np.round(np.mean(episode_len_test),3), 'std', np.round(np.std(episode_len_test),3))


    valid_metrics = {'reward': np.mean(rewards_valid), 'reward_std': np.std(rewards_valid), 'episode_len': np.mean(episode_len_valid), 'episode_len_std': np.std(episode_len_valid)}
    test_metrics = {'reward': np.mean(rewards_test), 'reward_std': np.std(rewards_test), 'episode_len': np.mean(episode_len_test), 'episode_len_std': np.std(episode_len_test)}
    return valid_metrics, test_metrics

