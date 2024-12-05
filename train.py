# from environments.env_gym_wrapper import TorchRLToGymWrapper
# from environments.env_logic import  BatchLogicProofEnv
from environments.env_logic_gym import LogicEnv_gym, IndexManager

import numpy as np
import datetime
import os
import random
import torch

from utils import get_device,simple_rollout, print_state_transition
from my_callbacks import SB3ModelCheckpoint, LogToFileCallback, EvalCallback
from dataset import DataHandler, Rule, DataHandler_corruptions
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

    device = get_device(args.device)

    data_handler = DataHandler_corruptions(
        dataset_name=args.dataset_name,
        base_path=args.data_path,
        # facts_file=args.facts_file,
        janus_file=args.janus_file,
        train_file= args.train_file,
        valid_file=args.valid_file,
        test_file= args.test_file
        )

    index_manager = IndexManager(data_handler.constants, 
                                 data_handler.predicates,
                                 data_handler.constant_no, 
                                 data_handler.predicate_no,
                                 args.variable_no,
                                 max_arity=data_handler.max_arity, 
                                 device=device)

    if args.learn_embedding:
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
    env = LogicEnv_gym(facts=data_handler.facts, 
                        train_queries=data_handler.train_queries,
                        valid_queries=data_handler.valid_queries,
                        test_queries=data_handler.test_queries,

                        train_labels=data_handler.train_labels,
                        valid_labels=data_handler.valid_labels, 
                        test_labels=data_handler.test_labels,

                        train_queries_positive=data_handler.train_queries_positive,
                        valid_queries_positive=data_handler.valid_queries_positive,
                        test_queries_positive=data_handler.test_queries_positive,

                        janus_file=data_handler.janus_path,

                        max_arity=data_handler.max_arity,
                        max_depth=args.max_depth,
                        device=device, 
                        index_manager=index_manager,
                        seed=args.seed_run_i)   
    
    eval_env = LogicEnv_gym(facts=data_handler.facts,
                            train_queries=data_handler.train_queries,
                            valid_queries=data_handler.valid_queries,
                            test_queries=data_handler.test_queries,

                            train_labels=data_handler.train_labels,
                            valid_labels=data_handler.valid_labels, 
                            test_labels=data_handler.test_labels,

                            train_queries_positive=data_handler.train_queries_positive,
                            valid_queries_positive=data_handler.valid_queries_positive,
                            test_queries_positive=data_handler.test_queries_positive,

                            max_arity=data_handler.max_arity,
                            max_depth=args.max_depth,
                            device=device, 
                            index_manager=index_manager,
                            seed=args.seed_run_i,
                            eval=True) 

    # INIT MODEL
    if args.model_name == "PPO":
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
        models =  os.listdir(args.models_path)
        models = [m for m in models if args.model_name in m and 'train' in m] # choose best model wrt training, not validation
        models.sort()
        if not models:
            print("No model found in", args.models_path,'!!!')
            args.load_model = False
        elif args.model_name == "PPO":
            print("Loading model from", args.models_path+"/"+models[-1])
            model = PPO.load(args.models+"/"+models[-1], device=device)
        else:
            print("Model not supported !!!!")
            args.load_model = False
        
    if not args.load_model and args.timesteps_train > 0:
        # reward_threshold_callback = StopTrainingOnRewardThreshold(reward_threshold=1, verbose=1)
        # no_improvement_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, verbose=1)
        date = '_'+str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

        eval_callback = EvalCallback(
            eval_env=eval_env, 
            model_path=os.path.join(args.models_path, args.run_signature) if args.save_model else None,
            log_path=log_filename if use_logger else None,
            eval_freq=5000,
            n_eval_episodes=5,
            deterministic=True,
            render=False,
            name=args.run_signature+date,
            # callback_on_new_best=reward_threshold_callback,
            # callback_after_eval=no_improvement_callback,
        )

        callbacks = [eval_callback]

        if use_logger:
            log_callback = LogToFileCallback(log_file=log_filename)
            callbacks.append(log_callback)    
    
        checkpoint_callback = SB3ModelCheckpoint(model,monitor='train/loss', frequency=5000, 
                                                    total_steps=args.timesteps_train, 
                                                    model_path=os.path.join(args.models_path, args.run_signature) if args.save_model else None,
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
        # checkpoint_callback.restore_best_ckpt()

        if use_WB:
            run.finish()   

    # TEST
    def test(data: list[Rule], verbose: int=0) -> list[float]:
        next_query = 0
        obs, _ = env.reset_from_query(data[next_query])
        print_state_transition(env.tensordict['state'], env.tensordict['derived_states'],env.tensordict['reward'], env.tensordict['done']) if verbose >=1 else None
        rewards_list = []
        episode_len_list = []
        trajectory_reward = 0
        episode_len = 0
        while next_query < len(data)-1:
            print('query',next_query) if verbose >=1 else None
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, truncated, info = env.step(action)
            print_state_transition(env.tensordict['state'], env.tensordict['derived_states'],env.tensordict['reward'], env.tensordict['done'], action=env.tensordict['action'],truncated=truncated) if verbose >=1 else None
            trajectory_reward += rewards
            episode_len += 1
            if dones:
                next_query += 1
                obs, _ = env.reset_from_query(data[next_query])
                rewards_list.append(trajectory_reward)
                episode_len_list.append(episode_len)
                trajectory_reward = 0
                episode_len = 0
                print_state_transition(env.tensordict['state'], env.tensordict['derived_states'],env.tensordict['reward'], env.tensordict['done']) if verbose >=1 else None
        return rewards_list, episode_len_list

    print('Testing val set...')
    positive_valid_queries = [q for q,l in zip(data_handler.valid_queries, data_handler.valid_labels) if l == 1]
    positive_test_queries = [q for q,l in zip(data_handler.test_queries, data_handler.test_labels) if l == 1]
    rewards_valid, episode_len_valid = test(positive_valid_queries)
    print('Testing test set...')
    rewards_test, episode_len_test = test(positive_test_queries)

    print('VALID: rewards avg',np.round(np.mean(rewards_valid),3), 'std', np.round(np.std(rewards_valid),3), 'episode len avg', np.round(np.mean(episode_len_valid),3), 'std', np.round(np.std(episode_len_valid),3))
    print('TEST: rewards avg',np.round(np.mean(rewards_test),3), 'std', np.round(np.std(rewards_test),3), 'episode len avg', np.round(np.mean(episode_len_test),3), 'std', np.round(np.std(episode_len_test),3))
    
    valid_metrics = {'reward': np.mean(rewards_valid), 'reward_std': np.std(rewards_valid), 'episode_len': np.mean(episode_len_valid), 'episode_len_std': np.std(episode_len_valid)}
    test_metrics = {'reward': np.mean(rewards_test), 'reward_std': np.std(rewards_test), 'episode_len': np.mean(episode_len_test), 'episode_len_std': np.std(episode_len_test)}
    return valid_metrics, test_metrics

