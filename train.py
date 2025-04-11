import numpy as np
import os
import random
import torch
from typing import Dict

# from env import LogicEnv_gym
# from index_manager import IndexManager
# from dataset import DataHandler
# from index_manager_v2 import IndexManager
# from model_SB3 import CustomActorCriticPolicy, CustomCombinedExtractor
from env_v4 import LogicEnv_gym_batch as LogicEnv_gym, IndexManager 
from dataset_v2 import DataHandler
from model_SB3_v2 import CustomActorCriticPolicy, CustomCombinedExtractor
from neg_sampling import get_sampler

from utils import get_device, print_eval_info
from my_callbacks import SB3ModelCheckpoint, CustomEvalCallback, EpochTimingCallback
from embeddings import get_embedder
from model_eval import eval_corruptions

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    StopTrainingOnMaxEpisodes,
    StopTrainingOnRewardThreshold,
    CallbackList,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from custom_vec_env import BatchedDummyVecEnv # Assuming you saved it as custom_vec_env.py

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback
import wandb
from pykeen.triples import TriplesFactory



def main(args,log_filename,use_logger,use_WB,WB_path,date):

    print(args.run_signature)

    torch.manual_seed(args.seed_run_i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed_run_i)
    random.seed(args.seed_run_i)
    np.random.seed(args.seed_run_i)

    device = get_device(args.device)
    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}")


    data_handler = DataHandler(
        dataset_name=args.dataset_name,
        base_path=args.data_path,
        # janus_file=args.janus_file,
        train_file= args.train_file,
        valid_file=args.valid_file,
        test_file= args.test_file,
        rules_file= args.rules_file,
        facts_file= args.facts_file,
        n_eval_queries = args.n_eval_queries,
        n_test_queries = args.n_test_queries,
        corruption_mode=args.corruption_mode,
        # non_provable_corruptions=args.non_provable_corruptions,
        # non_provable_queries=args.non_provable_queries,
        )

    # data_handler= data_handler.info

    args.n_eval_queries = len(data_handler.valid_queries) if args.n_eval_queries == None else min(args.n_eval_queries, len(data_handler.valid_queries))
    args.n_test_queries = len(data_handler.test_queries) if args.n_test_queries == None else min(args.n_test_queries, len(data_handler.valid_queries))

    index_manager = IndexManager(data_handler.constants,
                                data_handler.predicates,
                                data_handler.variables if args.rule_depend_var else set(),
                                rules=data_handler.rules,
                                # rule_depend_var=args.rule_depend_var,
                                max_arity=data_handler.max_arity,
                                device='cpu',
                                padding_atoms=args.padding_atoms)
    
    if args.corruption_mode == 'dynamic':
        np_facts = np.array([[f.args[0], f.predicate, f.args[1]] for f in data_handler.facts],dtype=str)
        triples_factory = TriplesFactory.from_labeled_triples(triples=np_facts,
                                                            entity_to_id=index_manager.constant_str2idx,
                                                            relation_to_id=index_manager.predicate_str2idx,
                                                            compact_id=False,
                                                            create_inverse_triples=False)
        
        data_handler.sampler = get_sampler(data_handler=data_handler, index_manager=index_manager, triples_factory=triples_factory,
                                           corruption_scheme=args.corruption_scheme)
        data_handler.triples_factory = triples_factory

    embedder_getter = get_embedder(args, 
                         data_handler, 
                         index_manager, 
                         device, 
                        #  n_body_constants=data_handler.n_digits if args.dataset_name == 'mnist_addition' else None, # to concat the body atoms
                         )
    embedder = embedder_getter.embedder

    # Define the state_embedding_size for the model
    args.atom_embedding_size = args.atom_embedding_size if args.atom_embedder != "concat" else (1+data_handler.max_arity)*args.atom_embedding_size
    args.state_embedding_size = args.atom_embedding_size if args.state_embedder != "concat" else args.atom_embedding_size*args.padding_atoms
    embedder.embed_dim = args.state_embedding_size

    # Define environment creation function
    def make_env(mode='train', seed=0, queries=None, labels=None):
        def _init():
            env = LogicEnv_gym(
                        batch_size= 2,
                        index_manager=index_manager,
                        data_handler=data_handler,
                        # queries=queries,
                        # labels=labels,
                        mode=mode,
                        corruption_mode=args.corruption_mode,
                        # corruption_scheme=args.corruption_scheme,
                        train_neg_pos_ratio=args.train_neg_pos_ratio,
                        seed=seed,
                        # dynamic_consult=args.dynamic_consult,
                        max_depth=args.max_depth,
                        # memory_pruning=args.memory_pruning,
                        end_proof_action=args.end_proof_action,
                        skip_unary_actions=args.skip_unary_actions,
                        truncate_atoms=args.truncate_atoms,
                        # truncate_states=args.truncate_states,
                        # padding_atoms=args.padding_atoms,
                        padding_states=args.padding_states,
                        device='cpu', 
                        # engine=args.engine,
                        )
            # env = Monitor(env)
            return env
        return _init

    # Create vectorized environments for training
    env_seeds = np.random.randint(0, 2**10, size=args.n_envs)
    env = BatchedDummyVecEnv([make_env(
                                    mode='train', 
                                    seed=int(env_seeds[i]), 
                                    queries=data_handler.train_queries, 
                                    labels=[1]*len(data_handler.train_queries)
                                    ) 
                                    for i in range(args.n_envs)])
    # env = DummyVecEnv([make_env(
    #                             mode='train', 
    #                             seed=int(env_seeds[i]), 
    #                             queries=data_handler.train_queries, 
    #                             labels=[1]*len(data_handler.train_queries)
    #                             ) 
    #                             for i in range(args.n_envs)])
    # env = SubprocVecEnv([make_env(
    #                             mode='train', 
    #                             seed=int(env_seeds[i]), 
    #                             queries=data_handler.train_queries, 
    #                             labels=[1]*len(data_handler.train_queries)
    #                             ) 
    #                             for i in range(args.n_envs)])

    # Create multiple environments for evaluation
    eval_env_seeds = np.random.randint(0, 2**10, size=args.n_eval_envs)
    eval_env = DummyVecEnv([make_env(
                                    mode='eval', 
                                    seed=int(eval_env_seeds[i]), 
                                    queries=data_handler.valid_queries,
                                    labels=[1]*len(data_handler.valid_queries),
                                    ) 
                                    for i in range(args.n_eval_envs)])
    # eval_env = SubprocVecEnv([make_env(
    #                                 mode='eval', 
    #                                 seed=int(eval_env_seeds[i]), 
    #                                 queries=data_handler.valid_queries,
    #                                 labels=[1]*len(data_handler.valid_queries),
    #                                 ) 
    #                                 for i in range(args.n_eval_envs)])

    callback_env_seeds = np.random.randint(0, 2**10, size=args.n_callback_envs)
    callback_env = DummyVecEnv([make_env(
                                    mode='eval', 
                                    seed=int(callback_env_seeds[i]), 
                                    queries=data_handler.valid_queries,
                                    labels=[1]*len(data_handler.valid_queries),
                                    ) 
                                    for i in range(args.n_callback_envs)])
    # callback_env = SubprocVecEnv([make_env(
    #                                 mode='eval',
    #                                 seed=int(callback_env_seeds[i]),
    #                                 queries=data_handler.valid_queries,
    #                                 labels=[1]*len(data_handler.valid_queries),
    #                                 )
    #                                 for i in range(args.n_callback_envs)])

    # INIT MODEL
    if args.model_name == "PPO":
        # from model_SB3 import PPO_custom as PPO
        from stable_baselines3 import PPO
        model = PPO(CustomActorCriticPolicy, 
                    env,
                    learning_rate=args.lr,
                    n_steps=args.n_steps,  # Adjust steps per environment
                    batch_size=args.batch_size,
                    n_epochs=args.n_epochs,
                    verbose=1, 
                    device=device,
                    ent_coef=args.ent_coef,
                    clip_range=args.clip_range,
                    policy_kwargs={'features_extractor_class':CustomCombinedExtractor,
                                    'features_extractor_kwargs':{'features_dim':embedder.embed_dim,
                                                                    'embedder': embedder}})
    else:
        raise ValueError("Model not supported")
    

    # TRAIN
    model_path = os.path.join(args.models_path,args.run_signature, args.run_signature + f'-seed_{args.seed_run_i}')
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

        eval_callback = CustomEvalCallback(eval_env=callback_env, 
                                    best_model_save_path=model_path if args.save_model else None,
                                    log_path=log_filename if use_logger else None,
                                    eval_freq=max(int(args.eval_freq//args.n_envs),1),
                                    n_eval_episodes=args.n_eval_queries,
                                    deterministic=True,
                                    render=False,
                                    name=model_name,
                                    callback_on_new_best=reward_threshold_callback if args.restore_best_val_model else None,
                                    # callback_after_eval=no_improvement_callback,
                                    )

        callbacks.append(eval_callback)
    
        checkpoint_callback = SB3ModelCheckpoint(model,monitor="rollout/ep_rew_mean", frequency=5000, 
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
                                n_envs = args.n_envs,
                                epochs = args.n_epochs)) 
            
            callbacks.append(WandbCallback(
                                            # gradient_save_freq=100,
                                            # model_save_path=WB_path,
                                            # verbose=2,
                                            ))


        callbacks = CallbackList(callbacks)
        model.learn(total_timesteps=args.timesteps_train, callback=callbacks)
        if args.restore_best_val_model:
            eval_callback.restore_best_ckpt()

        if use_WB:
            run.finish()   

    # TEST   

    print('\nTest set eval...')
    metrics_test = eval_corruptions(model,
                                    eval_env,
                                    data_handler.test_queries,
                                    corruption_mode=args.corruption_mode,
                                    corruptions=data_handler.test_corruptions if args.corruption_mode == 'static' else None,
                                    n_corruptions=args.test_negatives,
                                    consult_janus=False,
                                    )
    print_eval_info('Test', metrics_test)

    # if 'kinship' not in args.dataset_name or 'countries' not in args.dataset_name:
    #     print('Val set eval...')
    #     metrics_valid = eval_corruptions(model,
    #                                     eval_env,
    #                                     data_handler.valid_queries,
    #                                     corruption_mode=args.corruption_mode,
    #                                     corruptions=data_handler.valid_corruptions if args.corruption_mode == 'static' else None,
    #                                     n_corruptions=args.valid_negatives,
    #                                     consult_janus=False,
    #                                     )
    #     print_eval_info('Validation', metrics_valid)

    #     print('Train set eval...')
    #     metrics_train = eval_corruptions(model,
    #                                     eval_env,
    #                                     data_handler.train_queries,
    #                                     corruption_mode=args.corruption_mode,
    #                                     corruptions=data_handler.train_corruptions if args.corruption_mode == 'static' else None,
    #                                     n_corruptions=args.train_neg_pos_ratio,
    #                                     consult_janus=True,
    #                                     )
    #     print_eval_info('Train', metrics_train)
    # else:
    #     metrics_train = {k: 0 for k in metrics_test.keys()}
    #     metrics_valid = {k: 0 for k in metrics_test.keys()}
    metrics_train = {k: 0 for k in metrics_test.keys()}
    metrics_valid = {k: 0 for k in metrics_test.keys()}



    return metrics_train, metrics_valid, metrics_test

