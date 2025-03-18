import numpy as np
import os
import random
import torch
from typing import Dict

from environments.env_logic_gym import LogicEnv_gym, IndexManager
from utils import get_device, print_eval_info
from my_callbacks import SB3ModelCheckpoint, EvalCallback, EpochTimingCallback
from dataset import DataHandler
from model_SB3 import CustomActorCriticPolicy, CustomCombinedExtractor
from embeddings import get_embedder
from neg_sampling import get_sampler
from model_eval import eval_corruptions

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    StopTrainingOnMaxEpisodes,
    StopTrainingOnRewardThreshold,
    CallbackList,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
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

    data_handler = DataHandler(
        dataset_name=args.dataset_name,
        base_path=args.data_path,
        janus_file=args.janus_file,
        train_file= args.train_file,
        valid_file=args.valid_file,
        test_file= args.test_file,
        corruption_mode=args.corruption_mode,
        non_provable_corruptions=args.non_provable_corruptions,
        non_provable_queries=args.non_provable_queries,)
    data_handler= data_handler.info
    args.valid_negatives = len(data_handler.valid_queries) if args.valid_negatives == -1 else min(args.valid_negatives, len(data_handler.valid_queries))
    args.test_negatives = len(data_handler.test_queries) if args.test_negatives == -1 else min(args.test_negatives, len(data_handler.test_queries))

    index_manager = IndexManager(data_handler.constants,
                                data_handler.predicates,
                                data_handler.variables if args.rule_depend_var else set(),
                                data_handler.constant_no,
                                data_handler.predicate_no,
                                args.variable_no,
                                constants_images=data_handler.constants_images if args.dataset_name == 'mnist_addition' else set(),
                                constant_images_no=data_handler.constant_images_no if args.dataset_name == 'mnist_addition' else 0,
                                rules=data_handler.rules,
                                rule_depend_var=args.rule_depend_var,
                                max_arity=data_handler.max_arity,
                                device=device,
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


    args.n_envs = getattr(args, 'n_envs', 100)
    args.eval_envs = getattr(args, 'eval_envs', 10)  # Default to 10 eval environments
   
    # Define environment creation function
    def make_env(eval_mode=False, seed=0, valid_negatives=None):
        def _init():
            env = LogicEnv_gym(max_depth=args.max_depth,
                        device=device, 
                        index_manager=index_manager,
                        data_handler=data_handler,
                        seed=seed,
                        corruption_mode=args.corruption_mode,
                        corruption_scheme=args.corruption_scheme,
                        train_neg_pos_ratio=args.train_neg_pos_ratio,
                        limit_space=args.limit_space,
                        dynamic_consult=args.dynamic_consult,
                        end_proof_action=args.end_proof_action,
                        padding_atoms=args.padding_atoms,
                        padding_states=args.padding_states,
                        eval=eval_mode,
                        valid_negatives=valid_negatives if eval_mode else None)
            return env
        return _init

    # Create vectorized environments for training
    env_seeds = np.random.randint(0, 2**10, size=args.n_envs)
    env = DummyVecEnv([make_env(eval_mode=False, seed=int(env_seeds[i])) for i in range(args.n_envs)])

    # Create multiple environments for evaluation
    eval_env_seeds = np.random.randint(0, 2**10, size=args.eval_envs)
    eval_env = DummyVecEnv([make_env(eval_mode=True, seed=int(eval_env_seeds[i]), 
                                    valid_negatives=args.valid_negatives) for i in range(args.eval_envs)])

    # INIT MODEL
    if args.model_name == "PPO":
        from model_SB3 import PPO_custom as PPO
        model = PPO(CustomActorCriticPolicy, 
                    env,
                    learning_rate=args.lr,
                    n_steps=args.n_steps // args.n_envs,  # Adjust steps per environment
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

        eval_callback = EvalCallback(eval_env=eval_env, 
                                    model_path=model_path if args.save_model else None,
                                    log_path=log_filename if use_logger else None,
                                    eval_freq=max(int(args.eval_freq/args.n_envs),1),
                                    n_eval_episodes=args.valid_negatives,
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

    print('Test set eval...')
    metrics_test = eval_corruptions(data_handler.test_queries,
                                        eval_env, model, verbose=0,
                                        corruption_mode=args.corruption_mode,
                                        corruptions=data_handler.test_corruptions,
                                        n_corruptions=args.test_negatives,
                                        n_eval_envs=args.eval_envs)  
    print_eval_info('Test', metrics_test)

    if 'kinship' not in args.dataset_name:
        print('Val set eval...')
        metrics_valid = eval_corruptions(data_handler.valid_queries,
                                            eval_env, model, verbose=0,
                                            corruption_mode=args.corruption_mode,
                                            corruptions=data_handler.valid_corruptions,
                                            n_corruptions=args.valid_negatives,
                                            n_eval_envs=args.eval_envs)  
        print_eval_info('Validation', metrics_valid)

        print('Train set eval...')
        metrics_train = eval_corruptions(data_handler.train_queries,
                                            eval_env, model, verbose=0,
                                            corruption_mode=args.corruption_mode,
                                            corruptions=data_handler.train_corruptions,
                                            n_corruptions=args.train_neg_pos_ratio,
                                            n_eval_envs=args.eval_envs)    
        print_eval_info('Train', metrics_train)




    return metrics_train, metrics_valid, metrics_test

