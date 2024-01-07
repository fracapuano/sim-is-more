import os
import wandb
import torch
import argparse
from policy import (
    PeriodicEvalCallback, 
    ChangeDevice_Callback,
    TransitionsHistoryWrapper
)
from src import (
    NATS_Interface,
    to_scientific_notation, 
    seed_all
)
from custom_env import (
    envs_dict, 
    build_vec_env
)
from policy.policy import Policy
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import (
    CallbackList, 
    EveryNTimesteps
)

def boolean_string(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'

def parse_args()->object: 
    """Args function. 
    Returns:
        (object): args parser
    """
    parser = argparse.ArgumentParser()
    """The following args help define the scope of the problem to be solved."""
    parser.add_argument("--dataset", default="cifar10", type=str, help="Dataset on which to run the search. One in ['cifar10', 'cifar100', 'imagenet']")
    parser.add_argument("--searchspace", default="nats", type=str, 
                        choices=["nats"], help=f"Searchspace to be used. One in ['nats']")  # fbnet will follow
    parser.add_argument("--target-device", default="edgegpu", 
                        choices=["edgegpu", "raspi4", "pixel3", "eyeriss", "fpga"], type=str, help="Target device to be used.")
    
    """The following args help define the training procedure."""
    parser.add_argument("--algorithm", default="PPO", type=str, 
                        choices=["ppo", "trpo", "a2c"], help="RL Algorithm. One in ['ppo', 'trpo', 'a2c']")
    parser.add_argument("--env", default="oscar", type=str,
                        choices=list(envs_dict.keys()), help=f"Environment to be used. One in {list(envs_dict.keys())}")
    parser.add_argument("--verbose", default=0, type=int, help="Verbosity value")
    parser.add_argument("--train-timesteps", default=1e5, type=float, help="Number of timesteps to train the RL algorithm with")
    parser.add_argument("--evaluation-frequency", default=1e4, type = float, help="Frequency with which to evaluate policy during training.")
    parser.add_argument("--history-size", default=5, type=int, help="Number of previous states to be used in history-based policy")
    parser.add_argument("--test-episodes", default=25, type=int, help="Number of test episodes carried out during periodic evaluation")
    parser.add_argument("--seed", default=777, type=int, help="Random seed setted")
    parser.add_argument("--gamma", default=0.6, type=float, help="Discount factor")
    parser.add_argument("--learning-rate", default=3e-4, type=float, help="Learning rate for Deep RL training")
    parser.add_argument("--n-envs", default=1, type=int, help="Number of different envs to create at training time")
    parser.add_argument("--parallel-envs", default=True, type=boolean_string, help="Whether or not to train the agent using envs in multiprocessing")
    parser.add_argument("--offline", action="store_true", help="Wandb does not sync anything to the cloud")
    parser.add_argument("--epsilon-scheduling", default="const", type=str, 
                        choices=["exp", "sawtooth", "sine"], help="Whether or not to use scheduling for the epsilon parameter within PPO. \
                                                                   Accepted schedulers are ['exp', 'sawtooth', 'sine']")
    parser.add_argument("--min-eps", default=0.1, type=float, help="Minimum value for epsilon in epsilon scheduling")
    parser.add_argument("--max-eps", default=0.3, type=float, help="Maximum value for epsilon in epsilon scheduling")
    parser.add_argument("--use-wandb-callback", default=False, help="Whether or not to append the SB3 Wandb callback to the list of used callbacks.")
    
    """The following args are used to resume training from checkpoints."""
    parser.add_argument("--resume-training", action="store_true", help="Whether or not load and keep train an already trained model")
    parser.add_argument("--model-path", default="models/", type=str, help="Path to which the model to incrementally train is stored")
    
    #parser.add_argument("--default", action="store_true", help="Default mode, ignore all configurations")
    parser.add_argument("--debug", action="store_true", help="Default mode, ignore all configurations")
    
    return parser.parse_args()

def main():
    """Performs training and logs info to wandb."""
    args = parse_args()

    # unpacking args
    args = parse_args()

    # Unpacking args in the same order as they are defined in parse_args
    dataset = args.dataset
    searchspace = args.searchspace
    target_device = args.target_device

    algorithm = args.algorithm
    env_name = args.env
    verbose = args.verbose
    train_timesteps = int(args.train_timesteps)
    evaluate_every = int(args.evaluation_frequency)
    history_size = args.history_size
    test_episodes = int(args.test_episodes)
    seed = args.seed
    GAMMA = args.gamma
    learning_rate = args.learning_rate
    n_envs = args.n_envs
    parallel_envs = args.parallel_envs
    offline = args.offline
    epsilon_scheduling = args.epsilon_scheduling
    min_eps = args.min_eps
    max_eps = args.max_eps
    use_wandb_callback = args.use_wandb_callback

    resume_training = args.resume_training
    model_path = args.model_path

    if args.debug:
        # Debug mode settings in the same order
        dataset = "cifar10"
        searchspace = "nats"
        target_device = "edgegpu"
        algorithm = "PPO"
        env_name = "oscar"
        verbose = 1
        train_timesteps = int(1_000)
        evaluate_every = int(10)
        history_size = 5  # Default value or specify as needed
        test_episodes = 25
        seed = 777  # Default value or specify as needed
        GAMMA = 0.6  # Default value or specify as needed
        learning_rate = 3e-4  # Default value or specify as needed
        n_envs = 3
        parallel_envs = False
        offline = False
        epsilon_scheduling = "const"
        min_eps = 0.1  # Default value or specify as needed
        max_eps = 0.3  # Default value or specify as needed
        use_wandb_callback = True
        # resume_training and model_path can be set as required
        # resume_training=False
        # model_path="models/PPO_oscar_2e6_raspi4_6040.zip"

    # silencing wandb output
    # os.environ["WANDB_SILENT"] = "true" 

    if searchspace.lower() == "nats":
            searchspace_interface = NATS_Interface(dataset=dataset)
    else:
        raise NotImplementedError(
            f"Searchspace {searchspace} not implemented yet. Searchspaces that will be implemented: ['nats', 'fbnet']. FBNet to do."
        )
    
    # set seed for reproducibility
    seed_all(seed=seed)
    # to allow multiprocessing of various envs
    torch.set_num_threads(n_envs)

    # create env (gym.Env)
    env = envs_dict[env_name.lower()](searchspace_api=searchspace_interface)

    # changing the target device based on user input
    env.target_device = target_device

    # wrapping env in a history wrapper when needed
    if env.name == "marcella-plus":
        env = TransitionsHistoryWrapper(env=env, history_size=history_size)
    
    # build the envs according to spec
    envs = build_vec_env(
        env_=env,
        n_envs=n_envs, 
        subprocess=parallel_envs)
    
    # training config dictionary
    training_config = dict(
        algorithm=algorithm,
        env_name=env.name,
        discount_factor=GAMMA,
        train_timesteps=to_scientific_notation(train_timesteps),
        random_seed=seed,
        target_device=target_device
    )

    if verbose > 0: 
        print(training_config)

    # init wandb run - learning rate -> discount factor -> random seed -> algorithm_env_training_steps
    default_name = f"lr={to_scientific_notation(learning_rate)}"+\
                   f"/gamma={GAMMA}/seed={seed}/"+\
                   f"{algorithm.upper()}_{env.name}_{to_scientific_notation(train_timesteps)}"
    

    run = wandb.init(
        project="Debug-Oscar",
        config=training_config,
        name=default_name,
        mode="offline" if offline else "online",
        sync_tensorboard=True if use_wandb_callback else None
    )
    
    # best models are saved in models - when specialized for target hardware are stored in specific subfolders
    best_model_path = f"models/{env_name}/{target_device}/{run.name}"

    # this callback is wrapped in `EveryNTimesteps`
    inner_callback = PeriodicEvalCallback(
        env=envs,
        n_eval_episodes=test_episodes, 
        best_model_path=best_model_path
    )
    
    # invoke inner_callback every `evaluate_every` timesteps
    evaluation_callback = EveryNTimesteps(n_steps=evaluate_every, callback=inner_callback)
    callback_list = [evaluation_callback]

    if env.name == "marcella":
        changedevice_callback = ChangeDevice_Callback()
        # every 1 percent of the training procedure this procedure will change the underlying target device
        marcella_callback = EveryNTimesteps(n_steps=int(train_timesteps/100), callback=changedevice_callback)
        callback_list.append(marcella_callback)

    # instantiate a policy object
    policy = Policy(
        algo=algorithm,
        env=envs,
        lr=learning_rate,
        gamma=GAMMA,
        seed=seed,
        load_from_pathname=model_path if resume_training else None)
    
    # optionally triggers the epsilon scheduler to be triggered!
    if epsilon_scheduling.lower() != "const":
        policy.set_epsilon_scheduler(kind=epsilon_scheduling, start_epsilon=max_eps, end_epsilon=min_eps)
    
    if use_wandb_callback:
        policy.model.tensorboard_log = f"runs/{run.id}"
        # also adding wandb callback to the picture here
        callback_list.append(
            WandbCallback(
                gradient_save_freq=100 if args.debug else 0,  # to evaluate how training proceeds when debugging
                model_save_path=f"models/{run.id}"
                )
            )
        
    msg = f"Starting to train: {algorithm.upper()} on {env_name} for {to_scientific_notation(train_timesteps)} timesteps"
    print(msg)
    
    # creating one callback list only
    training_callbacks = CallbackList(callback_list)
    # training policy using multiple callbacks
    avg_return, std_return, *_ = policy.train(
        timesteps=train_timesteps,
        n_eval_episodes=test_episodes,
        callback_list=training_callbacks,
        return_best_model=True, 
        best_model_save_path=best_model_path
    )
    # print the number of times a better env is found
    if verbose > 0: 
        msg = f"""
            During training, the best model has been updated: {evaluation_callback.callback.bests_found} \
            times (out of {int(train_timesteps//evaluate_every)} evaluations)
        """
        print(msg)
        
        print(f"\tTraining completed! Training output available at: {best_model_path}.zip")
        print(f"\tAvg Return over latest test episodes: {round(avg_return, 2)} ± {round(std_return, 2)}")

    # end wandb
    run.finish(quiet=True)

if __name__=="__main__":
    main()

