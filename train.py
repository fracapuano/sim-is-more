import os
import json
import wandb
import torch
import warnings
import argparse
from utils import (
    boolean_string, 
    float_range, 
    create_searchspace, 
    get_distribution_devices
)
from policy import (
    PeriodicEvalCallback, 
    ChangeDevice_Callback,
    TransitionsHistoryWrapper
)
from src import (
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
# ignoring Gymnasium getattr warnings
warnings.filterwarnings('ignore', message='.*get variables from other wrappers', )  

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
    parser.add_argument("--task-weight", default=0.5, type=float, help="Task-associated weight in the reward function. This directly balances the hardware performance.")
    parser.add_argument("--hardware-weight", default=0.5, type=float, help="Hardware-associated weight in the reward function. This directly balances the hardware performance.")
    
    """The following args help define the training procedure."""
    parser.add_argument("--algorithm", default="PPO", type=str, 
                        choices=["ppo", "trpo", "a2c"], help="RL Algorithm. One in ['ppo', 'trpo', 'a2c']")
    parser.add_argument("--score-list", nargs="*", type=str, default=["naswot_score", "logsynflow_score", "skip_score"])
    parser.add_argument("--normalization-type", default="minmax", type=str, choices=["std", "minmax"], help="Normalization type to be used for hardware cost. One in ['std', 'minmax']")
    parser.add_argument("--env_name", default="oscar", type=str,
                        choices=list(envs_dict.keys()), help=f"Environment to be used. One in {list(envs_dict.keys())}")
    parser.add_argument("--verbose", default=0, type=int, help="Verbosity value")
    parser.add_argument("--leave-out-devices", nargs="*", default=[], type=str, help="Target device not to be used to fit Marcella(+).") 
    parser.add_argument("--distribution-devices", nargs="*", default=[], type=str, help="Target device to be used to compute the distribution for Marcella(+).") 
    parser.add_argument("--change-device-every", default=1, type=float_range(1, 100), help="Percentage of training timesteps after which to change the target device to be used.")
    parser.add_argument("--train-timesteps", default=1e5, type=float, help="Number of timesteps to train the RL algorithm with")
    parser.add_argument("--evaluate-every", default=1e4, type = float, help="Frequency with which to evaluate policy during training.")
    parser.add_argument("--history-len", default=5, type=int, help="Number of previous states to be used in history-based policy")
    parser.add_argument("--test-episodes", default=25, type=int, help="Number of test episodes carried out during periodic evaluation")
    parser.add_argument("--seed", default=777, type=int, help="Random seed setted")
    parser.add_argument("--gamma", default=0.6, type=float, help="Discount factor")
    parser.add_argument("--learning-rate", default=3e-4, type=float, help="Learning rate for Deep RL training")
    parser.add_argument("--entropy-coef", default=0., type=float, help="Entropy coefficient for A2C-based training")
    parser.add_argument("--n-envs", default=1, type=int, help="Number of different envs to create at training time")
    parser.add_argument("--parallel-envs", default=True, type=boolean_string, help="Whether or not to train the agent using envs in multiprocessing")
    parser.add_argument("--offline", action="store_true", help="Wandb does not sync anything to the cloud")
    parser.add_argument("--epsilon-scheduling", default="const", type=str, 
                        choices=["const", "exp", "sawtooth", "sine"], help="Whether or not to use scheduling for the epsilon parameter within PPO. \
                                                                   Accepted schedulers are ['const', 'exp', 'sawtooth', 'sine']")
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

    if args.debug:
        with open("src/default_training.json", "r") as f:
            default_args = argparse.Namespace(**json.load(f))
            # overriding default args with the ones passed as arguments
            for key, value in vars(default_args).items():
                setattr(args, key, value)
    
    # silencing wandb output
    # os.environ["WANDB_SILENT"] = "true" 
    
    # set seed for reproducibility
    seed_all(seed=args.seed)
    # to allow multiprocessing of various envs
    torch.set_num_threads(args.n_envs)
    
    searchspace_interface = create_searchspace(searchspace=args.searchspace, dataset=args.dataset)
    # create env (gym.Env)
    env = envs_dict[args.env_name.lower()](
        searchspace_api=searchspace_interface, 
        scores=args.score_list,
        target_device=args.target_device,
        weights=[args.task_weight, args.hardware_weight],
        normalization_type=args.normalization_type
    )

    history_wrap = lambda e: TransitionsHistoryWrapper(env=e, history_len=args.history_len)
    wrappers_list = [history_wrap] if env.name == "marcella-plus" else None
    
    if "marcella" in env.name:
        # leaving some devices from the list of devices to be used while training
        env.devices = get_distribution_devices(
            available_devices=searchspace_interface.get_devices(), chosen_devices=args.distribution_devices
        )
    
    # build the envs according to spec
    envs = build_vec_env(
        env=env,
        n_envs=args.n_envs, 
        subprocess=args.parallel_envs, 
        wrappers_list=wrappers_list)
    
    # training config dictionary
    training_config = dict(
        algorithm=args.algorithm,
        env_name=env.name,
        gamma=args.gamma,
        train_timesteps=to_scientific_notation(args.train_timesteps),
        random_seed=args.seed,
        target_device=args.target_device,
        task_weight=args.task_weight,
        hardware_weight=args.hardware_weight,
        score_list=args.score_list,
        learning_rate=args.learning_rate,
        epsilon_scheduling=args.epsilon_scheduling,
        min_eps=args.min_eps,
        max_eps=args.max_eps,
        history_len=args.history_len
    )

    if args.verbose > 0: 
        print(training_config)
    
    run = wandb.init(
        project="Revamp-Oscar",
        config=training_config,
        mode="offline" if args.offline else "online",
        sync_tensorboard=True if args.use_wandb_callback else None
    )
    
    # best models are saved in models - when specialized for target hardware are stored in specific subfolders
    best_model_path = f"models/{args.env_name}/{args.target_device}/{run.name}"
    
    # dumping training config to json file
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    with open(f"{best_model_path}/training_config.json", "w+") as f:
        args_dict = vars(args)
        if "marcella" in args.env_name:
            args_dict["devices"] = env.devices
            args_dict["target-device"] = None
        
        json.dump(args_dict, f, indent=4)

    # this callback is wrapped in `EveryNTimesteps`
    inner_callback = PeriodicEvalCallback(
        env=envs,
        n_eval_episodes=args.test_episodes, 
        best_model_path=best_model_path,
        log_video=False
    )
    
    # invoke inner_callback every `evaluate_every` timesteps
    evaluation_callback = EveryNTimesteps(n_steps=args.evaluate_every, callback=inner_callback)
    callback_list = [evaluation_callback]

    if env.name == "marcella":
        changedevice_callback = ChangeDevice_Callback()
        # every 1 percent of the training procedure this procedure will change the underlying target device
        marcella_callback = EveryNTimesteps(
            n_steps=int(args.change_device_every * args.train_timesteps / 100),
            callback=changedevice_callback
        )
        callback_list.append(marcella_callback)

    # instantiate a policy object
    policy = Policy(
        algo=args.algorithm,
        env=envs,
        lr=args.learning_rate,
        gamma=args.gamma,
        seed=args.seed,
        load_from_pathname=args.model_path if args.resume_training else None,
        entropy_coef=args.entropy_coef)
    
    # optionally triggers the epsilon scheduler to be triggered!
    if args.epsilon_scheduling.lower() != "const":
        policy.set_epsilon_scheduler(kind=args.epsilon_scheduling, start_epsilon=args.max_eps, end_epsilon=args.min_eps)
    
    if args.use_wandb_callback:
        policy.model.tensorboard_log = f"runs/{run.id}"
        # also adding wandb callback to the picture here
        callback_list.append(
            WandbCallback(
                gradient_save_freq=1000 if args.debug else 0,  # to evaluate how training proceeds when debugging
                )
            )
        
    msg = f"Starting to train: {args.algorithm.upper()} on {args.env_name} for {to_scientific_notation(args.train_timesteps)} timesteps"
    print(msg)
    
    # creating one callback list only
    training_callbacks = CallbackList(callback_list)
    # training policy using multiple callbacks
    avg_return, std_return, *_ = policy.train(
        timesteps=args.train_timesteps,
        n_eval_episodes=args.test_episodes,
        callback_list=training_callbacks,
        return_best_model=True, 
        best_model_save_path=best_model_path
    )
    # print the number of times a better env is found
    if args.verbose > 0: 
        msg = \
            f"""During training, the best model has been updated: {evaluation_callback.callback.bests_found} """ +\
            f"""times (out of {int(args.train_timesteps//args.evaluate_every)} evaluations)"""
        print(msg)
        
        print(f"\tTraining completed! Training output available at: {best_model_path}")
        print(f"\tAverage Return over latest test episodes: {round(avg_return, 2)} ± {round(std_return, 2)}")

    # end wandb
    run.finish(quiet=True)

if __name__=="__main__":
    main()

