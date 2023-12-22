import torch
import os
from policy.policy import Policy
from policy import (
    PureRL_PolicyCallback, 
    ChangeDevice_Callback
)
from stable_baselines3.common.callbacks import (
    CallbackList, 
    EveryNTimesteps
)
import wandb
from wandb.integration.sb3 import WandbCallback
import argparse
from custom_env import (
    envs_dict, 
    build_vec_env
)
from src import (
    NATS_Interface,
    to_scientific_notation, 
    seed_all
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
    parser.add_argument("--algorithm", default="PPO", type=str, help="RL Algorithm. One in ['ppo', 'trpo', 'a2c']")
    parser.add_argument("--dataset", default="cifar10", type=str, help="Dataset on which to run the search. One in ['cifar10', 'cifar100', 'imagenet']")
    parser.add_argument("--env", default="oscar", type=str, help=f"Environment to be used. One in {list(envs_dict.keys())}")
    parser.add_argument("--searchspace", default="nats", type=str, help=f"Searchspace to be used. One in ['nats', 'fbnet']")
    parser.add_argument("--target-device", default="raspi4", type=str, help="Target device (for hardware aware algorithms only)")
    parser.add_argument("--verbose", default=0, type=int, help="Verbosity value")
    parser.add_argument("--train-timesteps", default=1e5, type=float, help="Number of timesteps to train the RL algorithm with")
    parser.add_argument("--evaluation-frequency", default=1e4, type = float, help="Frequency with which to evaluate policy against random fair opponent")
    parser.add_argument("--test-episodes", default=25, type=int, help="Number of test matches the agent plays during periodic evaluation")
    parser.add_argument("--resume-training", action="store_true", help="Whether or not load and keep train an already trained model")
    parser.add_argument("--model-path", default="models/", type=str, help="Path to which the model to incrementally train is stored")
    parser.add_argument("--seed", default=777, type=int, help="Random seed setted")
    parser.add_argument("--gamma", default=0.6, type=float, help="Discount factor")
    parser.add_argument("--learning-rate", default=3e-4, type=float, help="Learning rate for Deep RL training")
    parser.add_argument("--n-envs", default=1, type=int, help="Number of different envs to create at training time")
    parser.add_argument("--offline", action="store_true", help="Wandb does not sync anything to the cloud")
    parser.add_argument("--epsilon-scheduling", default="const", type=str, help="Whether or not to use scheduling for the epsilon parameter within PPO.\
                                                                                Accepted schedulers are ['exp', 'sawtooth', 'sine']. min_eps, max_eps = 0.1, 0.3")
    parser.add_argument("--use-wandb-callback", default=False, help="Whether or not to append the SB3 Wandb callback to the list of used callbacks.")
    parser.add_argument("--parallel-envs", default=True, type=boolean_string, help="Whether or not to train the agent using envs in multiprocessing")
    parser.add_argument("--synthetic-devices", default=True, type=boolean_string, help="Whether or not to train the agent using synthetic devices.\
                                                                                        In v1, simply triggers a change in the underlying lookup table for the interface object used.")

    #parser.add_argument("--default", action="store_true", help="Default mode, ignore all configurations")
    parser.add_argument("--debug", action="store_true", help="Default mode, ignore all configurations")
    
    return parser.parse_args()

def main():
    """Performs training and logs info to wandb."""
    args = parse_args()

    algorithm=args.algorithm
    dataset=args.dataset
    env_name=args.env
    searchspace=args.searchspace
    target_device=args.target_device
    verbose=args.verbose
    train_timesteps=int(args.train_timesteps)
    evaluate_every=int(args.evaluation_frequency)
    test_episodes=int(args.test_episodes)
    resume_training=args.resume_training
    model_path=args.model_path
    seed=args.seed
    GAMMA=args.gamma
    learning_rate=args.learning_rate
    n_envs=args.n_envs
    offline=args.offline
    epsilon_scheduling=args.epsilon_scheduling
    use_wandb_callback=args.use_wandb_callback
    parallel_envs=args.parallel_envs
    use_synthetic_devices=args.synthetic_devices

    if args.debug: 
        algorithm="PPO"
        dataset="cifar10"
        env_name="marcella"
        searchspace="nats"
        n_envs=3
        train_timesteps=int(1e4)
        test_episodes=25
        evaluate_every=int(1e3)
        verbose=1
        offline=True
        epsilon_scheduling="sawtooth"
        use_wandb_callback=True
        multitask=False
        parallel_envs=False
        target_device="pixel3"
        resume_training=False
        model_path="models/PPO_oscar_2e6_raspi4_6040.zip"
        use_synthetic_devices=True


    if searchspace.lower() == "nats":
            searchspace_interface = NATS_Interface(dataset=dataset, use_synthetic_devices=use_synthetic_devices)
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
    
    # silencing wandb output
    os.environ["WANDB_SILENT"] = "true" 

    run = wandb.init(
        project="Debug-Oscar",
        config=training_config,
        name=default_name,
        mode="offline" if offline else "online",
        sync_tensorboard=True if use_wandb_callback else None
    )
    
    # best models are saved in models - when specialized for target hardware are stored in specific subfolders
    best_model_path = f"models/{target_device}/{run.name}"

    if multitask:
        best_model_path = "models/marcella/" + f"{target_device}/{run.name}"

    # this callback is wrapped in `EveryNTimesteps`
    inner_callback = PureRL_PolicyCallback(
        env=envs,
        n_eval_episodes=test_episodes, 
        best_model_path=best_model_path
    )
    
    # invoke inner_callback every `evaluate_every` timesteps
    evaluation_callback = EveryNTimesteps(n_steps=evaluate_every, callback=inner_callback)
    callback_list = [evaluation_callback]

    if env.name == "marcella":
        changedevice_callback = ChangeDevice_Callback()
        # every 5th percent of the training procedure this procedure will change the underlying target device
        marcella_callback = EveryNTimesteps(n_steps=int(train_timesteps/20), callback=changedevice_callback)
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
        policy.set_epsilon_scheduler(kind=epsilon_scheduling)
    
    if use_wandb_callback:
        policy.model.tensorboard_log = f"runs/{run.id}"
        # also adding wandb callback to the picture here
        callback_list.append(
            WandbCallback(
                gradient_save_freq=100,
                model_save_path=f"models/{run.id}"
                )
            )
        
    print(f"Starting to train: {algorithm.upper()} on {env_name} for {to_scientific_notation(train_timesteps)} timesteps.")
    
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
        print("BestsFound: ", evaluation_callback.callback.bests_found, f"(out of {int(train_timesteps//evaluate_every)} evaluations)")
        print(f"Training completed! Training output available at: {best_model_path}.zip")
        print(f"Avg Return over test episodes: {round(avg_return, 2)} ± {round(std_return, 2)}")

    # exit from wandb run
    wandb.finish()

if __name__=="__main__":
    main()

