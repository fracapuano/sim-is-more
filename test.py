from policy.policy import Policy
import argparse
import time
import pickle
from src import (
    seed_all, 
    NATS_Interface,
)
from custom_env import (
    envs_dict,
    build_vec_env
)
import numpy as np

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
    parser.add_argument("--model-path", default="models/", type=str, help="Path to which the model to incrementally train is stored")
    parser.add_argument("--dataset", default="cifar10", type=str, help="Dataset on which to run the search. One in ['cifar10', 'cifar100', 'imagenet']")
    parser.add_argument("--target-device", default="raspi4", type=str, help="Target device. For Hardware aware search only.")
    parser.add_argument("--verbose", default=0, type=int, help="Verbosity value")
    parser.add_argument("--test-episodes", default=5, type=int, help="Number of test episodes for the RL agent")
    parser.add_argument("--seed", default=777, type=int, help="Random seed setted")
    parser.add_argument("--fittest-ever", action="store_true", help="When provided, policy is prompted to return fittest individual EVER.")
    parser.add_argument("--save-results", action="store_true", help="Number of episodes to use.")

    #parser.add_argument("--default", action="store_true", help="Default mode, ignore all configurations")
    parser.add_argument("--debug", action="store_true", help="Default mode, ignore all configurations")
    
    return parser.parse_args()

def main():
    """Performs training and logs info to wandb."""
    args = parse_args()

    model_path=args.model_path
    dataset=args.dataset
    target_device=args.target_device
    verbose=args.verbose
    test_episodes=args.test_episodes
    seed=args.seed
    fittest_ever=args.fittest_ever
    
    if args.debug: 
        model_path="models/PPO_renas_100e5"
        dataset="ImageNet16-120"
        target_device=None
        test_episodes=1
        seed=0
        verbose=1
        fittest_ever=False

    # set seed for reproducibility
    seed_all(seed=seed)
    
    # model path is always in the format directory/<Algorithm>_<Env>_<TrainTimesteps>.zip
    algorithm, environment, *_ = model_path.split("/")[-1].split("_")

    # accessing to the env defined at the policy level
    searchspace_interface = NATS_Interface(dataset=dataset)

    # create env (gym.Env)
    env = envs_dict[environment.lower()](searchspace_api=searchspace_interface)
    # wrap env into a vectorized environment
    env = build_vec_env(
        env_=env,
        n_envs=1 # only one environment is needed for testing
    )
    
    # instantiate a testing suite
    policy = Policy(
        algo=algorithm,
        env=env,
        load_from_pathname=model_path
    )

    if verbose > 0: 
        print(f"Loading model from {model_path}")
        print(f"Testing {algorithm} on {environment}")
    
    # saving up the episode return and time duration
    returns, durations = np.zeros(test_episodes), np.zeros(test_episodes)
    
    initial_nets, terminal_nets = [], []

    for ep in range(test_episodes):
        done = False
        obs = env.reset() # Reset environment to initial state
        # save initial network
        initial_net = env.get_attr("current_net")[0].architecture
        initial_nets.append(initial_net)

        episode_return = 0
        start = time.time()

        while not done:  # Until the episode is over
            action = policy.predict(observation=obs, deterministic=True)  # deterministic is True while evaluating            
            obs, reward, done, _ = env.step(action)	# Step the simulator to the next timestep

            episode_return += reward
        
        terminal_net = env.get_attr("current_net")[0].architecture
        terminal_nets.append(terminal_net)
        
        durations[ep] = time.time()-start
        returns[ep] = episode_return
    
    # returning also the fittest individual in the environment buffer
    if fittest_ever:
        best_individual = max(list(env.history.keys()), key=lambda k: env.history[k].fitness)
        print(f"Fittest individual (ever): {best_individual}")
    else: 
        best_individual = env.get_attr("current_net")[0].architecture
        print(f"Fittest individual in terminal population: {best_individual}")
    
    print("Average episode return {:.4g}".format(returns.mean()))
    if fittest_ever:
        print("Final individual test accuracy: {:.5g}".format(searchspace_interface.\
                                                              architecture_to_accuracy(best_individual)))
    else:
        print("Final individual test accuracy: {:.5g}".format(searchspace_interface.\
                                                              list_to_accuracy(best_individual)))

    print("Final latency: {:.5g}".format(
        searchspace_interface.list_to_score(best_individual, f"{target_device}_latency")))

    if args.save_results:
        # saving all final networks obtained
        with open("TestResults.pkl", "wb") as results:
            pickle.dump(list(zip(initial_nets, terminal_nets)), results)

if __name__=="__main__":
    main()

