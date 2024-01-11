import re
import time
import pickle
import argparse
import numpy as np
from src import (
    seed_all, 
    NATS_Interface,
)
from policy import (
    Policy, 
    TransitionsHistoryWrapper
)
from custom_env import envs_dict
from typing import Text, Optional


def boolean_string(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'

def extract_historylen_from_path(model_path:Text)->Optional[int]:
    """Extract the history_len from the model path."""
    # Define the regex pattern to match 'historylen=<any_number>'
    pattern = r"history_len=(\d+)"

    # Search for the pattern in the input string
    match = re.search(pattern, model_path)

    # Extract and return the number if found, otherwise return None
    return int(match.group(1)) if match else None

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
    parser.add_argument("--best-ever", action="store_true", help="When provided, policy is prompted to return fittest individual EVER.")
    parser.add_argument("--render", action="store_true", help="When provided, triggers rendering.")
    parser.add_argument("--save-trajectories", action="store_true", help="When provided, saves all tested trajectories (initial_net, final_net).")

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
    best_ever=args.best_ever
    save_trajectories=args.save_trajectories
    
    if args.debug: 
        model_path="models/oscar/edgegpu/lr=3.0e-4/gamma=0.6/seed=777/PPO_oscar_1.0e5/PPO_oscar_1.0e5.zip"
        dataset="ImageNet16-120"
        target_device="edgegpu"
        test_episodes=1
        seed=0
        verbose=1
        best_ever=False

    # set seed for reproducibility
    seed_all(seed=seed)
    
    # model path is always in the format directory/<Algorithm>_<Env>_<TrainTimesteps>.zip
    algorithm, environment, *_ = model_path.split("/")[-1].split("_")

    # accessing to the env defined at the policy level
    searchspace_interface = NATS_Interface(dataset=dataset)

    # create env (gym.Env)
    env = envs_dict[environment.lower()](searchspace_api=searchspace_interface)
    
    # setting a target device for hardware aware search at test time
    env.target_device = target_device

    if environment == "marcella-plus":
        env = TransitionsHistoryWrapper(env=env, history_len=extract_historylen_from_path(model_path))

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
    
    initial_nets, episode_bests = [], []

    for ep in range(test_episodes):
        done = False
        obs, info = env.reset() # Reset environment to initial state
        # save initial network
        initial_net = env.current_net.architecture
        initial_nets.append(initial_net)

        episode_return = 0
        start = time.time()

        while not done:  # Until the episode is over
            action = policy.predict(observation=obs, deterministic=True)  # deterministic is True while evaluating            
            obs, reward, terminated, truncated, info = env.step(action)	# Step the simulator to the next timestep

            done = terminated or truncated

            episode_return += reward
            if args.render:
                env.render()
                
        durations[ep] = time.time() - start
        returns[ep] = episode_return
    
        best_individual = env.current_net.architecture
        episode_bests.append(best_individual)
        print(f"Network Designed: {best_individual}")
    
    print("Average episode return {:.4g}".format(returns.mean()))
    if best_ever:
        best_ever_net = max(episode_bests, key=lambda x: searchspace_interface.list_to_accuracy(x))
        print(f"Out of {test_episodes} test episodes, the best network designed is {best_ever_net}")
        print("With a validation accuracy of {:.5g}".format(searchspace_interface.architecture_to_accuracy(best_individual)))

    if save_trajectories:
        # saving all final networks obtained
        with open("test_results.pkl", "wb") as results:
            pickle.dump(list(zip(initial_nets, episode_bests)), results)

if __name__=="__main__":
    main()

