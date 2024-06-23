import re
import json
import pickle
import argparse
import numpy as np
from custom_env import envs_dict
from typing import Text, Optional
from train import create_searchspace
from src.utils import to_scientific_notation
from policy import Policy, TransitionsHistoryWrapper


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
    parser.add_argument("--folder-path", default="models/", type=str, help="Path at which to find the model and the training config are stored")
    parser.add_argument("--use-custom-device", action="store_true", help="When not provided, uses the custom device provided in the training config")
    parser.add_argument("--target-device", default="edgegpu", type=str, help="Target device for hardware aware search")
    
    parser.add_argument("--verbose", default=1, type=int, help="Verbosity level")
    parser.add_argument("--test-episodes", default=5, type=int, help="Number of test episodes for the RL agent")
    parser.add_argument("--seed", default=777, type=int, help="Random seed setted")
    parser.add_argument("--best-ever", action="store_true", help="When provided, policy is prompted to return fittest individual across all test episodes.")
    parser.add_argument("--render", action="store_true", help="When provided, triggers rendering.")
    parser.add_argument("--save-trajectories", action="store_true", help="When provided, saves all tested trajectories. Starting and terminal states only")

    #parser.add_argument("--default", action="store_true", help="Default mode, ignore all configurations")
    parser.add_argument("--debug", action="store_true", help="Default mode, ignore all configurations")
    
    return parser.parse_args()

def main():
    """Performs training and logs info to wandb."""
    args = parse_args()

    with open(f"{args.folder_path}/training_config.json", "r") as f:
        configuration = json.load(f)
    # reading setting parameters from training config file
    algorithm, env_name = configuration["algorithm"], configuration["env_name"]

    searchspace_interface = create_searchspace(searchspace=configuration["searchspace"], dataset=configuration["dataset"])
    
    # create env (gym.Env)
    env = envs_dict[env_name.lower()](
        searchspace_api=searchspace_interface,
        scores=configuration["score_list"],
        target_device=args.target_device if args.use_custom_device else configuration["target_device"],
        weights=[configuration["performance_weight"], configuration["efficiency_weight"]] 
        )
    
    if env_name == "marcella-plus":
        # Oscar is the only env supporting a specific target device at test time
        if args.use_custom_device:
            env = envs_dict["oscar"](
                searchspace_api=searchspace_interface,
                scores=configuration["score_list"],
                target_device=args.target_device,
                weights=[configuration["task_weight"], configuration["hardware_weight"]],
                cutoff_percentile=100
            )
        
        env = TransitionsHistoryWrapper(env=env, history_len=configuration["history_len"])

    if args.render:
        env.unwrapped.rendering_test_mode = True
        env.unwrapped.render_mode = "human"
    
    model_name = \
        f"{args.folder_path}/{algorithm.upper()}_{env_name}_{to_scientific_notation(configuration['train_timesteps'])}.zip"

    # instantiate a testing suite
    policy = Policy(
        algo=algorithm,
        env=env,
        load_from_pathname=model_name
    )

    if args.verbose > 0: 
        print(f"Loading model from {model_name}")
        print(f"Testing {algorithm} on {env_name}")
    
    # saving up the episode return
    returns = np.zeros(args.test_episodes)
    initial_nets, episode_bests = [], []

    for ep in range(args.test_episodes):
        done = False
        obs, info = env.reset() # Reset environment to initial state
        # save initial network
        initial_net = env.current_net.architecture
        initial_nets.append(initial_net)
        
        # accumulating the episode return
        episode_return = 0
        while not done:  # Until the episode is over
            action = policy.predict(observation=obs, deterministic=True)  # deterministic is True while evaluating            
            obs, reward, terminated, truncated, info = env.step(action)	# Step the simulator to the next timestep
            done = terminated or truncated
            episode_return += reward

            if args.render:
                env.render()

        returns[ep] = episode_return
        # saving the best individual of the episode
        best_individual = env.current_net.architecture
        episode_bests.append(best_individual)
        if args.verbose:
            print(f"Network Designed: {best_individual}")
    
    print("Average episode return {:.4g}".format(returns.mean()))
    if args.best_ever:
        best_ever_net = max(episode_bests, key=lambda x: searchspace_interface.list_to_accuracy(x))
        print(f"Out of {args.test_episodes} test episodes, the best network designed is {best_ever_net}")
        print("With a validation accuracy of {:.5g}".format(searchspace_interface.architecture_to_accuracy(best_individual)))

    if args.save_trajectories:
        # saving all final networks obtained
        with open(f"{args.folder_path}/test_results.pkl", "wb") as results:
            pickle.dump(list(zip(initial_nets, episode_bests)), results)


if __name__=="__main__":
    main()

