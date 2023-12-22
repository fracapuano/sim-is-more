"""Tests env with a random policy."""
import argparse
from custom_env import (
    envs_dict
)
from src import NATS_Interface
from pprint import pprint
import time
import numpy as np

def parse_args()->object: 
    """Args function. 
    Returns:
        (object): args parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", type=str, help="Dataset to be considered. One in ['cifar10', 'cifar100', 'ImageNet16-120'].s")
    parser.add_argument("--env", default="oscar", type=str, help="Environment to test random policy on.")
    parser.add_argument("--verbose", action="store_true", help="Whether or not to print out the info dictionary.")
    parser.add_argument("--n-episodes", default=30, type=int, help="Number of episodes to use.")
    
    return parser.parse_args()

def main(): 
    """Tests random policy on envs_dict[args.env]"""
    args = parse_args()    
    
    searchspace_interface = NATS_Interface(dataset=args.dataset)
    env = envs_dict[args.env.lower()](searchspace_api=searchspace_interface)
    
    print('State space:', env.observation_space)
    print('Action space:', env.action_space)

    durations = []
    for _ in range(args.n_episodes): 
        done = False
        obs, info = env.reset() # Reset environment to initial state
            
        start = time.time()
        while not done:  # Until the episode is over
            action = env.action_space.sample()	# Sample random action            
            observation, reward, terminated, truncated, info = env.step(action)	# Step the simulator to the next timestep
            
            if args.verbose: 
                pprint(info)

            done = terminated or truncated
            
        end = time.time()
        print(f"Episode {_} duration: "+"{:.4f}".format(end - start))
        durations.append(end - start)

    print("Average Episode duration: {:.4g} (s)".format(np.mean(durations)))

if __name__=="__main__": 
    main()

