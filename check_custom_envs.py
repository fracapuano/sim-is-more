from stable_baselines3.common.env_checker import check_env
from custom_env import envs_dict
from src.interfaces import NATS_Interface
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='nasenv', help='Environment name')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    searchspace_interface = NATS_Interface()
    env = envs_dict[args.env](
        searchspace_api=searchspace_interface
    )
    
    check_env(env, warn=True)

if __name__ == '__main__':
    main()