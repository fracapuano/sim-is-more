"""
Wrapper class for policy training and evaluation. 
This is particularly useful to test out different algorithms and training procedures.
"""
import torch
import os

import gymnasium as gym
from numpy.typing import NDArray
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import A2C, PPO
from sb3_contrib import TRPO
from src import get_project_root, to_scientific_notation
from custom_env import (
    create_epsilon_scheduler,
    create_learning_rate_scheduler
)
from gymnasium import spaces
from typing import List, Tuple, Union, Dict, Text

class Policy:
    def __init__(self,
                 algo:str=None,
                 env:gym.Env=None,
                 lr:float=3e-4,
                 device:str="cpu",
                 seed:int=None,
                 gamma:float=0.99,
                 load_from_pathname:str=None):

        self.seed = seed
        self.device = device
        self.env = env
        self.algo = algo.lower()
        self.gamma = gamma
        self.obs_is_dict = isinstance(self.env.observation_space, spaces.Dict)

        # either train from scratch (create_model) or from partially trained agent (load_model)
        if load_from_pathname is None:
            self.model = self.create_model(algo=self.algo, lr=lr)
            self.model_loaded = False
        else:
            self.model = self.load_model(self.algo, load_from_pathname)
            self.model_loaded = True

    def create_model(self, algo:str, lr:float)->BaseAlgorithm:
        """This function instanties a `BaseAlgorithm` object starting from a string identifying 
        the algorithm to be used and the value of the learning rate.
        
        Args: 
            algo (str): Algorithm to be used. One in ['ppo', 'a2c', 'trpo'].
            lr (float): Learning rate to be used for each algorithm. 
        
        Returns: 
            BaseAlgorithm: RL base algorithm (as defined in stable-baselines3).
        """
        if algo == 'ppo':
            model = PPO("MlpPolicy" if not self.obs_is_dict else "MultiInputPolicy", 
                        self.env, 
                        learning_rate=lr,
                        seed=self.seed, 
                        device=self.device, 
                        gamma=self.gamma)

        elif algo == 'a2c':
            model = A2C("MlpPolicy" if not self.obs_is_dict else "MultiInputPolicy", 
                        self.env, 
                        learning_rate=lr,
                        seed=self.seed, 
                        device=self.device, 
                        gamma=self.gamma)
        
        elif algo == 'trpo': 
            model = TRPO("MlpPolicy" if not self.obs_is_dict else "MultiInputPolicy", 
                        self.env, 
                        learning_rate=lr,
                        seed=self.seed, 
                        device=self.device,
                        gamma=self.gamma)
        else:
            raise ValueError(f"RL Algo not supported: {algo}. Supported algorithms ['ppo', 'a2c', 'trpo']")
        
        return model

    def load_model(self, algo: str, pathname: str) -> BaseAlgorithm:
        """
        Load a pre-trained RL model based on the provided algorithm and file path.

        Args:
            algo (str): Algorithm to be used. Must be one of ['ppo', 'trpo', 'sac'].
            pathname (str): File path to the pre-trained model.

        Returns:
            BaseAlgorithm: A pre-trained RL model loaded from the specified file path.

        Raises:
            ValueError: If the provided algorithm is not supported. Supported algorithms are ['ppo', 'trpo', 'a2c'].
        """
        if algo == 'ppo':
            model = PPO.load(pathname, 
                             env=self.env,
                             custom_objects = {'observation_space': self.env.observation_space, 
                                               'action_space': self.env.action_space}
                            )
        elif algo == 'trpo':
            model = TRPO.load(pathname, 
                             env=self.env,
                             custom_objects = {'observation_space': self.env.observation_space, 
                                               'action_space': self.env.action_space})
        elif algo == 'a2c':
            model = A2C.load(pathname)
        else:
            raise ValueError(f"RL Algo not supported: {algo}. Supported algorithms: ['trpo', 'ppo', 'a2c']")

        return model

    def train(self,
              timesteps:int=1000,
              n_eval_episodes:int=50,
              show_progressbar:bool=True,
              callback_list:List[BaseCallback]=None,
              best_model_save_path:str=str(get_project_root()) + "/models/",
              return_best_model:bool=True,
              verbose:int=0, 
              reset_timesteps:bool=True)->Union[Tuple[float, float], Tuple[float, float, Dict[str, str]]]:
        """
        Train a model using a custom list of callbacks, with optional best model selection and return.

        Args:
            timesteps (int): Number of training timesteps.
            n_eval_episodes (int): Number of episodes used for evaluation.
            show_progressbar (bool): Whether to display a progress bar during training.
            callback_list (List[BaseCallback]): List of custom callback objects.
            best_model_save_path (str): Path to save the best model.
            return_best_model (bool): Whether to return the best model and evaluation results.
            verbose (int): Verbosity level during training.
            reset_timesteps (bool): Whether to reset the number of timesteps before training.

        Returns:
            Union[Tuple[float, float], Tuple[float, float, Dict[str, str]]]:
                If return_best_model is False:
                    Tuple of the mean reward and standard deviation of rewards achieved during evaluation.
                If return_best_model is True:
                    Tuple of the mean reward, standard deviation of rewards, and additional information.
                    Additional information is a dictionary containing the key 'which_one' indicating whether
                    the final model or the best model was selected.

        Raises:
            ValueError: If the best_model.zip file does not exist or if there have been too few evaluations performed.
        """
        # get current verbosity level
        old_verbose = self.model.verbose
        # set verbose
        self.model.verbose = verbose
        # learns using custom callback
        self.model.learn(
            total_timesteps=timesteps, 
            callback=callback_list,
            reset_num_timesteps=reset_timesteps and self.model_loaded,
            progress_bar=show_progressbar
            )
        # reset actual verbosity level
        self.model.verbose = old_verbose
        
        if not return_best_model:
            return self.eval(n_eval_episodes)
        else:
            # Find best model among last and best
            reward_final, std_reward_final = self.eval(n_eval_episodes=n_eval_episodes)
            
            if not os.path.exists(os.path.join(best_model_save_path, "best_model.zip")):
                # best_model.zip is created during evaluation, so must exists if enough training steps 
                # have been provided
                print("best_model.zip hasn't been saved because too few evaluations have been performed.")
                raise ValueError("Check eval_freq and training timesteps used!")
            
            # access to best model
            best_model = self.load_model(self.algo, os.path.join(best_model_save_path, "best_model.zip"))
            # obtain mean and std of the return yielded by such a model
            reward_best, std_reward_best = evaluate_policy(best_model, 
                                                           self.env,
                                                           n_eval_episodes=n_eval_episodes)

            # comparing average cumulative reward obtained over n_episodes
            final_better_than_last = reward_final > reward_best
            # custom name for best env: <ALGO>_<ENV>_<TRAINTIMESTEPS>.zip
            model_name = f"{self.algo.upper()}_{self.env.get_attr('name')[0]}_{to_scientific_notation(timesteps)}.zip"

            if final_better_than_last:
                # delete old model - current final yields higher return
                os.remove(os.path.join(best_model_save_path, "best_model.zip"))
                # save current model as best one ever produced 
                self.model.save(os.path.join(best_model_save_path, model_name))
                best_mean_reward, best_std_reward = reward_final, std_reward_final
                which_one = 'final'
            else:
                os.rename(os.path.join(best_model_save_path, "best_model.zip"), 
                          os.path.join(best_model_save_path, model_name))
                # discaring the final model, best is better
                best_mean_reward, best_std_reward = reward_best, std_reward_best
                which_one = 'best'

            info = {'which_one': which_one}

            return best_mean_reward, best_std_reward, info

    def eval(self, n_eval_episodes:int=50, render:bool=False) -> Tuple[float, float]:
        """
        Evaluate the performance of the trained model.

        Args:
            n_eval_episodes (int): Number of episodes to use for evaluation.
            render (bool): Whether to render the environment during evaluation.

        Returns:
            Tuple[float, float]: Mean reward and standard deviation of rewards achieved during evaluation.
        """
        mean_reward, std_reward = evaluate_policy(self.model, self.model.get_env(), n_eval_episodes=n_eval_episodes, render=render)
        return mean_reward, std_reward

    def predict(self, observation:NDArray, deterministic:bool=False):
        """
        Exposes method to interface with model prediction. Predictions are related to 
        observations and are either deterministic or not according to `deterministic`.
        """
        # predict also returns last hidden state
        return self.model.predict(observation, deterministic=deterministic)[0]

    def state_dict(self):
        return self.model.policy.state_dict()

    def save_state_dict(self, pathname:str):
        torch.save(self.state_dict(), pathname)

    def load_state_dict(self, path_or_state_dict:Union[str, dict]):
        if isinstance(path_or_state_dict, str):
            self.model.policy.load_state_dict(torch.load(
                path_or_state_dict, map_location=torch.device(self.device)), strict=True)
        else:
            self.model.policy.load_state_dict(path_or_state_dict, strict=True)

    def save_full_state(self, pathname:str):
        """Saves the model at `pathname`"""
        self.model.save(pathname)

    def set_epsilon_scheduler(self, kind:Text="exp"):
        """Sets the scheduler for the PPO algorithm"""
        if self.algo.lower() != "ppo": 
            raise ValueError("Epsilon-scheduling currently only supported for the PPO algorithm.")
        # overwrites the clip range attribute within self.model
        self.model.clip_range = create_epsilon_scheduler(kind=kind)
    
    def set_lr_scheduler(self, kind:Text="exp"):
        """Sets the learning rate scheduler"""
        # overwrites the clip range attribute within self.model
        self.model.learning_rate = create_learning_rate_scheduler(kind=kind)
    
    @staticmethod
    def load_full_state():
        raise ValueError('Use the constructor with load_from_pathname parameter')
        