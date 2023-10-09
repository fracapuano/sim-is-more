"""Custom callbacks to be used during training to record the learnign process."""
import gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import wandb
import numpy as np

from typing import Tuple

class StatsAgg:
    def __init__(self): 
        self.reset_stats()

    def logmetrics_callback(self, locals_dict:dict, globals_dict:dict):
        """
        Uses access to locals() to elaborate information. 
        Intended to be used inside of stable_baselines3 `evaluate_policy`
        """
        info_dict = locals_dict["info"]
        # append metrics of interest
        self.mean_fitness.append(info_dict["mean_fitness"])
        self.std_fitness.append(info_dict["std_fitness"])

        if info_dict["timestep"] == 1: 
            self.initial_max_fitness.append(info_dict["current_max_fitness"])
        
        if info_dict["timestep"] == 10: 
            self.tengens_max_fitness.append(info_dict["current_max_fitness"])

        if info_dict["timestep"] == 30: 
            self.thirthygens_max_fitness.append(info_dict["current_max_fitness"])

        if locals_dict["done"]:
            self.terminal_max_fitness.append(info_dict["current_max_fitness"])

    def reset_stats(self):
        self.mean_fitness, self.std_fitness = [], []
        self.initial_max_fitness, self.terminal_max_fitness = [], []
        self.tengens_max_fitness, self.thirthygens_max_fitness = [], []
        

class Hybrid_PolicyCallback(BaseCallback): 
    """Custom callback inheriting from `BaseCallback`.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug.

    Performs various actions when triggered (intended to be a child of EventCallback): 
        1. Evaluates current policy (for n_eval_episodes)
        2. Updates a current best_policy variable
        3. Logs stuff on wandb. More details on what is logged in :meth:_on_step.
    """
    def __init__(
            self, 
            env:Tuple[gym.Env, VecEnv], 
            render:bool=False,
            verbose:int=0,
            n_eval_episodes:int=50, 
            best_model_path:str="models/"):
        """Init function defines callback context."""
        super().__init__(verbose)

        self._envs = env
        self.render = render
        self.n_eval_episodes = n_eval_episodes
        self.EvaluationStats = StatsAgg()
        # resets environment
        self._envs.reset()
        # current best model and best model's return in test trajectories
        self.best_model_path = best_model_path
        self.best_model = None
        self.best_model_mean_reward = -float("inf")
        self.bests_found = 0

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `_env.step()`.
        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.
        :return: (bool) If the callback returns False, training is aborted early.
        """
        # flush past statistics
        self.EvaluationStats.reset_stats()
        # obtain mean and std of cumulative reward over n_eval_episodes
        mean_cum_reward, std_cum_reward = evaluate_policy(
            self.model,
            self.model.get_env(),
            callback=self.EvaluationStats.logmetrics_callback,
            n_eval_episodes=self.n_eval_episodes,
            render=self.render)
        
        # reshaping populated metrics
        mean_fitness_data = np.array(self.EvaluationStats.mean_fitness).\
                            reshape((self.n_eval_episodes, -1))
        std_fitness_data = np.array(self.EvaluationStats.std_fitness).\
                            reshape((self.n_eval_episodes, -1))
        
        mean_initial_fitness, mean_10its_fitness, mean_30its_fitness, mean_final_fitness = \
            [np.mean(arr) for arr in [self.EvaluationStats.initial_max_fitness, 
                                     self.EvaluationStats.tengens_max_fitness, 
                                     self.EvaluationStats.thirthygens_max_fitness, 
                                     self.EvaluationStats.terminal_max_fitness]]

        wandb.log({
            "Average Min(Mean Fitness in Population)": mean_fitness_data.mean(axis=0).min(),
            "Average Max(Mean Fitness in Population)": mean_fitness_data.mean(axis=0).max(),
            "Average Min(Std Fitness)": std_fitness_data.mean(axis=0).min(),
            "Average Max(Std Fitness)": std_fitness_data.mean(axis=0).max(), 
            "Average Initial Max-Fitness in Population": mean_initial_fitness,
            "Average 10-Its Max-Fitness in Population": mean_10its_fitness, 
            "Average 30-Its Max-Fitness in Population": mean_30its_fitness,
            "Average Final Fitness in Population": mean_final_fitness
        })
        
        # checks if this model is better than current best. If so, update current best
        if mean_cum_reward >= self.best_model_mean_reward:
            self.best_model = self.model
            self.best_model_mean_reward = mean_cum_reward
            # save best model
            self.best_model.save(path=f"{self.best_model_path}/best_model.zip")
            self.bests_found += 1

        wandb.log({
             "Mean Cumulative Reward": mean_cum_reward,
             "Std of Cumulative Reward": std_cum_reward,
        })
        
        return True
    
    def get_best_model(self, return_reward:bool=True): 
        if return_reward:
            return self.best_model, self.best_model_mean_reward
        else: 
            return self.best_model
