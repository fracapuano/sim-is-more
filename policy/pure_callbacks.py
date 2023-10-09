"""Custom callbacks to be used during training to record the learnign process."""
import gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import wandb
import numpy as np

from typing import Tuple, Text, List

class EpisodeStatsAgg:
    def __init__(self, 
                 metrics:List[Text]=["test_accuracy", "training_free_score", "latency"], 
                 episode_length:int=50):
        
        self.metrics = metrics
        self.episode_length = episode_length
        self.reset_stats()

    @property
    def stat_keys(self):
        return [1] + [int(perc * self.episode_length) for perc in [0.33, 0.66]] + [self.episode_length-1]

    def logmetrics_callback(self, locals_dict:dict, globals_dict:dict):
        """
        Uses access to locals() to elaborate information. 
        Intended to be used inside of stable_baselines3 `evaluate_policy`
        """
        info_dict = locals_dict["info"]
        # append metrics of interest, at the frequency considered
        if all([info_dict["timestep"] in getattr(self, metric).keys() for metric in self.metrics]):
            # for code readibility only
            timestep_counter = info_dict["timestep"]
            # logging the measured metrics from the locals dict
            for metric in self.metrics:
                # retrieving the metric dictionary -> accessing the timestep of interest ->
                # -> loggin the metric of interest, if in info_dict (depends on env), else logs -1 (unrealistic value)
                getattr(self, metric)[timestep_counter].append(info_dict.get(metric, -1))
            
    def reset_stats(self):
        # storing information on the various metrics at different timesteps thresholds - creating
        # one dictionary per metric tracked to allow for modularity
        for metric in self.metrics:
            setattr(self, metric, {key: [] for key in self.stat_keys})

class PureRL_PolicyCallback(BaseCallback): 
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
            best_model_path:Text="models/"):
        """Init function defines callback context."""
        super().__init__(verbose)

        episode_duration = env.max_timesteps if not isinstance(env, VecEnv) else env.get_attr("max_timesteps")[0] 

        self._envs = env
        self.render = render
        self.n_eval_episodes = n_eval_episodes
        self.EpisodeStatAggregator = EpisodeStatsAgg(episode_length=episode_duration)
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
        self.EpisodeStatAggregator.reset_stats()
        # obtain mean and std of cumulative reward over n_eval_episodes
        mean_cum_reward, std_cum_reward = evaluate_policy(
            self.model,
            self.model.get_env(),
            callback=self.EpisodeStatAggregator.logmetrics_callback,
            n_eval_episodes=self.n_eval_episodes,
            render=self.render)
        
        mean_or_item = lambda arr: np.mean(arr) if len(arr)!=1 else arr[0]

        # averaging the measurements collected in EpisodeStatAggregator over the different
        # test episodes to reduce measurement bias
        wandb.log({
            f"{metric}_{key}_timesteps": mean_or_item(getattr(self.EpisodeStatAggregator, metric)[key])
            for metric in self.EpisodeStatAggregator.metrics
            for key in self.EpisodeStatAggregator.stat_keys
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
