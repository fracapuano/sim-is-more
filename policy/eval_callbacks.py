"""Custom callbacks to be used during training to record the learnign process."""
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import wandb
import numpy as np
import warnings  # suppressing the warnings from gymnasium due to stable_baselines3
from typing import Text, List

class ScoresEvolutionTracker:
    def __init__(self, 
                 metrics:List[Text]=["training_free_score", "current_net_latency", "current_net_latency_percentile", "test_accuracy"],
                 number_of_envs:int=1,
                 episode_length:int=50,
                 number_of_episodes:int=50):
        
        # Which metrics to log
        self.metrics = metrics
        # Number of environments
        self.number_of_envs = number_of_envs
        # How many timesteps per episode
        self.episode_length = episode_length
        # How many episodes to evaluate
        self.number_of_episodes = number_of_episodes
        # How many episodes are associated with each environment
        self.episodes_per_env = np.array(
            [(self.number_of_episodes + i) // self.number_of_envs for i in range(self.number_of_envs)], dtype="int"
        )
        # Keeping track of how many episodes have been terminated
        self.number_of_terminated_episodes = 0

        self.reset_stats()

    def reset_stats(self):
        """Flushing values previously tracked."""
        self.number_of_terminated_episodes = 0 
        for metric in self.metrics:
            setattr(self, f"{metric}_buffer", np.zeros((self.number_of_episodes, self.episode_length)))
        

    def logmetrics_callback(self, locals_dict:dict, globals_dict:dict):
        """
        Uses access to locals() to elaborate information. 
        Intended to be used inside of stable_baselines3 `evaluate_policy`
        """
        info_dict = locals_dict["info"]
        timestep_counter = info_dict["timestep"]
        
        episode_counter = locals_dict["episode_counts"]
        episode_counter_targets = locals_dict["episode_count_targets"]
        env_index = locals_dict["i"]
        # This routes the metric to the correct row in the buffer
        episode_index = episode_counter_targets[:env_index].sum() + episode_counter[env_index]
        
        for metric in self.metrics:
            buffer = getattr(self, f"{metric}_buffer")
            buffer[episode_index, timestep_counter] = (info_dict.get(metric, -1))
        
        self.number_of_terminated_episodes += 1 if info_dict["is_terminated"] else 0


class PeriodicEvalCallback(BaseCallback): 
    """Custom callback inheriting from `BaseCallback`.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug.

    Performs various actions when triggered (intended to be a child of EventCallback): 
        1. Evaluates current policy (for n_eval_episodes)
        2. Updates a current best_policy variable
        3. Logs stuff on wandb. More details on what is logged in :meth:_on_step.
    """
    def __init__(
            self, 
            env:VecEnv, 
            render:bool=False,
            verbose:int=0,
            n_eval_episodes:int=50,
            best_model_path:Text="models/"):
        """Init function defines callback context."""
        super().__init__(verbose)

        """Suppressing this warning as it is raised by stable_baselines3 `env_method` function."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.episode_duration = env.env_method("get_max_timesteps")[0]

        self._envs = env
        self.number_of_envs = self._envs.num_envs
        self.render = render
        self.n_eval_episodes = n_eval_episodes
        self.episodes_tracker = ScoresEvolutionTracker(
            episode_length=self.episode_duration, 
            number_of_envs=self.number_of_envs,
            number_of_episodes=self.n_eval_episodes
        )
        # current best model and best model's return in test trajectories
        self.best_model_path = best_model_path
        self.best_model = None
        self.best_model_mean_reward = -float("inf")
        self.bests_found = 0

        # reset environments
        self._envs.reset()

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `_env.step()`.
        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.
        :return: (bool) If the callback returns False, training is aborted early.
        """
        # flush past statistics
        self.episodes_tracker.reset_stats()
        # obtain mean and std of cumulative reward over n_eval_episodes
        mean_cum_reward, std_cum_reward = evaluate_policy(
            self.model,
            self.model.get_env(),
            callback=self.episodes_tracker.logmetrics_callback,
            n_eval_episodes=self.n_eval_episodes,
            render=self.render)
        
        for metric in self.episodes_tracker.metrics:
            tracked_values = getattr(self.episodes_tracker, f"{metric}_buffer")[:, 1:].mean(axis=0)
            # using wandb to log the evolution of the metric over training
            data = [[x, y] for (x, y) in zip(np.arange(1, self.episode_duration), tracked_values)]
            table = wandb.Table(data=data, columns = ["timestep", metric])
            wandb.log(
                {f"{metric}-evolution-over-training": \
                    wandb.plot.line(table, 
                                    "timestep", 
                                    metric,
                                    title=f"Average Evolution of {metric}")
                }
            )
        
        # How many episodes were terminated during the evaluation
        wandb.log({"TerminationRate": self.episodes_tracker.number_of_terminated_episodes / self.n_eval_episodes})

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

