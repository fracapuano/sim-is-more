"""Custom callbacks to be used during training to record the learnign process."""
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from stable_baselines3.common.vec_env import VecEnv

class ChangeDevice_Callback(BaseCallback): 
    """Custom callback inheriting from `BaseCallback`.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug.

    Performs various actions when triggered (intended to be a child of EventCallback): 
        1. Evaluates current policy (for n_eval_episodes)
        2. Updates a current best_policy variable
        3. Logs stuff on wandb. More details on what is logged in :meth:_on_step.
    """
    def __init__(
            self,
            verbose:int=0):
        """Init function defines callback context."""
        super().__init__(verbose)
        
        self.devices_history = []

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `_env.step()`.
        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.
        :return: (bool) If the callback returns False, training is aborted early.
        """
        # accessing the list of possible devices per env considered - all envs have the same devices, hence [0]
        self.available_devices = list(
            self.training_env.env_method("get_devices")[0].keys()
        )
        # storing the current hardware used for training
        current_device = self.model.env.env_method("get_target_device") if isinstance(self.model.env, VecEnv) \
                           else self.model.env.target_device
        # stores the target hardware the model has been currently training on
        self.devices_history.append(current_device)
        # sampling the new device here ensures all envs are trained on the same hardware at any given times
        new_device = np.random.choice(self.available_devices)

        if isinstance(self.model.env, VecEnv):
            # switching to device freeze False to change device at next episode reset
            self.model.env.env_method("switch_device_freeze")
            # making sure all envs have the same device
            self.model.env.env_method("set_next_device", new_device)

        else: # env is a single env, not even DummyVec
            self.model.env.device_freeze = False

        return True
    
    def get_devices_history(self):
        """Returns the full history of hardware devices"""
        return self.devices_history

