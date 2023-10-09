from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from .base_env import BaseNASEnv
from copy import deepcopy as copy
import numpy as np
from scipy import signal

from typing import List, Text, Dict

class NASIndividual:
    """
    Class used to represent architectures and store information relative to their performance.
    """
    def __init__(self, 
                 architecture:List[Text],
                 architecture_string_to_idx:Dict[Text, int], 
                 index:int):
        
        self._architecture = architecture
        # initialize the individual fitness to None.
        self._fitness = None
        # the index allows retrieving the element performance metrics
        self.index = index
        # this maps architecture STRINGS to the corresponding index in the searchspace
        self.architecture_string_to_idx = architecture_string_to_idx
    
    @property
    def architecture(self):
        return self._architecture
    
    @property
    def fitness(self):
        return self._fitness
    
    def update_idx(self):
        self.index = self.architecture_string_to_idx["/".join(self._architecture)]

    def update_architecture(self, new_architecture:List[Text]): 
        """Update current architecture with new one."""
        self._architecture = new_architecture
        self.update_idx()


def build_vec_env(
        env_:BaseNASEnv, 
        n_envs:int=1, 
        subprocess:bool=True)->VecEnv:
    """Simply builds an env using default configuration for the environment.

    Args: 
        n_envs (int, optional): Number of different envs to instantiate (using VecEnvs). Defaults to 1.
        subprocess (bool, optional): Whether or not to create multiple copies of the same environment using 
                                     suprocesses (so that multiple envs can run in parallel). When False, uses
                                     the usual DummyVecEnv. Defaults to True.
        device (str, optional): Device on which to run the environment. Defaults to "cpu".
    
    Returns: 
    """
    # define environment (on top of xi)
    def make_env():
        env = copy(env_)
        """Wraps env with a Monitor object."""
        wrapped_env = Monitor(env=env)
        return wrapped_env

    # vectorized environment, wrapped with Monitor
    if subprocess:
        envs = SubprocVecEnv([make_env for _ in range(n_envs)])
    else: 
        envs = DummyVecEnv([make_env for _ in range(n_envs)])

    return envs

def create_epsilon_scheduler(
        start_epsilon:float=0.3, 
        end_epsilon:float=0.1,
        spike_every:float=0.05,
        kind:Text="exp"
        ):
    if kind.lower()=="exp":
        # starting_point = (1, start_epsilon); ending_point = (0, end_epsilon)
        a, b = end_epsilon, start_epsilon/end_epsilon
        # Create the scheduler function, which depends on the fraction of remaining timesteps here to be observed
        scheduler = lambda percent_training_left: a * (b ** percent_training_left)
    
    elif kind.lower()=="sawtooth":
        # starting_point = (1, start_epsilon); ending_point = (0, end_epsilon)
        a, b = end_epsilon, start_epsilon/end_epsilon
        # this creates the sawtooth list of values to use as values for epsilon
        n_spikes = int(1/spike_every)
        # Create the scheduler function, which depends on the fraction of remaining timesteps here to be observed
        def scheduler(percent_training_left):
            # checking if current portion of training is a multiple of a spike point
            exponential_peak = a * (b ** percent_training_left)
            sawtooth_peak = 0.1 * signal.sawtooth(2 * np.pi * n_spikes * percent_training_left).item()
            # this returns either the sawtooth peak or the exponential value
            return max(exponential_peak, exponential_peak + sawtooth_peak)
    
    elif kind.lower()=="sine":
         # starting_point = (1, start_epsilon); ending_point = (0, end_epsilon)
        a, b = end_epsilon, start_epsilon/end_epsilon
        # this creates the sawtooth list of values to use as values for epsilon
        n_spikes = int(1/spike_every)
        # Create the scheduler function, which depends on the fraction of remaining timesteps here to be observed
        def scheduler(percent_training_left):
            # checking if current portion of training is a multiple of a spike point
            exponential_peak = a * (b ** percent_training_left)
            sine_peak = 0.1 * np.sin(2 * np.pi * n_spikes * percent_training_left).item()
            # this returns either the sawtooth peak or the exponential value
            return max(exponential_peak, exponential_peak + sine_peak)
    else:
        raise ValueError(f"Scheduler type: {kind} not implemented! Implemented schedulers: ['exp', 'sine', 'sawtooth']")

    return scheduler

def create_learning_rate_scheduler(
        max_lr:float=3e-3, 
        min_lr:float=3e-4,
        spike_every:float=0.05,
        kind:Text="exp"
        ):
    
    if kind.lower()=="exp":
        # starting_point = (1, start_epsilon); ending_point = (0, end_epsilon)
        a, b = min_lr, max_lr/min_lr
        # Create the scheduler function, which depends on the fraction of remaining timesteps here to be observed
        scheduler = lambda percent_training_left: a * (b ** percent_training_left)
    
    elif kind.lower()=="sawtooth":
        # starting_point = (1, start_epsilon); ending_point = (0, end_epsilon)
        a, b = min_lr, max_lr/min_lr
        # this creates the sawtooth list of values to use as values for epsilon
        n_spikes = int(1/spike_every)
        # Create the scheduler function, which depends on the fraction of remaining timesteps here to be observed
        def scheduler(percent_training_left):
            # checking if current portion of training is a multiple of a spike point
            exponential_peak = a * (b ** percent_training_left)
            sawtooth_peak = 0.1 * signal.sawtooth(2 * np.pi * n_spikes * percent_training_left).item()
            # this returns either the sawtooth peak or the exponential value
            return max(exponential_peak, exponential_peak + sawtooth_peak)
    
    elif kind.lower()=="sine":
         # starting_point = (1, start_epsilon); ending_point = (0, end_epsilon)
        a, b = min_lr, max_lr/min_lr
        # this creates the sawtooth list of values to use as values for epsilon
        n_spikes = int(1/spike_every)
        # Create the scheduler function, which depends on the fraction of remaining timesteps here to be observed
        def scheduler(percent_training_left):
            # checking if current portion of training is a multiple of a spike point
            exponential_peak = a * (b ** percent_training_left)
            sine_peak = 0.1 * np.sin(2 * np.pi * n_spikes * percent_training_left).item()
            # this returns either the sawtooth peak or the exponential value
            return max(exponential_peak, exponential_peak + sine_peak)
    else:
        raise ValueError(f"Scheduler type: {kind} not implemented! Implemented schedulers: ['exp', 'sine', 'sawtooth']")

    return scheduler