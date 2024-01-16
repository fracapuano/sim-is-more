from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from copy import copy
import numpy as np
from scipy import signal
from scipy.stats import truncnorm
from typing import List, Text, Dict, Optional, Iterable
from numpy.typing import NDArray
from abc import ABC, abstractmethod


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
        env:gym.Env, 
        n_envs:int=1, 
        subprocess:bool=True, 
        wrappers_list:Optional[Iterable[gym.Wrapper]]=None)->VecEnv:
    """Simply builds an env using default configuration for the environment.

    Args: 
        n_envs (int, optional): Number of different envs to instantiate (using VecEnvs). Defaults to 1.
        subprocess (bool, optional): Whether or not to create multiple copies of the same environment using 
                                     suprocesses (so that multiple envs can run in parallel). When False, uses
                                     the usual DummyVecEnv. Defaults to True.
        device (str, optional): Device on which to run the environment. Defaults to "cpu".
    
    Returns:
        (VecEnv): Vectorized environment.
    """
    # define environment
    def make_env():
        new_env = copy(env)
        if wrappers_list is not None:
            """Wraps env with a provided wrapping objects."""
            for wrapper in wrappers_list:
                new_env = wrapper(new_env)
        
        """Wraps env with a Monitor object."""
        wrapped_env = Monitor(env=new_env)
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


class ProbabilityDistribution(ABC):
    """
    Class used to represent a probability distribution.
    This is used to sample latency values for blocks in the network.
    """

    @abstractmethod
    def sample(self) -> NDArray[np.float64]:
        """Sample k values from the distribution."""
        raise NotImplementedError("This method should be implemented in a subclass!")

class TruncatedNormalDistribution(ProbabilityDistribution):
    def __init__(self,
                 mean:float=0,
                 std:float=1,
                 lower_bound:float=1e-3,
                 upper_bound:float=1e3):
        
        # storing the distribution
        self.distribution = truncnorm
        # storing mean and std
        self.mean, self.std = mean, std

        # storing clipping ranges
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
    
    def sample(self, k:int=1):
        return truncnorm.rvs(
            a=self.lower_bound, 
            b=self.upper_bound, 
            loc=self.mean,
            scale=self.std,
            size=k)
