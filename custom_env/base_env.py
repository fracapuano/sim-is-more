import gym
import numpy as np
from abc import abstractproperty
from src import Base_Interface
from typing import Text

class BaseNASEnv(gym.Env): 
    """Base gym.Env class for NAS algorithms."""
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 5
    }
    def __init__(self, searchspace_interface:Base_Interface):
        """Init function. Here characteristics of the base nas env are defined"""
        super().__init__()

        # Define the search space
        self.searchspace = searchspace_interface
        # Define observation space
        self.observation_space=None
        # Define action space
        self.action_space=None

    @abstractproperty
    def name(self)->Text:
        raise NotImplementedError("Must override get_observation in child classes!")
    
    def get_observation(self):
        """Returns the present observation for the environment"""
        raise NotImplementedError("Must override get_observation in child classes!")
    
    def step(self, action:np.ndarray):
        raise NotImplementedError("Step method must be ovveridden in child classes!")
    
    def reset(self, seed:int=None):
        raise NotImplementedError("Reset method must be ovveridden in child classes!")

