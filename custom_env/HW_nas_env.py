from gym import spaces
import numpy as np
from commons import BaseInterface
from .nas_env import NASEnv
from .utils import NASIndividual
from typing import Iterable, Text

class HW_NASEnv(NASEnv): 
    """
    gym.Env for Hardware-aware pure-RL-based NAS. Architectures are evaluated using training-free metrics 
    only as well as specific hardware
    """
    def __init__(self, 
                 searchspace_api:BaseInterface,
                 scores:Iterable[Text]=["naswot_score", "logsynflow_score", "skip_score"], 
                 n_mods:int=1,
                 max_timesteps:int=50, 
                 target_device:Text="raspi4", 
                 weights:Iterable[float]=[0.6, 0.4]):
        
        self.target_device = target_device
        self.weights = np.array(weights) if not isinstance(weights, np.ndarray) else weights

        super().__init__(searchspace_api=searchspace_api, 
                         scores=scores, 
                         n_mods=n_mods, 
                         max_timesteps=max_timesteps)
    
    @property
    def name(self): 
        return "hw-nasenv"

    def fitness_function(self, individual:NASIndividual)->NASIndividual: 
        """
        Directly overwrites the fitness attribute for a given individual.

        Args: 
            individual (NASIndividual): Individual to score.

        Returns:
            NASIndividual: Individual, with fitness field.
        """
        if individual.fitness is None:  # None at initialization only
            scores = np.array([
                self.normalize_score(
                    score_value=self.searchspace.list_to_score(input_list=individual.architecture, 
                                                               score=score), 
                    score_name=score
                )
                for score in self.score_names
            ])
            hardware_performance = np.array([
                self.normalize_score(
                    score_value=self.searchspace.list_to_score(input_list=individual.architecture, 
                                                               score=f"{self.target_device}_{metric}"),
                    score_name=f"{self.target_device}_{metric}"
                )
                for metric in ["latency"]  # change here to add more hardware aware metrics
            ])
            # individual fitness is a convex combination of multiple scores
            network_score = (np.ones_like(scores) / len(scores)) @ scores
            network_hardware_performance =  (np.ones_like(hardware_performance) / len(hardware_performance)) @ hardware_performance
            
            # in the hardware aware contest performance is in a direct tradeoff with hardware performance
            individual._fitness = np.array([network_score, -network_hardware_performance]) @ self.weights
        
        return individual

    def _get_info(self)->dict: 
        """Return the info dictionary."""
        info_dict = {
            "current_network": self.current_net.architecture,
            "test_accuracy": self.searchspace.list_to_accuracy(self.current_net.architecture),
            "training_free_score": self.current_net.fitness,
            "timestep": self.timestep_counter,
            "latency": self.searchspace.list_to_score(input_list=self.current_net.architecture, 
                                                     score=f"{self.target_device}_latency"),
        }

        return info_dict