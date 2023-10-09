from gym import spaces
import numpy as np
from itertools import chain
from numpy.typing import NDArray
from .freerenas_env import (
    FreeRENASEnv
)
from commons import (
    BaseInterface,
    BaseIndividual,
    Individual,
)
from typing import Iterable, Text

class HW_FreeRENASEnv(FreeRENASEnv):
    """
    gym.Env for the Hardware aware FreeRENAS algorithm, i.e. a RENAS-like algorithm using 
    Training-Free metrics and oriented to solve the NAS problem for a specific hardware.
    In this environment we present an integration of DRL with Genetic Computations.

    The RENAS paper can be found at: https://arxiv.org/abs/1808.00193
    The FreeREA paper can be found at: https://arxiv.org/pdf/2207.05135.pdf
    The hardware aware metrics can be found at: https://openreview.net/pdf?id=_0kaDkv3dVf
    """
    def __init__(self, 
                 searchspace_api:BaseInterface,
                 genetics_dict:dict={}, 
                 scores:Iterable[Text]=["naswot_score", "logsynflow_score", "skip_score"], 
                 target_device:Text="raspi4", 
                 weights:Iterable[float]=[0.6, 0.4]):
        
        self.target_device = target_device
        self.weights = np.array(weights) if not isinstance(weights, np.ndarray) else weights

        # init super method
        super().__init__(searchspace_api=searchspace_api, 
                         genetics_dict=genetics_dict, 
                         scores=scores)
        
    @property
    def name(self)->Text:
        return "hwfreerenas"
    
    def fitness_function(self, individual:BaseIndividual)->Individual: 
        """
        Directly overwrites the fitness attribute for a given individual.
        In RENAS, the fitness score coincides with the test accuracy.

        Args: 
            individual (Individual): Individual to score.

        Returns:
            Individual: Individual, with fitness field.
        """
        if individual.fitness is None:  # None at initialization 
            scores = np.array([
                self.normalize_score(
                    score_value=self.searchspace.list_to_score(input_list=individual.genotype, score=score), 
                    score_name=score
                )
                for score in self.score_names
            ])
            hardware_performance = np.array([
                self.normalize_score(
                    score_value=self.searchspace.list_to_score(input_list=individual.genotype, 
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
