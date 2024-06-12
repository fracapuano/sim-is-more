from .utils import NASIndividual
import numpy as np
from typing import Text, Iterable, Literal
from numpy.typing import NDArray
from src.interfaces.base_interface import Base_Interface

class Rewardv0:
    """
    Reward function for the HW-NAS environment.
    Background material available at: https://github.com/fracapuano/OSCAR/issues/17#issuecomment-2137170731
    """
    def __init__(self,
                 searchspace:Base_Interface,
                 score_names:Iterable[Text]=["normalized_validation_accuracy", "normalized_latency"],
                 weights:NDArray=np.array([0.5, 0.5]), 
                ):
        
        self.searchspace = searchspace
        self.score_names = score_names
        self.weights = weights

        self.accuracy_stats = self.searchspace.get_accuracy_stats()
        self.latency_stats = self.searchspace.get_latency_stats()
        
    def fitness_function(self, individual:NASIndividual)->NASIndividual: 
        """
        Directly overwrites the fitness attribute for a given individual.

        Args: 
            individual (NASIndividual): Individual to score.

        Returns:
            NASIndividual: Individual, with fitness field.
        """
        if individual.fitness is None:  # None at initialization time
            normalized_accuracy_score = self.get_normalized_accuracy(individual)
            normalized_latency_score =  self.get_normalized_latency(individual)
            
            # in the hardware aware contest performance is in a direct tradeoff with hardware performance
            individual._fitness = self.combine_scores(normalized_accuracy_score, normalized_latency_score).item()
        
        return individual

    def get_normalized_accuracy(self, individual:NASIndividual, norm_style:Literal["minmax", "zscale"]="minmax")->float:
        """
        Normalize the accuracy of the individual.
        
        Args:
            individual (NASIndividual): The individual to normalize.
            norm_style (Literal["minmax", "zscale"], optional): The normalization style. Defaults to "minmax".
        
        Raises:
            ValueError: If an invalid normalization style is provided.
            
        Returns:
            float: The normalized accuracy.
        """
        accuracy = self.searchspace.architecture_to_accuracy(individual.architecture)
        if norm_style == "minmax":
            # min-max normalize the accuracy
            return (accuracy - self.accuracy_stats["min"]) / (self.accuracy_stats["max"] - self.accuracy_stats["min"])
        elif norm_style == "zscale":
            # z-score normalize the accuracy
            return (accuracy - self.accuracy_stats["mean"]) / self.accuracy_stats["std"]
        else:
            raise ValueError(f"Invalid normalization style: {norm_style}. Accepted values are 'minmax' and 'zscale'")
        
    def get_normalized_latency(self, individual:NASIndividual)->float:
        """
        Normalize the latency of the individual.
        
        Args:
            individual (NASIndividual): The individual to normalize.
        
        Returns:
            float: The normalized latency.
        """
        latency = self.searchspace.architecture_to_score(\
            individual.architecture, score=f"{self.searchspace.target_device}_latency")\
        
        # min-max inverse-normalize the latency
        return 1 - ((self.latency_stats["max"] - latency) / (self.latency_stats["max"] - self.latency_stats["min"]))

    def combine_scores(self, performance_scores:float, efficiency_score:float)->float:
        """
        Combine the performance and efficiency scores into a single score.

        Args:
            performance_scores (float): The performance scores.
            efficiency_score (float): The efficiency scores.

        Returns:
            float: The combined scores.
        """
        if not (isinstance(performance_scores, float) and isinstance(efficiency_score, float)):
            raise ValueError(f"""
                The input scores must both be float! 
                Provided input: performance_score {type(performance_scores)}, efficiency_score {type(efficiency_score)}
            """)
        
        return (np.array([performance_scores, efficiency_score]) @ self.weights).item()  # returning a float

    def get_reward(self, individual:NASIndividual)->float:
        """
        Compute the reward associated to the modification operation.
        Here, the reward is the fitness of the newly generated individual
        """
        # here the reward is the fitness of the individual
        return individual.fitness

