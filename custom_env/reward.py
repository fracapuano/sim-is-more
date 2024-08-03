from .nas_env import NASEnv
from .utils import NASIndividual
import numpy as np
from typing import Literal
from numpy.typing import NDArray
from src.interfaces.base_interface import Base_Interface
from collections import namedtuple

class Rewardv0:
    """
    Reward function for the HW-NAS environment.
    Background material available at: https://github.com/fracapuano/OSCAR/issues/17#issuecomment-2137170731
    """
    performance_score = "normalized_validation_accuracy"
    efficiency_score = "normalized_latency"

    reward_version: str = "rewardv0"
    def __init__(self,
                 searchspace:Base_Interface,
                 weights:NDArray=np.array([0.5, 0.5]), 
                ):
        
        self.searchspace = searchspace
        self.weights = weights

        self.accuracy_stats = self.searchspace.get_accuracy_stats()
    
    @property
    def latency_stats(self):
        if not hasattr(self, "_env"):
            return self.searchspace.get_latency_stats()
        elif hasattr(self, "_env") and getattr(self, "_env").name == "marcella-plus":
            return self._env.latency_stats

    def get_performance_score(self, individual:NASIndividual)->float:
        """
        Return the performance score of the individual.
        """
        return self.get_normalized_accuracy(individual)
    
    def get_efficiency_score(self, individual:NASIndividual)->float:
        """
        Return the efficiency score of the individual.
        """
        return self.get_normalized_latency(individual)

    def fitness_function(self, individual:NASIndividual)->NASIndividual: 
        """
        Directly overwrites the fitness attribute for a given individual.

        Args: 
            individual (NASIndividual): Individual to score.

        Returns:
            NASIndividual: Individual, with fitness field.
        """
        if individual.fitness is None:  # None at initialization
            normalized_performance_score = self.get_performance_score(individual)
            normalized_efficiency_score =  self.get_efficiency_score(individual)
            
            # in the hardware aware contest performance is in a direct tradeoff with hardware performance
            individual._fitness = self.combine_scores(normalized_performance_score, normalized_efficiency_score)
        
        return individual

    def get_individual_scores(self, individual:NASIndividual)->dict:
        """
        Return the performance and efficiency scores of the individual.
        """
        return {
            "reward_performance_score": self.get_performance_score(individual),
            "reward_efficiency_score": self.get_efficiency_score(individual)
        }

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
        accuracy = self.searchspace.list_to_accuracy(individual.architecture)
        if norm_style == "minmax":
            # min-max normalize the accuracy
            return (accuracy - self.accuracy_stats["min"]) / (self.accuracy_stats["max"] - self.accuracy_stats["min"])
        elif norm_style == "zscale":
            # z-score normalize the accuracy
            return (accuracy - self.accuracy_stats["mean"]) / self.accuracy_stats["std"]
        else:
            raise ValueError(f"Invalid normalization style: {norm_style}. Accepted values are 'minmax' and 'zscale'")
        
    def get_normalized_latency(self, individual:NASIndividual, norm_style:Literal["minmax", "zscale"]="minmax")->float:
        """
        Normalize the latency of the individual.
        
        Args:
            individual (NASIndividual): The individual to normalize.
            norm_style (Literal["minmax", "zscale"], optional): The normalization style. Defaults to "minmax".
        
        Returns:
            float: The normalized latency. Higher is better.
        """
        if not hasattr(self, "_env"):
            latency = self.searchspace.list_to_score(\
                individual.architecture, score=f"{self.searchspace.target_device}_latency")
        elif hasattr(self, "_env") and getattr(self, "_env").name == "marcella-plus":  # updating how we compute latency for M+
            latency = self._env.compute_hardware_cost(
                individual.architecture
            )
        
        if norm_style == "minmax":
            # min-max inverse-normalize the latency
            return 1 - ((latency - self.latency_stats["min"]) / (self.latency_stats["max"] - self.latency_stats["min"]))

        elif norm_style == "zscale":
            # z-score inverse-normalize the latency
            return -1 * ((latency - self.latency_stats["mean"]) / self.latency_stats["std"])

        else:
            raise ValueError(f"Invalid normalization style: {norm_style}. Accepted values are 'minmax' and 'zscale'")

    def combine_scores(self, performance_scores:float, efficiency_score:float)->float:
        """
        Combine the performance and efficiency scores into a single score.

        Args:
            performance_scores (float): The performance scores. The higher the better.
            efficiency_score (float): The efficiency scores. The higher the better.

        Returns:
            float: The combined score. The higher the better.
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
        return self.fitness_function(individual).fitness

    def _set_env(self, env:NASEnv):
        self._env = env


class Rewardv1(Rewardv0):
    """
    Reward function for the HW-NAS environment.
    Overwrites methods of Rewardv0 to make improvements in performance and efficiency
    exponentially more important.
    """
    reward_version: str = "rewardv1"

    def __init__(self, *args, exponent=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.exponent = exponent  # Control the steepness of the exponential curve

    def combine_scores(self, performance_score: float, efficiency_score: float) -> float:
        """
        Combine the performance and efficiency scores into a single score with an exponential trait.

        Args:
            performance_score (float): The performance score. The higher the better.
            efficiency_score (float): The efficiency score. The higher the better.

        Returns:
            float: The combined score. The higher the better.
        """
        if not (isinstance(performance_score, float) and isinstance(efficiency_score, float)):
            raise ValueError(f"""
                The input scores must both be float! 
                Provided input: performance_score {type(performance_score)}, efficiency_score {type(efficiency_score)}
            """)

        # Apply exponential transformation to both scores
        transformed_performance = self._exponential_transform(performance_score)
        transformed_efficiency = self._exponential_transform(efficiency_score)

        # Combine the transformed scores using the weights
        return (np.array([transformed_performance, transformed_efficiency]) @ self.weights).item()

    def _exponential_transform(self, score: float) -> float:
        """
        Apply an exponential transformation to the score.

        Args:
            score (float): The input score (between 0 and 1).

        Returns:
            float: The transformed score.
        """
        return score ** self.exponent
    
class Rewardv2(Rewardv1):
    """
    Reward function for the HW-NAS environment.
    Overwrites methods of Rewardv0 to make improvements in performance and efficiency
    exponentially more important.
    """
    reward_version: str = "rewardv2"

    def __init__(self, *args, exponent=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.exponent = exponent  # Control the steepness of the exponential curve

    def get_reward(self, individual:NASIndividual)->float:
        """
        Compute the reward associated to the modification operation.
        Here, the reward is the fitness of the newly generated individual.

        The reward is also offset with a small costant to prevent the agent from stalling.
        """
        # here the reward is the fitness of the individual
        return self.fitness_function(individual).fitness - 0.5
    
class Rewardv3(Rewardv0):
    """
    Reward function for the HW-NAS environment.
    Overwrites methods of Rewardv0 to consider training free metrics instead of
    the validation accuracy for fitness computation.
    """
    reward_version: str = "rewardv3"
    performance_score = "normalized_training_free_score"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.score_names = [
            "naswot_score",
            "logsynflow_score",
            "skip_score"
        ]

    def get_performance_score(self, individual:NASIndividual)->float:
        """
        Return the performance score of the individual.
        """
        return self.get_normalized_training_free_score(individual)

    def get_normalized_training_free_score(
            self, 
            individual:NASIndividual, 
            norm_style:Literal["minmax", "zscale"]="minmax"
        )->float:
        """
        Return the normalized training free score of the individual.
        """
        training_free_scores = []
        for score_name in self.score_names:
            # compute the individual's score
            individual_score = self.searchspace.list_to_score(individual.architecture, score_name)

            if norm_style == "minmax":
                score_min, score_max = self.searchspace.get_score_min_and_max(score_name)
                training_free_scores.append(
                    (individual_score - score_min) / (score_max - score_min)
                )
            elif norm_style == "zscale":
                score_mean, score_std = self.searchspace.get_score_mean_and_std(score_name)
                training_free_scores.append(
                    (individual_score - score_mean) / (score_std)
                )
        # return the average of the training free scores -- giving equal weight to all scores
        return np.mean(training_free_scores).item()

class Rewardv4(Rewardv3):
    """
    Reward function for the HW-NAS environment.
    Overwrites methods of Rewardv3 to introduce exponentiation of reward components.
    """
    reward_version: str = "rewardv4"

    def __init__(self, *args, exponent=6, **kwargs):
        super().__init__(*args, **kwargs)
        self.exponent = exponent  # Control the steepness of the exponential curve
    
    def combine_scores(self, performance_score: float, efficiency_score: float) -> float:
        """
        Combine the performance and efficiency scores into a single score with an exponential trait.

        Args:
            performance_score (float): The performance score. The higher the better.
            efficiency_score (float): The efficiency score. The higher the better.

        Returns:
            float: The combined score. The higher the better.
        """
        if not (isinstance(performance_score, float) and isinstance(efficiency_score, float)):
            raise ValueError(f"""
                The input scores must both be float! 
                Provided input: performance_score {type(performance_score)}, efficiency_score {type(efficiency_score)}
            """)

        # Apply exponential transformation to both scores
        transformed_performance = self._exponential_transform(performance_score)
        transformed_efficiency = self._exponential_transform(efficiency_score)

        # Combine the transformed scores using the weights
        return (np.array([transformed_performance, transformed_efficiency]) @ self.weights).item()

    def _exponential_transform(self, score: float) -> float:
        """
        Apply an exponential transformation to the score.

        Args:
            score (float): The input score (between 0 and 1).

        Returns:
            float: The transformed score.
        """
        return score ** self.exponent
    
class Rewardv5(Rewardv3):
    """
    Reward function for the HW-NAS environment.
    Overwrites methods of Rewardv3 to introduce exponentiation of reward components.
    """
    reward_version: str = "rewardv5"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_reward(self, individual: NASIndividual, **kwargs) -> float:
        """
        Compute the reward for the individual.
        """
        this_fitness = self.fitness_function(individual).fitness 

        if not hasattr(self, "previous_fitness"):
            self.previous_fitness = this_fitness
            return 0
        else:
            differential_fitness = this_fitness - self.previous_fitness
            self.previous_fitness = this_fitness
            return differential_fitness

class Rewardv6(Rewardv5):
    """
    Reward function for the HW-NAS environment.
    Overwrites methods of Rewardv5 to introduce exponentiation of the fitness function.
    """
    reward_version: str = "rewardv6"

    def __init__(self, exponent:int = 4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exponent = exponent
    
    def _exponential_transform(self, score: float) -> float:
        """
        Apply an exponential transformation to the score.

        Args:
            score (float): The input score (between 0 and 1).

        Returns:
            float: The transformed score.
        """
        return score ** self.exponent

    def fitness_function(self, individual: NASIndividual) -> NASIndividual:
        """Applies exponential transportation to the fitness score to skew distribution."""
        individual_fitness = super().fitness_function(individual).fitness
        FitnessObject: namedtuple = namedtuple("FitnessObject", "fitness") 
        
        return FitnessObject(self._exponential_transform(individual_fitness))

