from .utils import NASIndividual
import numpy as np
from typing import Text, Union, Iterable
from numpy.typing import NDArray

class Rewardv0:

    def fitness_function(self, individual:NASIndividual)->NASIndividual: 
        """
        Directly overwrites the fitness attribute for a given individual.

        Args: 
            individual (NASIndividual): Individual to score.

        Returns:
            NASIndividual: Individual, with fitness field.
        """
        if individual.fitness is None:  # None at initialization only
            tf_scores = np.array([
                self.normalize_score(
                    score_value=self.searchspace.list_to_score(input_list=individual.architecture, score=score), 
                    score_name=score,
                    type=self.normalization_type
                )
                for score in self.score_names
            ])
            hardware_cost = np.array([
                self.normalize_score(
                    score_value=self.compute_hardware_cost(architecture_list=individual.architecture),
                    score_name=f"{self.target_device}_{metric}",
                    type=self.normalization_type
                )
                for metric in ["latency"]  # change here to add more hardware aware metrics
            ])
            # individual fitness is a linear combination of multiple scores
            network_tf_score = (np.ones_like(tf_scores) / len(tf_scores)) @ tf_scores
            network_hw_score =  1 - ((np.ones_like(hardware_cost) / len(hardware_cost)) @ hardware_cost)
            
            # saving the scores within each individual
            individual._scores = \
                {s_name: s for s_name, s in zip(self.score_names, tf_scores)} | \
                {p_name: p for p_name, p in zip(["normalized-latency"], hardware_cost)} | \
                {"network_tf_score": network_tf_score, 
                    "network_hw_score": network_hw_score}

            # in the hardware aware contest performance is in a direct tradeoff with hardware performance
            individual._fitness = self.combine_scores(network_tf_score, network_hw_score).item()
        
        return individual

    def combine_scores(self, performance_scores:Union[float, NDArray], efficiency_score:Union[float, NDArray])->Union[float, NDArray]:
        """
        Combine the performance and efficiency scores into a single score.

        Args:
            performance_scores (Union[float, NDArray]): The performance scores.
            efficiency_score (Union[float, NDArray]): The efficiency scores.

        Returns:
            Union[float, NDArray]: The combined scores.
        """
        to_array = lambda x: np.array(x) if not isinstance(x, np.ndarray) else x
        log_squash = lambda x: x - np.log10(1e-1+1-x)

        w = -0.3
        # transform = lambda x: log_squash(to_array(x)).reshape(-1,1)
        transform = lambda x: to_array(x).reshape(-1,1)
        
        # return np.hstack([transform(performance_scores), transform(efficiency_score)]) @ self.weights
        return transform(performance_scores) * transform(1e-3 + 1-efficiency_score)**w

    def compute_hardware_cost(self, architecture_list:Iterable[Text])->float:
        """
        Computes the hardware cost based on the given architecture list.

        Args:
            architecture_list (Iterable[Text]): The list representation of the architecture.

        Returns:
            float: The computed hardware cost.
        """
        return self.searchspace.list_to_score(input_list=architecture_list, score=f"{self.target_device}_latency")

    def get_reward(self, new_individual:NASIndividual)->float:
        """
        Compute the reward associated to the modification operation.
        Here, the reward is the fitness of the newly generated individual
        """
        # removing a small penalty to the reward to discourage stalling
        return new_individual.fitness - 1