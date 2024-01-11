from custom_env.utils import NASIndividual
from src import Base_Interface
from .oscar import OscarEnv
from typing import (
    Iterable, 
    Text, 
    Optional, 
    Dict, 
    Tuple, 
    List
)
from numpy.typing import NDArray
from .utils import (
    ProbabilityDistribution, 
    NASIndividual,
    TruncatedNormalDistribution
)
import numpy as np
import random
from scipy.stats import percentileofscore
from operator import itemgetter


class MarcellaPlusEnv(OscarEnv):
    """
    gym.Env for Multi-task Hardware-aware pure-RL-based NAS. Architectures are evaluated using training-free metrics 
    only as well as specific hardware related metrics. During the training procedure, the target device the agent
    interacts with is updated to force the agent to solve the task independently on the actual device.
    """
    def __init__(self, 
                 searchspace_api:Base_Interface,
                 blocks_distribution:Optional[Dict[Text, ProbabilityDistribution]]=None,
                 cutoff_percentile:float=85,
                 custom_devices:Optional[Iterable[Text]]=None,
                 **kwargs):
        
        self.searchspace = searchspace_api
        """
        This variable stores the distribution of latencies for each block.
        At each call of the `reset()` method, blocks' latencies are sampled from said distributions

        If custom_devices is None, the distribution of latencies is reconstruced accessing all devices directly 
        from the searchspace object and fitting a truncated normal distribution on the latencies of each block.
        """
        if blocks_distribution is None:
            # retrieving the measurements of all the devices from the searchspace object
            device_measurements = self.searchspace.blocks_latency(custom_devices=custom_devices)

            self.blocks_distribution = {
                op: TruncatedNormalDistribution(
                    mean=device_measurements[op].mean(), 
                    std=device_measurements[op].std()
                )
                for op in self.searchspace.all_ops
            }
        else:
            self.blocks_distribution = blocks_distribution
        
        # sample a new set of blocks' latencies
        self._sample_blocks_latencies()

        super().__init__(
            searchspace_api=searchspace_api,
            cutoff_percentile=cutoff_percentile,
            target_device=None,  # removing target device, devices are simulated here
            **kwargs
        )

    @property
    def name(self): 
        return "marcella-plus"
    
    def normalize_hardware_cost(self, hardware_cost:float, type:Text="std")->float:
        """
        Applies some transformation to the hardware cost.

        Args:
            hardware_cost (float): Hardware cost value to be normalized.

        Returns:
            float: hardware cost value.
        """
        if type == "std":
            return (hardware_cost - self.mean_hardware_cost) / self.std_hardware_cost

        else:
            msg = \
            f"""
            Normalization type {type} not implemented. Currently supported types are:
            - std: Z-normalization
            """
            raise NotImplementedError(msg)

    def compute_hardware_cost(self, architecture_list:List[Text])->float:
        """
        Computes the hardware cost of a given architecture.

        Args:
            architecture_list (List[Text]): Architecture to be evaluated.

        Returns:
            float: Hardware cost of the architecture.
        """
        # [op~block, op2~block2, ...] -> [op, op2, ...]
        get_architecture_ops = lambda a: list(map(lambda x: x.split("~")[0], a))
        
        return sum(itemgetter(*get_architecture_ops(architecture_list))(self.blocks_latency))

    def _set_hardware_costs(self):
        """
        Sets the hardware costs based on the latency of the networks within the network pool on the target device.
        """
        self.hardware_costs = np.fromiter(
            map(lambda a: self.compute_hardware_cost(self.searchspace.architecture_to_list(a)), self.network_pool), 
            dtype="float"
        )

        # extracting the mean and std of the scores
        self.hw_score_mean, self.hw_score_std = self.hardware_costs.mean(), self.hardware_costs.std()
        
        # normalizing the scores using mean and std
        self.normalized_hardware_costs = (self.hardware_costs - self.hw_score_mean) / self.hw_score_std

    def fitness_function(self, individual:NASIndividual)->NASIndividual: 
        """
        Directly overwrites the fitness attribute for a given individual.

        Args: 
            individual (NASIndividual): Individual to score.

        Returns:
            NASIndividual: Individual, with fitness field.
        """
        scores = np.array([
            self.normalize_score(
                score_value=self.searchspace.list_to_score(input_list=individual.architecture, 
                                                            score=score), 
                score_name=score
            )
            for score in self.score_names
        ])
        # computing hardware performance of current individual
        hardware_performance = np.array([
            self.normalize_hardware_cost(
                self.compute_hardware_cost(architecture_list=individual.architecture)
            )
        ])

        # individual fitness is a convex combination of multiple scores
        network_score = (np.ones_like(scores) / len(scores)) @ scores
        network_hardware_performance = \
            (np.ones_like(hardware_performance) / len(hardware_performance)) @ hardware_performance
        
        # saving the different scores per each individual
        individual._scores = \
            {s_name: s for s_name, s in zip(self.score_names, scores)} | \
            {p_name: p for p_name, p in zip(["normalized-latency"], hardware_performance)}

        # in the hardware aware contest performance is in a direct tradeoff with hardware performance
        individual._fitness = np.array([network_score, -network_hardware_performance]) @ self.weights
        
        return individual
    
    def _sample_blocks_latencies(self) -> None:
        """
        Samples a new set of latencies for each block.

        This method generates a new set of latencies for each block in the search space.
        The latencies are sampled from a distribution specific to each operation.

        Returns:
            None
        """
        # sampling a new set of latencies for each block
        self.blocks_latency = {
            op: self.blocks_distribution[op].sample().item() for op in self.searchspace.all_ops
        }

    def reset(self, seed:Optional[int]=None)->NDArray:
        """Resets custom env attributes."""
        # sampling a new starting observation
        self._observation = self.observation_space.sample()
        
        # sampling a new distribution of blocks' latencies
        self._sample_blocks_latencies()
        # computing hardware cost samples
        self._set_hardware_costs()
        
        # FIXME: possibly storing the blocks' latencies in the observation
        # self._observation["blocks_latency"] = self.blocks_latency
                
        # storing mean and std of hardware cost in the observation
        self.mean_hardware_cost = np.mean(self.hardware_costs)
        self.std_hardware_cost = np.std(self.hardware_costs)
        
        # setting the latency cutoff to the cutoff percentile-th of the hardware measures
        self.max_latency = np.percentile(self.hardware_costs, self.cutoff_percentile)
        self.update_current_net()

        # flushing out timestep counter and the buffer of observations
        self.timestep_counter= 0
        self.observations_buffer.clear()

        return self._get_obs(), self._get_info()
    
    def is_terminated(self)->bool:
        """
        Checks whether or not the episode is terminated.
        MarcellaPlusEnv is terminated when the latency of the current network is higher than the latency cutoff.
        """
        return bool(self.compute_hardware_cost(self.current_net.architecture) > self.max_latency)

    def step(self, action:NDArray)->Tuple[NDArray, float, bool, dict]: 
        """Steps the episode having a given action.
        
        Args:
            action (NDArray): Action to be performed.
        
        Returns:
            Tuple[NDArray, float, bool, dict]: New observation (after having performed the action), 
                                               reward value,
                                               done signal (True at episode termination), 
                                               info dictionary
        """
        
        # increment timestep counter (used to declare episode termination)
        self.timestep_counter += 1

        # split action into a list of 2-item tuples (as per the definition of action space)
        mods_list = [(action[i], action[i+1]) for i in range(0, len(action), 2)]
        original_encoded = self._observation["architecture"]
        # copying parent encoding before performing mutations - copy=True is slow, using a = I @ b instead
        new_individual_encoded = np.diag(np.ones_like(original_encoded)) @ original_encoded
        # perform all the modifications in `mods_list`
        for mod in mods_list:
            new_individual_encoded = \
                self.perform_modification(new_individual=new_individual_encoded, modification=mod)
        
        # creating new individual with reinforced-controlled genotype
        reinforced_individual = NASIndividual(architecture=None,
                                              index=None,
                                              architecture_string_to_idx=self.searchspace.architecture_to_index)
        # mounting the architecture on the new individual
        reinforced_individual = self.mount_architecture(reinforced_individual, new_individual_encoded)
        # score individual based on its test accuracy
        reinforced_individual = self.fitness_function(reinforced_individual)
        # compute the reward associated with producing reinforced_individual
        reward = self.get_reward(new_individual=reinforced_individual)
        
        # overwrite current obs architecture
        self._observation["architecture"] = new_individual_encoded
        # update consequently the current net field
        self.update_current_net()
        # update current obs latency value
        self._observation["latency_value"] = \
            np.array([self.compute_hardware_cost(architecture_list=self.current_net.architecture)], dtype=np.float32)

        # check whether or not the episode is terminated
        terminated = self.is_terminated()
        # check if the episode is truncated
        truncated = self.is_truncated()
        # retrieve info
        info = self._get_info()

        if terminated:
            reward = -1

        # storing the reward in a variable to be accessed by the render method
        self.step_reward = reward

        return self._observation, reward, terminated, truncated, info

