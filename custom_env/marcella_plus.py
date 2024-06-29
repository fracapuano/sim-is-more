from src import Base_Interface
from .oscar import OscarEnv
from typing import (
    Iterable, 
    Text, 
    Optional, 
    Dict, 
    List
)
from numpy.typing import NDArray
from .utils import (
    ProbabilityDistribution, 
    TruncatedNormalDistribution,
    shuffle_dict_values,
    UniformProbabilityDistribution
)
import numpy as np
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
            **kwargs
        )

    @property
    def name(self): 
        return "marcella-plus"
    
    def normalize_score(self, score_value:float, score_name:Text, type:Text="std")->float:
        """
        Normalize the given score value based on the score name and type.
        This is needed because unlike OscarEnv (MarcellaEnv) we cannot read indicators like
        mean, std, min or max from a lookup table as they change from reset to reset.

        Args:
            score_value (float): The score value to be normalized.
            score_name (Text): The name of the score.
            type (Text, optional): The type of normalization to be applied. Defaults to "std".

        Returns:
            float: The normalized score value.
        """
        if "latency" in score_name:
            return self.normalize_hardware_cost(hardware_cost=score_value, type=type)
        else:
            return super().normalize_score(score_value, score_name, type)

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

        elif type == "minmax":
            return (hardware_cost - self.min_hardware_cost) / (self.max_hardware_cost - self.min_hardware_cost)

        else:
            msg = \
            f"""
            Normalization type {type} not implemented. Currently supported types are:
            - std: Z-normalization
            - minmax: Min-Max normalization
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

    def _sample_blocks_latencies(self)->None:
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

    def _set_normalization_params(self):
        """
        Sets the normalization parameters for the hardware costs.
        """
        if self.normalization_type == "std":
            self.mean_hardware_cost = self.hardware_costs.mean().item()
            self.std_hardware_cost = self.hardware_costs.std().item()

        elif self.normalization_type == "minmax":
            self.min_hardware_cost = self.hardware_costs.min().item()
            self.max_hardware_cost = self.hardware_costs.max().item()

    def reset(self, seed:Optional[int]=None)->NDArray:
        """Resets custom env attributes."""
        # sampling a new starting observation
        self._observation = self.observation_space.sample()

        # shuffling the operations latency distributions at each reset
        #self.blocks_distribution = shuffle_dict_values(self.blocks_distribution)
        
        # aligning the reward handler with the current state of Marcella+
        self.reward_handler._env = self

        # sampling a new distribution of blocks' latencies
        self._sample_blocks_latencies()
        
        # computing hardware cost samples
        self._set_hardware_costs()
        self._set_normalization_params()

        self.latency_stats = {
                "min": self.hardware_costs.min().item(),
                "max": self.hardware_costs.max().item(),
                "mean": self.hardware_costs.mean().item(),
                "std": self.hardware_costs.std().item()
            }
        
        # possibly storing the blocks' latencies in the observation
        # self._observation["blocks_latency"] = self.blocks_latency
        
        # setting the latency cutoff to the cutoff percentile-th of the hardware measures
        self.max_latency = np.percentile(self.hardware_costs, self.cutoff_percentile)
        self.update_current_net()

        # flushing out timestep counter and the buffer of observations
        self.timestep_counter= 0
        self.observations_buffer.clear()

        return self._get_obs(), self._get_info()

