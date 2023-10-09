from gym import spaces
import numpy as np
from itertools import chain
from numpy.typing import NDArray
from .base_env import BaseNASEnv
from commons import BaseInterface
from .utils import NASIndividual
from typing import Iterable, Tuple, Text

class NASEnv(BaseNASEnv): 
    """
    gym.Env for pure-RL-based NAS. Architectures are evaluated using training-free metrics only.

    This env is mainly inspired from two works: 
    1. Neural Architecture Search with Reinforcement Learning, https://arxiv.org/pdf/1611.01578.pdf
    2. FreeREA: Training-Free Evolution-Based Architecture Search, https://arxiv.org/pdf/2207.05135.pdf
    """
    def __init__(self, 
                 searchspace_api:BaseInterface,
                 scores:Iterable[Text]=["naswot_score", "logsynflow_score", "skip_score"], 
                 n_mods:int=1,
                 max_timesteps:int=50):
        
        self.searchspace = searchspace_api
        self.score_names = scores
        self.n_mods = n_mods
        self.max_timesteps = max_timesteps

        """
        Each individual network can be univoquely identified with `m` (clearly enough, `m = m(searchspace)`) characters. 
        Each of these characters can take any of the `n` operations in the search-space, thus each individual can be 
        perfectly represented via integer encoding.
        In this case, observations would be `(m,)` tensors whose values would range in [0, n-1] and
        represent one of the different `n` genes.
        """
        self.observation_space = spaces.MultiDiscrete(
            [len(self.searchspace.all_ops) for _ in range(self.searchspace.architecture_len)]
            )
        """
        The controller interaction with individual networks takes place choosing what block to change
        and how to do it.
        In particular, it chooses one (or many) of the different `m` blocks and then selects one of the `n` 
        available operations for the modification here considered. Selecting the block `m+1` corresponds 
        to not applying any change to the architecture considered. The controller applies up to 
        `n_modifications` modifications.
        """
        # single action = [(block_to_modify, new_block)] * n_modifications
        action_space = list(chain(*[
            (self.searchspace.architecture_len + 1, len(self.searchspace.all_ops))
            for _ in range(self.n_mods)])
        )
        self.action_space = spaces.MultiDiscrete(action_space)
        
        # initializes population, timestep counter and current maximal fitness
        self.reset()
    
    @property
    def name(self): 
        return "nasenv"

    def normalize_score(self, score_value:float, score_name:Text, type:Text="std")->float:
        """
        Normalize the given score value using a specified normalization type.

        Args:
            score_value (float): The score value to be normalized.
            score_name (Text): The name of the score used for normalization.
            type (Text, optional): The type of normalization to be applied. Defaults to "std".

        Returns:
            float: The normalized score value.

        Raises:
            ValueError: If the specified normalization type is not available.

        Note:
            The available normalization types are:
            - "std": Standard score normalization using mean and standard deviation.
        """
        if type == "std":
            score_mean = self.searchspace.get_score_mean(score_name)
            score_std = self.searchspace.get_score_std(score_name)
            
            return (score_value - score_mean) / score_std
        else:
            raise ValueError(f"Normalization type {type} not available!")
    
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
            # individual fitness is a convex combination of multiple scores - all scores here have the same weight
            individual._fitness = (np.ones_like(scores) / len(scores)) @ scores
        
        return individual

    def mount_architecture(self, empty_individual:NASIndividual, architecture_encoded:NDArray)->NASIndividual:
        """
        Mount the architecture on the empty individual using the specified architecture encoding.

        Args:
            empty_individual (NASIndividual): The empty individual to be mounted with the architecture.
            architecture_encoded (NDArray): The encoded architecture representation.

        Returns:
            NASIndividual: The individual with the updated architecture.

        Note:
            The method decodes the architecture from the encoding, converts it to a list representation,
            and updates the architecture of the empty individual with the decoded architecture.
        """
        # architecture encoded -> architecture decoded -> architecture list -> individual with updated architecture list
        empty_individual.update_architecture(
            self.searchspace.architecture_to_list(
                architecture_string=self.searchspace.decode_architecture(
                    architecture_encoded=architecture_encoded, 
                    onehot=False
                    )
                )
            )
        
        return empty_individual

    def update_current_net(self):
        """
        Update the current net individual with the encoded architecture stored in the observation.

        Note:
            The method initializes the current net individual and mounts the architecture on it using the encoded observation.

        """
        # initialize the current net individual
        self.current_net = NASIndividual(architecture=None, 
                                         index=None, 
                                         architecture_string_to_idx=self.searchspace.architecture_to_index)
        
        self.current_net = self.mount_architecture(self.current_net, self._observation)
        
        # updating the fitness value
        self.current_net = self.fitness_function(self.current_net)

    def perform_modification(self, new_individual:NDArray, modification:Tuple[int, int])->NDArray: 
        """
        Perform modification on the new individual based on the specified modification tuple.

        Args:
            new_individual (NDArray): The new individual to be modified.
            modification (Tuple[int, int]): The modification tuple specifying where and how to make the modification.

        Returns:
            NDArray: The modified individual.

        Note:
            The modification operation updates the new individual based on the specified modification tuple.
        """

        try: 
            where_to_change, how_to_change = modification
            # overwriting mutant_locus with mutant_gene
            new_individual[where_to_change] = how_to_change
        except IndexError: 
            """
            Index error is caused by modification[0] being outside of observation boundaries. 
            This corresponds to perform the "leave-as-is" action
            """
            pass

        return new_individual
        
    def _get_obs(self)->NDArray: 
        """Return current observation."""
        return self._observation
    
    def _get_info(self)->dict: 
        """Return the info dictionary."""
        info_dict = {
            "current_network": self.current_net.architecture,
            "test_accuracy": self.searchspace.list_to_accuracy(self.current_net.architecture),
            "training_free_score": self.current_net.fitness,
            "timestep": self.timestep_counter, 
        }

        return info_dict

    def is_done(self)->bool: 
        """Returns `True` at episode termination and `False` before."""
        return self.timestep_counter + 1 >= self.max_timesteps
    
    def reset(self)->NDArray:
        """Resets custom env attributes."""
        self._observation = self.observation_space.sample()
        self.update_current_net()
        self.timestep_counter= 0

        return self._get_obs()
    
    def get_reward(self, new_individual:NASIndividual)->float:
        """
        Compute the reward associated to the modification operation.
        Here, the reward is defined as the gain in fitness between the original and new invidual.
        """
        new_individual_fitness, current_individual_fitness = new_individual.fitness, self.current_net.fitness
        # we want to reward actions that increase the value of fitness (proxy for increase in test accuracy)
        return new_individual_fitness - current_individual_fitness
    
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
        original_encoded = self._observation
        # copying parent encoding before performing mutations - copy=True is slow, using a = I @ b instead
        new_individual_encoded = np.diag(np.ones_like(original_encoded)) @ original_encoded
        # perform all the modifications in `mods_list`
        for mod in mods_list:
            new_individual_encoded = self.perform_modification(new_individual=new_individual_encoded, 
                                                               modification=mod)
        
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
        # check whether or not the episode is terminated
        terminated = self.is_done()
        
        # overwrite current obs
        self._observation = new_individual_encoded
        # update consequently the current net field
        self.update_current_net()

        # retrieve info
        info = self._get_info()

        return self._observation, reward, terminated, info

