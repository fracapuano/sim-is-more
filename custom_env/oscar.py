import random
import pygame
from .render import (
    create_background_scatter,
    create_background_vlines,
    draw_architectures_on_background,
    draw_hbars,
    figure_to_image
)
import numpy as np
from itertools import chain
from .nas_env import NASEnv
from gymnasium import spaces
from collections import deque
from src import Base_Interface
import matplotlib.pyplot as plt
from operator import itemgetter
from numpy.typing import NDArray
from .utils import NASIndividual
from scipy.stats import percentileofscore
from typing import Iterable, Text, Tuple, Dict, Optional


class OscarEnv(NASEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 5, 
    }
    """
    gym.Env for Hardware-aware RL-based NAS. 
    Architectures are evaluated using training-free and measured latency, as per hardware-related metrics.
    """
    def __init__(self, 
                 searchspace_api:Base_Interface,
                 scores:Iterable[Text]=["naswot_score", "logsynflow_score", "skip_score"],
                 n_mods:int=1,
                 max_timesteps:int=20,
                 cutoff_percentile:float=85.,
                 target_device:Text="edgegpu",
                 weights:Iterable[float]=[0.6, 0.4],
                 latency_cutoff:Optional[float]=None,
                 observation_buffer_size:int=10,
                 normalization_type:Optional[Text]=None,
                 n_samples:Optional[int]=None):
        
        super().__init__(
            searchspace_api=searchspace_api,
            scores=scores,
            n_mods=n_mods,
            max_timesteps=max_timesteps, 
            normalization_type=normalization_type
        )
        # setting the target device
        self.target_device = target_device
        # casting weights to numpy array
        self.weights = np.array(weights)
        # initializing the observations buffer size, for rendering purposes
        self.observations_buffer_size = observation_buffer_size
        self.observations_buffer = deque(maxlen=self.observations_buffer_size)
        
        # initializing a pool of networks to be used for distributional-insights purposes
        self.init_networks_pool(n_samples=n_samples)
        # store the hardware costs for all the networks in the pool
        self._set_hardware_costs()
        
        # latency cutoff can be hardcoded from outside for specific experiments
        if latency_cutoff is not None:
            self.max_latency = latency_cutoff
        else:
            self.cutoff_percentile = cutoff_percentile
            self.max_latency = self.get_max_latency(percentile=cutoff_percentile)

        """
        Each individual network can be univoquely identified with `m` (clearly enough, `m = m(searchspace)`) 
        characters, as well as its hardware oriented performance.
        Each of these characters can take any of the `n` operations in the search-space, thus each individual 
        can be perfectly represented via integer encoding.
        In this case, observations would be `(m+1,)` tensors whose values would range in [0, n-1] for the first
        (m,) elements and the latency value of the architecture in position (m+1). For readibility, I have encoded
        this structure using a `gym.spaces.Dict`
        """
        self.observation_space = spaces.Dict(
            {
                "architecture": spaces.MultiDiscrete([
                    len(self.searchspace.all_ops) for _ in range(self.searchspace.architecture_len)
                    ]),
                "latency_value": spaces.Box(low=0, high=float("inf"), shape=(1,))
            }
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

    @property
    def name(self):
        return "oscar"
    
    def get_max_latency(self, percentile:Optional[float]=None) -> float:
        """
        Returns the maximum latency value for a given percentile.

        Parameters:
            percentile (float): The desired percentile value.

        Returns:
            float: The maximum latency value for the given percentile.
        """
        percentile = percentile if percentile is not None else self.cutoff_percentile
        return float(np.percentile(
            self.hardware_costs,
            percentile)
        )

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
                    score_value=self.searchspace.list_to_score(input_list=individual.architecture, score=score), 
                    score_name=score,
                    type=self.normalization_type
                )
                for score in self.score_names
            ])
            hardware_performance = np.array([
                self.normalize_score(
                    score_value=self.searchspace.list_to_score(input_list=individual.architecture, score=f"{self.target_device}_{metric}"),
                    score_name=f"{self.target_device}_{metric}",
                    type=self.normalization_type
                )
                for metric in ["latency"]  # change here to add more hardware aware metrics
            ])
            # individual fitness is a linear combination of multiple scores
            network_score = (np.ones_like(scores) / len(scores)) @ scores
            network_hardware_performance =  (np.ones_like(hardware_performance) / len(hardware_performance)) @ hardware_performance
            
            # saving the scores within each individual
            individual._scores = \
                {s_name: s for s_name, s in zip(self.score_names, scores)} | \
                {p_name: p for p_name, p in zip(["normalized-latency"], hardware_performance)}

            # in the hardware aware contest performance is in a direct tradeoff with hardware performance
            individual._fitness = np.array([network_score, -network_hardware_performance]) @ self.weights
        
        return individual

    def compute_hardware_cost(self, architecture_list:Iterable[Text])->float:
        """
        Computes the hardware cost based on the given architecture list.

        Args:
            architecture_list (Iterable[Text]): The list representation of the architecture.

        Returns:
            float: The computed hardware cost.
        """
        return self.searchspace.list_to_score(input_list=architecture_list, score=f"{self.target_device}_latency")

    def _get_obs(self)->Dict[Text, spaces.Space]:
        return self._observation
    
    def _get_info(self)->dict: 
        """Return the info dictionary."""
        current_net_latency = self._get_obs()["latency_value"].item()

        training_free_coefficients = np.ones(len(self.score_names)) / len(self.score_names)
        training_free_score = \
            np.array(itemgetter(*self.score_names)(self.current_net._scores)).reshape(-1,) @ training_free_coefficients
        
        info_dict = {
            "current_network": self.current_net.architecture,
            "training_free_score": training_free_score,
            "timestep": self.timestep_counter,
            "current_net_latency": current_net_latency,
            "current_net_latency_percentile": percentileofscore(self.hardware_costs, current_net_latency),
            "latency_cutoff": self.max_latency,
            "is_terminated": self.is_terminated(),
            "is_truncated": self.is_truncated(),
            # test_accuracy is obtained from a lookup table and never accessed during training
            "test_accuracy": self.searchspace.list_to_accuracy(input_list=self.current_net.architecture),
            "networks_seen": len(self.networks_seen)
        }
        info_dict |= self.current_net._scores
        
        return info_dict

    def is_terminated(self)->bool:
        """
        Returns `True` if the episode is terminated and `False` otherwise.
        Episodes are terminated when an agent produces an architecture with a latency value greater than the cutoff.
        """
        return self._observation["latency_value"].item() >= self.max_latency

    def get_reward(self, new_individual:NASIndividual)->float:
        """
        Compute the reward associated to the modification operation.
        Here, the reward is the fitness of the newly generated individual
        """
        return new_individual.fitness

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
        # check whether or not the episode is truncated
        truncated = self.is_truncated()
        # retrieve info
        info = self._get_info()

        # if terminated:
        #     reward = -1

        # storing the reward in a variable to be accessed by the render method
        self.step_reward = reward

        # keeping track of the number of networks seen from initialization
        self.networks_seen.add(self.searchspace.list_to_architecture(self.current_net.architecture))

        return self._observation, reward, terminated, truncated, info

    def reset(self, seed:Optional[int]=None)->NDArray:
        """Resets custom env attributes."""
        self._observation = self.observation_space.sample()
        self.update_current_net()
        self.timestep_counter= 0
        self.observations_buffer.clear()

        return self._get_obs(), self._get_info()
    
    def _set_combined_scores(self):
        """
        Calculates the combined scores for all architectures in the network pool.

        The combined scores are obtained by normalizing the training free scores for each architecture,
        and then combining them using the combination coefficients.

        Returns:
            None
        """
        # all three scores have the same weight here
        scores_combination_coeffs = np.ones(len(self.score_names)) / len(self.score_names)
        
        # Obtaining a matrix with the normalized training free scores for all the architectures in the searchspace
        self.training_free_scores = np.hstack([
                np.array([
                    # obtaining the score value from the searchspace object
                    self.searchspace.architecture_to_score(architecture_string=a, score=score_name)
                    for score_name in self.score_names]).reshape(-1,1)
                for a in self.networks_pool]
        )
        # computing the mean and std of the scores for the network pool considered   
        self.scores_mean, self.scores_std = np.vstack(
            [self.training_free_scores.mean(axis=1), self.training_free_scores.std(axis=1)]
        )
        
        # normalizing the scores using mean and std
        self.normalized_training_free_scores = (self.training_free_scores - self.scores_mean.reshape(-1,1)) / self.scores_std.reshape(-1,1)

        # combining the scores using the combination coefficients
        self.combined_scores = scores_combination_coeffs @ self.normalized_training_free_scores

    def _set_hardware_costs(self):
        """
        Sets the hardware costs based on the latency of the networks within the network pool on the target device.
        """
        self.hardware_costs = np.fromiter(
            map(lambda a: self.searchspace.architecture_to_score(a, score=f"{self.target_device}_latency"), self.networks_pool), 
            dtype="float"
        )

        # extracting the mean and std of the scores
        self.hw_score_mean, self.hw_score_std = self.hardware_costs.mean(), self.hardware_costs.std()
        
        # normalizing the scores using mean and std
        self.normalized_hardware_costs = (self.hardware_costs - self.hw_score_mean) / self.hw_score_std

    def unpack_buffer(self)->Tuple[NDArray, NDArray]:
        """
        Unpack the observations buffer into x and y coordinates.

        Returns:
            Tuple[NDArray, NDArray]: The x and y coordinates.
        """
        x_coordinates = np.fromiter(
            map(lambda x: x["current_net_latency"], self.observations_buffer),
            dtype="float"
        )
        y_coordinates = np.fromiter(
            map(lambda x: x["training_free_score"], self.observations_buffer),
            dtype="float"
        )

        return x_coordinates, y_coordinates

    def _render_frame(self, mode:Text="human", draw_background:bool=True)->Optional[NDArray]: 
        """
        Renders a frame of the environment.

        Args:
            mode (Text, optional): The rendering mode. Defaults to "human".
            draw_background (bool, optional): Whether to draw the background scatter plot. Defaults to True.

        Returns:
            Optional[NDArray]: The rendered frame as an RGB array if mode is "rgb_array", None otherwise.
        """ 
        screen_size = (640*2, 480*2)

        # populating the combined scores and hardware costs attributes if not already done
        if not hasattr(self, "combined_scores") and draw_background:
            self._set_combined_scores()

        # Initialize Pygame window and clock if not already done
        if not hasattr(self, "window"):
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(screen_size)

        if not hasattr(self, "clock"):
            self.clock = pygame.time.Clock()
        
        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, gridspec_kw={"height_ratios": [8,1,1]}, dpi=400, layout="constrained")
        if draw_background:
            ax1 = create_background_scatter(ax1, self.hardware_costs, self.combined_scores)
        
        # drawing the latency cutoff
        ax1 = create_background_vlines(ax1, self.max_latency, label="Latency Cutoff")
        # drawing the architectures
        ax1 = draw_architectures_on_background(ax1, *self.unpack_buffer(), label="Current Network")
        ax1.set_xlabel("Latency (ms)"); ax1.set_ylabel("Combined Training-Free Score")
        
        # drawing test accuracy and latency percentile bars
        info_dict = self._get_info()
        ax2 = draw_hbars(
            ax2, 
            ["Acc(a)", "Lat(a)"], 
            [info_dict["test_accuracy"], info_dict["current_net_latency_percentile"]],
            height=0.5
        )
        ax2.set_xlim(0, 100)
        ax2.set_ylim(-0.75, 1.75)
        ax2.set_yticks(ax2.get_yticks(), ax2.get_yticklabels())
        ax2.set_title("Current Network Test Accuracy & Latency Percentile")

        # drawing the reward bar
        ax3 = draw_hbars(ax3, [r"$r_t$"], [self.step_reward], height=0.5)
        ax3.set_xlim(-1, 1)
        ax3.set_ylim(-0.5, 0.5)
        ax3.set_title("One-step Transition Reward")
        
        # setting a suptitle
        fig.suptitle("Timestep {}/{}".format(self.timestep_counter, self.max_timesteps))
        
        architectures_frame = figure_to_image(fig, *screen_size)
        self.window.blit(architectures_frame, (0, 0))

        pygame.event.pump()
        # Limit the frame rate
        self.clock.tick(self.metadata["render_fps"])

        if mode=="rgb_array":
            return pygame.surfarray.array3d(self.window)
        elif mode=="human":
            # Update the display
            pygame.display.flip()
            return None

    def render(self, mode="human", draw_background:bool=True):
        """Calls the render frame method."""
        # Storing information associated with current observation in the buffer
        self.observations_buffer.append(self._get_info())
        
        return self._render_frame(mode=mode, draw_background=draw_background)

