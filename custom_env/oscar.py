from gymnasium import spaces
import numpy as np
from src import Base_Interface
from .nas_env import NASEnv
from .utils import NASIndividual
from typing import Iterable, Text, Tuple, Dict, Optional
from numpy.typing import NDArray
from itertools import chain
from operator import itemgetter
import pygame
import matplotlib.pyplot as plt
from PIL import Image
from collections import deque
from copy import copy

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
                 max_timesteps:int=50,
                 cutoff_percentile:float=85.,
                 target_device:Text="edgegpu",
                 weights:Iterable[float]=[0.6, 0.4],
                 latency_cutoff:Optional[float]=None,
                 observation_buffer_size:int=10):
        
        self.searchspace = searchspace_api
        self.score_names = scores
        self.n_mods = n_mods
        self.max_timesteps = max_timesteps
        self.observations_buffer_size = observation_buffer_size

        # latency cutoff can be hardcoded from outside for specific experiments
        if latency_cutoff is not None:
            self.max_latency = latency_cutoff
        else:
            self.max_latency = \
                searchspace_api.latency_readings.percentile_to_value(
                    device=target_device,
                    percentile=cutoff_percentile
                )
        
        self.target_device = target_device
        self.weights = np.array(weights) if not isinstance(weights, np.ndarray) else weights

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
                # latency is always less than 100 ms on all devices considered here
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
    
    def get_max_timesteps(self)->int:
        return self.max_timesteps

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
            score_mean, score_std = self.searchspace.get_score_mean_and_std(score_name)
            
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
            
            # saving the scores within each individual
            individual._scores = \
                {s_name: s for s_name, s in zip(self.score_names, scores)} | \
                {p_name: p for p_name, p in zip(["standardized-latency"], hardware_performance)}

            # in the hardware aware contest performance is in a direct tradeoff with hardware performance
            individual._fitness = np.array([network_score, -network_hardware_performance]) @ self.weights
        
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
        
        self.current_net = self.mount_architecture(self.current_net, self._observation["architecture"])
        
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

    def _get_obs(self)->Dict[Text, spaces.Space]:
        return self._observation
    
    def _get_info(self)->dict: 
        """Return the info dictionary."""
        current_net_latency = self.searchspace.list_to_score(
                    input_list=self.current_net.architecture, 
                    score=f"{self.target_device}_latency"
        )
        info_dict = {
            "current_network": self.current_net.architecture,
            "training_free_score": sum(itemgetter(*self.score_names)(self.current_net._scores)),
            "timestep": self.timestep_counter,
            "current_net_latency": current_net_latency,
            "current_net_latency_percentile": self.searchspace.latency_readings.value_to_percentile(
                device=self.target_device,
                value=current_net_latency),
            "latency_cutoff": self.max_latency,
            "is_terminated": self.is_terminated(),
            "is_truncated": self.is_truncated(),
            # test_accuracy is obtained from a lookup table and never accessed during training
            "test_accuracy": self.searchspace.list_to_accuracy(input_list=self.current_net.architecture),
        }
        info_dict |= self.current_net._scores
        
        return info_dict

    def is_done(self)->bool: 
        """
        Returns `True` at episode termination and `False` before.
        DEPRECATED: use `is_terminated` instead.
        """
        Warning("is_done is deprecated, use is_terminated instead")
        self.timesteps_over = self.timestep_counter >= self.max_timesteps
        self.latency_over = self._observation["latency_value"].item() >= self.max_latency

        return self.timesteps_over or self.latency_over

    def is_terminated(self)->bool:
        """
        Returns `True` if the episode is terminated and `False` otherwise.
        Episodes are terminated when an agent produces an architecture with a latency value greater than the cutoff.
        """
        return self._observation["latency_value"].item() >= self.max_latency

    def reset(self, seed:Optional[int]=None)->NDArray:
        """Resets custom env attributes."""
        super().reset(seed=seed)

        self._observation = self.observation_space.sample()
        self.update_current_net()
        self.timestep_counter= 0
        self.observations_buffer = deque(maxlen=self.observations_buffer_size)

        return self._get_obs(), self._get_info()

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
            np.array(
                [self.searchspace.list_to_score(input_list=self.current_net.architecture, score=f"{self.target_device}_latency")],
                dtype=np.float32
            )
        
        # check whether or not the episode is terminated
        terminated = self.is_terminated()
        # check whether or not the episode is truncated
        truncated = self.is_truncated()
        # retrieve info
        info = self._get_info()

        if terminated:
            reward = -1

        # storing the reward in a variable to be accessed by the render method
        self.step_reward = reward

        return self._observation, reward, terminated, truncated, info

    def _set_combined_scores(self):
        """
        Calculates the combined scores for all architectures in the search space.

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
                    # obtaining the score value from the lookup table
                    self.searchspace.list_to_score(
                        self.searchspace.architecture_to_list(
                            self.searchspace.lookup_table[i]["architecture_string"]
                            ), 
                        score=score_name)
                    for score_name in self.score_names]).reshape(-1,1)
                for i in range(len(self.searchspace))]
        )
        
        # extracting the mean and std of the scores
        scores_mean, scores_std = \
            np.vstack([np.array(self.searchspace.get_score_mean_and_std(s)) for s in self.score_names]).T
        
        # normalizing the scores using mean and std
        self.training_free_scores = (self.training_free_scores - scores_mean.reshape(-1,1)) / scores_std.reshape(-1,1)

        # combining the scores using the combination coefficients
        self.combined_scores = scores_combination_coeffs @ self.training_free_scores

    def _set_hardware_costs(self):
        """
        Sets the hardware costs based on the latency readings of the target device.

        This method retrieves the latency readings for the target device from the searchspace
        and assigns them to the hardware_costs attribute of the current instance.
        """
        self.hardware_costs = np.array(
            getattr(self.searchspace.latency_readings, f"{self.target_device}_readings")
        )
        
    def _draw_background(self)->Tuple[plt.Figure, plt.Axes]:
        
        fig, ax = plt.subplots()

        ax.scatter(
            self.hardware_costs,
            self.combined_scores,
            s=10,
            c="0.8", # light gray
            zorder=0
        )

        ax.set_ylabel("Combined Training-Free Score")
        ax.set_xlabel("Latency")
        # retrieving the current axes limits for vlines method
        ymin, ymax = ax.get_ylim()
        # displaying the latency cutoff
        ax.vlines(
            x=self.max_latency,
            ymin=ymin,
            ymax=ymax,
            colors="red",
            linestyles="dashed",
            label="Latency Cutoff"
        )

        return fig, ax

    def _draw_architectures(self, fig:plt.Figure, ax:plt.Axes)->Tuple[plt.Figure, plt.Axes]:
        
        # unpacking the observations buffer -> turning it into a list to avoid being consumed by the map function
        architectures = list(map(lambda x: x["current_network"], self.observations_buffer))

        x_coordinates = np.fromiter(
            map(lambda x: self.hardware_costs[self.searchspace.architecture_to_index["/".join(x)]], architectures),
            dtype="float"
        )
        # unpacking the buffer into relevant coordinates
        y_coordinates = np.fromiter(
            map(lambda x: self.combined_scores[self.searchspace.architecture_to_index["/".join(x)]], architectures),
            dtype="float"
        )
        
        line, = ax.plot([], [], zorder=1, c="red")
        scatt = ax.scatter([],[], s=50, c="red", marker="X", label="Current Network")

        line.set_data(x_coordinates, y_coordinates)
        scatt.set_offsets(np.c_[x_coordinates, y_coordinates])
        ax.legend(loc = "upper right", framealpha=1., fontsize=12)

        return fig, ax

    def _draw_accuracy_latency_bars(self)->Tuple[plt.Figure, plt.Axes]:
        
        fig, ax = plt.subplots()
        # retrieving the test accuracy and latency percentile from the info dictionary
        info_dict = self._get_info()

        ax.bar(
            x=["Test Accuracy", "Latency (Percentile)"],
            height=[info_dict["test_accuracy"], info_dict["current_net_latency_percentile"]],
        )
        ax.set_ylabel("Test Accuracy / Latency (Percentile)")
        ax.set_ylim(0, 100)

        return fig, ax
    
    def _draw_reward_bar(self)->Tuple[plt.Figure, plt.Axes]:

        fig, ax = plt.subplots()

        ax.bar(
            x=["Reward"],
            height=self.step_reward,
            width=0.5
        )
        ax.set_ylim(-1, 1)
        ax.set_xlim(-0.75, 0.75)
        ax.set_ylabel("Reward")

        return fig, ax
        
    def _render_frame(self):
        # The first 640 pixels are for the scatter plot - The second 640 are for test accuracy and latency - The last 640 are for rewards
        splits = (640, 640, 640)
        screen_size = (sum(splits), 480)

        # populating the combined scores and hardware costs attributes if not already done
        if not hasattr(self, "combined_scores"):
            self._set_combined_scores()
        if not hasattr(self, "hardware_costs"):
            self._set_hardware_costs()

        # Initialize Pygame window and clock if not already done
        if not hasattr(self, "window"):
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(screen_size)

        if not hasattr(self, "clock"):
            self.clock = pygame.time.Clock()

        # The first 640 pixels are for the scatter plot
        # The second 320 are for test accuracy and latency
        # The last 320 are for rewards
        splits = (640, 640, 640)
        screen_size = (sum(splits), 480)

        # populating the combined scores and hardware costs attributes if not already done
        if not hasattr(self, "combined_scores"):
            self._set_combined_scores()
        if not hasattr(self, "hardware_costs"):
            self._set_hardware_costs()

        # Initialize Pygame window and clock if not already done
        if getattr(self, "window", None) is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(screen_size)

        if getattr(self, "clock", None) is None:
            self.clock = pygame.time.Clock()

        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        
        # 1) Draw the networks
        fig, ax = self._draw_architectures(
            *self._draw_background()
        )

        fig.canvas.draw()
        X = np.array(fig.canvas.renderer.buffer_rgba())
        architectures_rgb_array = np.array(Image.fromarray(X).convert('RGB'))
        plt.close(fig)
        
        # Transpose the frame array to get the correct orientation
        architectures_rgb_array = np.transpose(architectures_rgb_array, axes=(1, 0, 2))
        # Make a surface from the frame array
        architectures_frame = pygame.surfarray.make_surface(architectures_rgb_array)
        # rescaling the surfaces to fit screen size
        architectures_frame = pygame.transform.scale(architectures_frame, (splits[0], screen_size[1]))

        self.window.blit(architectures_frame, (0, 0))
        del fig, ax

        # 2) Draw the test-accuracy and latency bars
        fig, ax = self._draw_accuracy_latency_bars()
        fig.canvas.draw()
        X = np.array(fig.canvas.renderer.buffer_rgba())
        accuracy_latency_rgb_array = np.array(Image.fromarray(X).convert('RGB'))
        plt.close(fig)
        accuracy_latency_rgb_array = np.transpose(accuracy_latency_rgb_array, axes=(1, 0, 2))
        accuracy_latency_frame = pygame.surfarray.make_surface(accuracy_latency_rgb_array)
        accuracy_latency_frame = pygame.transform.scale(accuracy_latency_frame, (splits[1], screen_size[1]))
        self.window.blit(accuracy_latency_frame, (sum(splits[:1]), 0))
        del fig, ax

        # 3) Draw the reward bar
        fig, ax = self._draw_reward_bar()
        fig.canvas.draw()
        X = np.array(fig.canvas.renderer.buffer_rgba())
        reward_rgb_array = np.array(Image.fromarray(X).convert('RGB'))
        plt.close(fig)
        reward_rgb_array = np.transpose(reward_rgb_array, axes=(1, 0, 2))
        reward_frame = pygame.surfarray.make_surface(reward_rgb_array)
        reward_frame = pygame.transform.scale(reward_frame, (splits[2], screen_size[1]))
        self.window.blit(reward_frame, (sum(splits[:2]), 0))

        pygame.event.pump()
        # Limit the frame rate
        self.clock.tick(self.metadata["render_fps"])
        # Update the display
        pygame.display.flip()

    def render(self, mode="human"):
        """Calls the render frame method."""
        
        # Storing information associated with current observation in the buffer
        self.observations_buffer.append(self._get_info())

        return self._render_frame()
    
