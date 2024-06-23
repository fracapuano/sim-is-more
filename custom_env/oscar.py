import pygame
from .render import (
    create_background_scatter,
    create_background_vlines,
    draw_architectures_on_background,
    draw_hbars,
    figure_to_image,
    is_pareto_efficient
)
import numpy as np
from itertools import chain
from .reward import Rewardv1 as Reward
from .nas_env import NASEnv
from gymnasium import spaces
from collections import deque
from src import Base_Interface
import matplotlib.pyplot as plt
from operator import itemgetter
from numpy.typing import NDArray
from .utils import NASIndividual
from scipy.stats import percentileofscore
from typing import Iterable, Text, Tuple, Dict, Optional, Union
from warnings import warn


class OscarEnv(NASEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
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
        # setting the target device, env and searchspace-wise
        self.target_device = target_device
        self.searchspace.target_device = target_device
        # casting weights to numpy array
        self.weights = np.array(weights)
        # initializing the reward handler -- this moves reward computation out of the environment
        self.reward_handler = Reward(searchspace=self.searchspace, weights=self.weights)
        # initializing the observations buffer size, for rendering purposes
        self.observations_buffer_size = observation_buffer_size
        self.observations_buffer = deque(maxlen=self.observations_buffer_size)
        
        # initializing a pool of networks to be used for distributional-insights purposes
        self.init_networks_pool(n_samples=n_samples)
        # instantiates the reference architecture (to analyze differences in hardware costs across devices)
        self.init_reference_architecture()
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
            (self.searchspace.architecture_len, len(self.searchspace.all_ops))
            for _ in range(self.n_mods)])
        )
        self.action_space = spaces.MultiDiscrete(action_space)
        
        # init render mode
        self.render_mode = "rgb_array"

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

    def init_reference_architecture(self)->None:
        """This method initializes the reference architecture for the environment.
        """
        self.reference_architecture = np.random.choice(self.networks_pool)
    
    def get_reference_architecture(self, architecture_list:bool=True)->Union[Text, Iterable[Text]]:
        """This method returns the reference architecture for the environment.
        
        Returns:
            Union[Text, Iterable[Text]]: The reference architecture, either as a string or in list format.
        """
        return self.reference_architecture if not architecture_list else self.searchspace.architecture_to_list(self.reference_architecture)

    def fitness_function(self, individual:NASIndividual)->NASIndividual: 
        """
        Directly overwrites the fitness attribute for a given individual.

        Args: 
            individual (NASIndividual): Individual to score.

        Returns:
            NASIndividual: Individual, with fitness field.
        """
        warn("This method is deprecated. Use the fitness_function method from the reward_handler attribute!", DeprecationWarning, stacklevel=2)
        
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
        warn("This method is deprecated. Use the fitness_function method from the reward_handler attribute!", DeprecationWarning, stacklevel=2)

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
        warn("This method is deprecated. Use the fitness_function method from the reward_handler attribute!", DeprecationWarning, stacklevel=2)

        return self.searchspace.list_to_score(input_list=architecture_list, score=f"{self.target_device}_latency")

    def _get_obs(self)->Dict[Text, spaces.Space]:
        return self._observation
    
    def _get_info(self)->dict: 
        """Return the info dictionary."""
        current_net_latency = self._get_obs()["latency_value"].item()

        info_dict = {
            "current_network": self.current_net.architecture,
            "timestep": self.timestep_counter,
            "reward": self.reward_handler.get_reward(individual=self.current_net),
            "current_net_latency": current_net_latency,
            "current_net_latency_percentile": percentileofscore(self.hardware_costs, current_net_latency),
            "latency_cutoff": self.max_latency,
            "is_terminated": self.is_terminated(),
            "is_truncated": self.is_truncated(),
            "networks_seen": len(self.networks_seen),
            
            # test_accuracy is obtained from a lookup table and never accessed during training
            "test_accuracy": self.searchspace.list_to_accuracy(input_list=self.current_net.architecture),
        }
        info_dict |= self.reward_handler.get_individual_scores(individual=self.current_net)
        
        return info_dict

    def is_terminated(self)->bool:
        """
        Returns `True` if the episode is terminated and `False` otherwise.
        Episodes are terminated when an agent produces an architecture with a latency value greater than the cutoff.

        May be setted to always return False when no termination condition is needed.
        """
        # return bool(self.current_net_latency > self.max_latency)
        return False

    def get_reward(self, individual:NASIndividual)->float:
        """
        Compute the reward associated to the modification operation.
        Here, the reward is the fitness of the newly generated individual
        """
        return self.reward_handler.get_reward(individual=individual)

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
        # compute the reward associated with producing reinforced_individual
        reward = self.get_reward(individual=reinforced_individual)
        
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

        if terminated:
            reward = -1

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
    
    def plot_networks_on_searchspace(
            self, 
            terminal_networks:list[NASIndividual], 
            fitness_color:bool=True)->plt.axis:
        """
        Creates a visualization of the terminal networks plotted on the whole searchspace.

        Args:
            terminal_networks (list[NASIndividual]): The terminal networks to plot.
            fitness_color (bool, optional): Whether to color the networks based on their fitness. Defaults to True.
        
        Returns:
            plt.axis: The axis of the plot.
        """
        if not isinstance(terminal_networks, list):
            # turning iterable into list to allow multiple iterations
            terminal_networks = list(terminal_networks)
            
        if not all(map(lambda x: isinstance(x, NASIndividual), terminal_networks)):
            raise ValueError("Input networks are not all of NASIndividual type!")
        
        def get_individual_coordinates(individual:NASIndividual)->Tuple[float, float]:
            performance_score = self.reward_handler.get_performance_score(individual=individual)
            efficiency_score = self.reward_handler.get_efficiency_score(individual=individual)

            return (efficiency_score, performance_score)
        
        _, ax = plt.subplots(dpi=150)

        fitness = None  # override, if fitness_color
        if fitness_color:
            fitness = list(
                map(
                    lambda n: self.reward_handler.\
                        fitness_function(
                            self.architecture_to_individual(n)
                        ).fitness,
                    self.networks_pool
                    )
                )
        
        ax = create_background_scatter(
            ax, 
            *zip(*map(lambda n: get_individual_coordinates(self.architecture_to_individual(n)), 
                      self.networks_pool)),
            c=fitness
        )

        # This plot the networks on the background 
        ax.scatter(
            *zip(*map(get_individual_coordinates, terminal_networks)),
            c="red",
            s=25,
            marker="X"
        )

        return ax

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
        if self.normalization_type == "std":
            # computing the mean and std of the scores for the network pool considered   
            self.scores_mean, self.scores_std = np.vstack(
                [self.training_free_scores.mean(axis=1), self.training_free_scores.std(axis=1)]
            )
            
            # normalizing the scores using mean and std
            self.normalized_training_free_scores = (self.training_free_scores - self.scores_mean.reshape(-1,1)) / self.scores_std.reshape(-1,1)

        elif self.normalization_type == "minmax":
            # computing the min and max of the scores for the network pool considered
            self.scores_min, self.scores_max = np.vstack(
                [self.training_free_scores.min(axis=1), self.training_free_scores.max(axis=1)]
            )
            
            # normalizing the scores using min and max
            self.normalized_training_free_scores = (self.training_free_scores - self.scores_min.reshape(-1,1)) / (self.scores_max.reshape(-1,1) - self.scores_min.reshape(-1,1))
        
        # combining the scores using the combination coefficients
        self.combined_scores = scores_combination_coeffs @ self.normalized_training_free_scores

    def _set_hardware_costs(self):
        """
        Sets the hardware costs based on the latency of the networks within the network pool on the target device.
        """
        self.hardware_costs = np.fromiter(
            map(lambda a: self.compute_hardware_cost(self.searchspace.architecture_to_list(a)), self.networks_pool), 
            dtype="float"
        )
        if self.normalization_type == "std":
            # extracting the mean and std of the scores
            self.hw_score_mean, self.hw_score_std = self.hardware_costs.mean(), self.hardware_costs.std()
            
            # normalizing the scores using mean and std
            self.normalized_hardware_costs = (self.hardware_costs - self.hw_score_mean) / self.hw_score_std
        
        elif self.normalization_type == "minmax":
            # extracting the min and max of the scores
            self.hw_score_min, self.hw_score_max = self.hardware_costs.min(), self.hardware_costs.max()
            
            # normalizing the scores using min and max
            self.normalized_hardware_costs = (self.hardware_costs - self.hw_score_min) / (self.hw_score_max - self.hw_score_min)
    
    def _set_test_accuracies(self):
        """
        Sets the test accuracy of all the networks within the network pool considered.
        """
        self.test_accuracies = np.fromiter(
            map(lambda a: self.searchspace.architecture_to_accuracy(a), self.networks_pool), 
            dtype="float"
        )

    def unpack_buffer(self, x_getter:str="current_net_latency", y_getter:str="training_free_score")->Tuple[NDArray, NDArray]:
        """
        Unpack the observations buffer into x and y coordinates.

        Args:
            x_getter (str, optional): The name of the attribute to use as x coordinate. Defaults to "current_net_latency".
            y_getter (str, optional): The name of the attribute to use as y coordinate. Defaults to "training_free_score".

        Returns:
            Tuple[NDArray, NDArray]: The x and y coordinates.
        """
        x_coordinates = np.fromiter(
            map(lambda info: info[x_getter], self.observations_buffer),
            dtype="float"
        )
        y_coordinates = np.fromiter(
            map(lambda info: info[y_getter], self.observations_buffer),
            dtype="float"
        )

        return x_coordinates, y_coordinates

    def _render_frame(self, 
                      mode:Text="human", 
                      draw_background:bool=True,
                      use_accuracy:bool=False)->Optional[NDArray]: 
        """
        Renders a frame of the environment.

        Args:
            mode (Text, optional): The rendering mode. Defaults to "human".
            draw_background (bool, optional): Whether to draw the background scatter plot. Defaults to True.
            use_accuracy (bool, optional): Whether to use the test accuracy as y coordinate. Defaults to False.

        Returns:
            Optional[NDArray]: The rendered frame as an RGB array if mode is "rgb_array", None otherwise.
        """ 
        screen_size = (640*1.5, 480*1.5)

        if not use_accuracy:
            # populating the combined scores and hardware costs attributes if not already done
            if not hasattr(self, "combined_scores") and draw_background:
                self._set_combined_scores()
        else:
            if not hasattr(self, "test_accuracies") and draw_background:
                self._set_test_accuracies()
        
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
        
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, gridspec_kw={"height_ratios": [8,1,2]}, dpi=400, layout="constrained")
        
        y_getter = "test_accuracy" if use_accuracy else "training_free_score"
        y_label = f"Test Accuracy, {self.searchspace.dataset}" if use_accuracy else "Combined Training-Free Score"
        performance_measures = self.test_accuracies if use_accuracy else self.combined_scores

        if draw_background:
            # coloring points based on fitness value
            fitness = list(map(lambda n: self.reward_handler.fitness_function(self.architecture_to_individual(n)).fitness, self.networks_pool))
            ax1 = create_background_scatter(ax1, self.hardware_costs, performance_measures, c=fitness)
        
        architectures_and_costs = np.hstack((self.hardware_costs.reshape(-1,1), -1 * performance_measures.reshape(-1,1)))
        # computing the Pareto front
        pareto_front = architectures_and_costs[is_pareto_efficient(architectures_and_costs)]
        # sorting by increasing values of latency
        pareto_front = pareto_front[pareto_front[:, 0].argsort()]
        
        # drawing the Pareto front
        ax1.plot(
            pareto_front[:,0], -1 * pareto_front[:,1], zorder=1, lw=2, ls="--", color="tab:red", label="Pareto Front"
        )
        ax1.set_title(f"Latency vs. Performance Indicator (Weights: {self.weights})")

        # drawing the latency cutoff
        if self.cutoff_percentile != 100:
            ax1 = create_background_vlines(ax1, self.max_latency, label="Latency Cutoff")
        
        # drawing the architectures
        ax1 = draw_architectures_on_background(
            ax1,
            *self.unpack_buffer(y_getter=y_getter), 
            label="Current Network", 
            zorder=2
        )
        ax1.set_xlabel("Latency (ms)"); ax1.set_ylabel(y_label)

        # removing legend
        ax1.get_legend().remove()
        
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
        ax3 = draw_hbars(ax3,
                        [r"$r_t$", "efficiency_score(a)", "performance_score(a)"], 
                        [
                            info_dict["reward"], 
                            info_dict["reward_efficiency_score"], 
                            info_dict["reward_performance_score"]
                        ],
                        height=0.5)
        #ax3.set_xlim(-1, 1)
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
        return self._render_frame(
            mode=self.render_mode, 
            draw_background=draw_background, 
            use_accuracy=getattr(self, "rendering_test_mode", False)
        )

