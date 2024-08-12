"""General Gym Wrapper (https://www.gymlibrary.dev/api/wrappers/#general-wrappers)
    for stacking previous observations and actions in the current observation vector
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from itertools import chain
from collections import OrderedDict
from collections import deque
from copy import copy

class TransitionsHistoryWrapper(gym.Wrapper):
    def __init__(self, env:gym.Env, history_len:int=1):
            """General Gym Wrapper (https://www.gymlibrary.dev/api/wrappers/#general-wrappers)
            for stacking previous actions in the current observation vector.
            This is useful for RL-approaches benefiting from accessing an history of previous transitions to
            inform their decisions.

            Args:
                    env (VecEnv): env to be augmented with the history buffer.
                    history_len (int, optional): Number of timesteps to store information for. Defaults to 1.
            """
            super().__init__(env)
            assert isinstance(env, gym.Env), 'This wrapper only works with environments of class gym.Env'
            assert env.action_space.sample().ndim == 1, 'Actions are assumed to be flat on one-dim vector'

            # Initialize the history buffer
            self.history_len = history_len
            # The context window is the number of previous transitions to consider in the observation
            self.context_window = history_len + 1
            # This initializes the buffers as empty deque
            self.observations_deque = deque(
                 maxlen=history_len
            )
            self.actions_deque = deque(
                 maxlen=history_len
            )

            """
            The observation space is overridden to make it aware of the last history_len transitions.
            The observation space is a dictionary with the following components:
            - "architectures": Represents the self.history_len+1 architectures observed. It is a MultiDiscrete space
                with the number of discrete values equal to the number of operations in the search space.
                It allows to consider the current architecture as well as the previous ones (hence the +1 for the context
                window). Starting from -1, it allows to represent empty networks.
            - "latency_values": Represents the self.history_len + 1values of latency observed. It is a Box space
                with a shape of (self.history_len+1,).
            - "actions_performed": Represents the self.history_len actions performed. It is a MultiDiscrete space
                with the number of discrete values equal to the number of modifications in the search space +1
                (to account for the "no modification" action).
                Starting from -1, it allows to represent empty networks.
            """
            self.empty_architecture_placeholder = len(self.env.searchspace.all_ops) + 1
            self.empty_action_placeholder = self.env.searchspace.architecture_len + 2, len(self.env.searchspace.all_ops) + 1
            
            # handle to create the MultiDiscrete space for the architectures
            architectures_list = [
                                    [
                                        len(self.env.searchspace.all_ops)+1
                                        for _ in range(self.env.searchspace.architecture_len)
                                    
                                    ]
                                    for _ in range(self.context_window)
                                ]
            # handle to create the MultiDiscrete space for the actions
            actions_list = [
                [
                    (self.env.searchspace.architecture_len + 1 + 1, len(self.env.searchspace.all_ops) + 1) 
                    for _ in range(self.env.n_mods)
                ]
                for _ in range(self.history_len)  # storing all the actions up to t-1
            ]

            self.observation_space = spaces.Dict({
                # allowing for negative values allows to have empty input nodes in the policy network
                    "architecture": spaces.MultiDiscrete(
                            list(chain.from_iterable(architectures_list)),
                            # start=[-1 for _ in range(self.env.searchspace.architecture_len * self.context_window)]
                    ),
                    "latency_value": spaces.Box(low=0., high=float("inf"), shape=(self.context_window,)),
                    "action_performed": spaces.MultiDiscrete(
                            list(chain.from_iterable([val for sublist in actions_list for val in sublist])),
                            # start=[-1 for _ in range(2 * self.env.n_mods * self.history_len)]
                    )
            })

    def reset(self, **kwargs):
        """Resetting the buffers for observations and latencies"""
        for _ in range(self.history_len):
            # representing empty networks with self.empty_placeholder and with 0 latency
            self.observations_deque.appendleft(
                OrderedDict({
                    "architecture": (self.empty_architecture_placeholder - 1) * np.ones(self.env.searchspace.architecture_len, dtype=np.int64), 
                    "latency_value": 0 * np.ones(1, dtype=np.float32)
                })
            )
            # representing empty actions with choosing self.empty_placeholder 
            self.actions_deque.appendleft(
                # shape (-1, self.empty_placeholder)
                np.array(
                    [
                        (self.empty_action_placeholder[0] - 1, self.empty_action_placeholder[1] - 1) 
                        for _ in range(self.env.n_mods)
                    ], 
                    dtype=np.int64).\
                    flatten()
            )
        
        # retrieving the observation from the environment
        obs, info = self.env.reset(**kwargs)
        # stacking the initial observation to the left of the history buffer
        self.observations_deque.appendleft(copy(obs))

        return self._buffers_to_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # storing the new observation and action. At init, initial obs is stored right away
        self.observations_deque.appendleft(copy(obs))
        self.actions_deque.appendleft(action)
        
        return self._buffers_to_obs(), reward, terminated, truncated, info

    def _buffers_to_obs(self):
        """
        Returns the current buffer of observations + actions into a digestible format
        for the policy network.
        """
        # unpacking the observations in the deque into architectures and latency values
        observation = {
            "architecture": np.concatenate([o["architecture"] for o in self.observations_deque]),
            "latency_value": np.concatenate([o["latency_value"] for o in self.observations_deque]),
            "action_performed": np.concatenate(self.actions_deque)
        }

        return OrderedDict(observation)
    
    