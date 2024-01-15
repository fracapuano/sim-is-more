from .oscar import OscarEnv
from itertools import cycle
from src import Base_Interface
from numpy.typing import NDArray
from typing import Iterable, Text, Dict, Optional

class MarcellaEnv(OscarEnv): 
    """
    gym.Env for Multi-task Hardware-aware pure-RL-based NAS. Architectures are evaluated using training-free metrics 
    only as well as specific hardware related metrics. During the training procedure, the target device the agent
    interacts with is updated to force the agent to solve the task independently on the actual device.
    """
    def __init__(self, 
                 searchspace_api:Base_Interface,
                 cutoff_percentile:float=85.,
                 target_device:Text="edgegpu",
                 latency_cutoff:Optional[float]=None,
                 devices:Iterable[Text]=["edgegpu", "eyeriss", "raspi4"],
                 n_samples:Optional[int]=None,
                 **kwargs):

        super().__init__(
            searchspace_api=searchspace_api,
            cutoff_percentile=cutoff_percentile,
            latency_cutoff=latency_cutoff,
            target_device=target_device,
            n_samples=n_samples,
            **kwargs
        )
        """
        This argument stores the list of devices used for multi-tasking training and the respective latency cutoffs. 
        Each different device indeed has diverse distributions for what concern the best latency performance.
        """
        self.device_freeze = True
        self.next_device = None  # this will be set to some device at the first multitask callback call
        self.devices = devices

    @property
    def name(self): 
        return "marcella"
    
    def get_target_device(self):
        """Returns the current target device."""
        return self.target_device
    
    def get_devices(self):
        """Returns the list of devices used for multitasking."""
        return self.devices

    def change_device(self):
        """
        Change the target device based on random selection if device freeze is not enabled.

        Returns:
            None

        Note:
            The method randomly selects a new device from the available devices and updates the target
            device and latency cutoff accordingly. This only happens if device freeze is not enabled.

        """
        if not self.device_freeze:
            self.target_device = self.next_device
            self.max_latency = self.get_max_latency(self, percentile=self.cutoff_percentile)
            # entered the loop because device freeze was False, switch sets it to True
            self.device_freeze = True

    def set_next_device(self, next_device:Text):
        """
        Set the next device to be used without changing the current device.

        Args:
            next_device (Text): The next device to be set.

        Returns:
            None

        Note:
            The method sets the next device to be used without changing the current device. 
            The next device can be used for future operations or as a reference for device switching.

        """
        self.next_device = next_device

    def switch_device_freeze(self):
        """
        Toggle the device freeze mode.

        Note:
            The method toggles the device freeze mode. If it was previously enabled, it will be disabled, 
            and vice versa.

        """
        self.device_freeze = not self.device_freeze
    
    def reset(self, seed:Optional[int]=None)->NDArray:
        """Resets custom env attributes."""

        self._observation = self.observation_space.sample()
        self.change_device()
        
        # clearing the buffer of observation collected
        self.observations_buffer.clear()
        # updating the current network
        self.update_current_net()
        # recomputing the hardware costs after the device switch
        self._set_hardware_costs()
        # resetting the maximal latency allowed
        self.max_latency = self.get_max_latency()

        self.timestep_counter= 0

        return self._get_obs(), self._get_info()
    
