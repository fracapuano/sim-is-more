import random
import argparse
from typing import Union, Optional, List, Callable
from src.interfaces import Base_Interface, NATS_Interface


def boolean_string(s)->bool:
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'

def float_range(min:float, max:float)->Callable:
    """Return function handle of an argument type function for 
       ArgumentParser checking a float range: mini <= arg <= maxi
         mini - minimum acceptable argument
         maxi - maximum acceptable argument"""

    # Define the function with default arguments
    def float_range_checker(arg):
        """New Type function for argparse - a float within predefined range."""
        try:
            f = float(arg)
        except ValueError:  
            raise argparse.ArgumentTypeError("must be a floating point number")
        if f < min or f > max:
            raise argparse.ArgumentTypeError("must be in the closed interval [" + str(min) + "..." + str(max)+"]")
        return f

    # Return function handle to checking function
    return float_range_checker

def create_searchspace(searchspace:str, dataset:str)->Base_Interface:
    """
    Creates a search space based on the given search space type and dataset.

    Args:
        searchspace (str): The type of search space to create.
        dataset (str): The dataset to use for creating the search space.

    Returns:
        Base_Interface: An instance of the search space interface.

    Raises:
        NotImplementedError: If the specified search space is not implemented yet.
    """
    if searchspace.lower() == "nats":
        return NATS_Interface(dataset=dataset)
    else:
        raise NotImplementedError(
            f"Searchspace {searchspace} not implemented yet. Searchspaces that will be implemented: ['nats', 'fbnet']. FBNet to do."
        )
    
def get_distribution_devices(available_devices:List[str], chosen_devices:Optional[Union[List[str], int]])->List[str]:
    """Get the list of devices to be used for the distribution of the Marcella(+) model.

    Args:
        available_devices (List[str]): The list of devices available for the model.
        chosen_devices (List[str]): The list of devices to be used for the distribution.

    Returns:
        List[str]: The list of devices to be used for the distribution.
    """
    if chosen_devices is None:
        return available_devices
    elif isinstance(chosen_devices, int):
        return random.sample(available_devices, k=chosen_devices)
    elif isinstance(chosen_devices, list):
        intersection_devices = set(chosen_devices).intersection(available_devices)
        if len(intersection_devices) == len(chosen_devices):
            return chosen_devices
        else:
            msg = f"Passed devices: {chosen_devices} / Available devices: {available_devices}"
            print("***Warning***"+msg)
            print("The chosen_devices argument contains devices that are not available!")
            return list(intersection_devices)
    else:
        raise ValueError("The chosen_devices argument must be either None, a list of devices or an integer.")
