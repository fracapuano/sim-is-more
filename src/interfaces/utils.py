from typing import Dict

def validate_search_space(searchspace_dict:Dict)->bool:        
    """
    This function performs checks on the search space dictionary used within searchspaces.

    Args: 
        searchspace_dict (Dict): Search space dictionary.
    
    Raises:
        ValueError: If the search space dictionary does not contain all the required keys.
                    Required keys are:
                    - name
                    - devices
                    - metrics
                    - operations
                    - architecture_length
                    - number_of_architectures

    Returns:
        bool: True if all the required keys are present, None otherwise (raises error)
    """
    """This function performs some basic validations on the search space file."""
    name_is_here = "name" in searchspace_dict
    devices_is_here = "devices" in searchspace_dict
    metrics_is_here = "metrics" in searchspace_dict
    operations_is_here = "operations" in searchspace_dict
    architecture_length_is_here = "architecture_length" in searchspace_dict
    number_of_architectures_is_here = "number_of_architectures" in searchspace_dict

    conditions = [
        name_is_here,
        devices_is_here,
        metrics_is_here,
        operations_is_here,
        architecture_length_is_here,
        number_of_architectures_is_here
    ]

    if all(conditions):
        return True
    else:
        print("Invalid search space file! Make sure the interface file is correct!")
        print("""
                Please ensure you have the following keys in the json input file: 
                \n\t- name
                \n\t- devices
                \n\t- metrics
                \n\t- operations
                \n\t- architecture_length
                \n\t- number_of_architectures
                """)
        raise ValueError("Invalid search space file! Make sure the interface file is correct!")


class Device:
    """Typing-only class for device objects"""

     