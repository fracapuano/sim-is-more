from typing import Dict, List

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


def list_to_base_n_integer(n:int, base_n_list:List[int])->int:
    """Convert a list in which each element take values up to n to a base-n integer value.
    EXAMPLE: 
    n, input_list = 5, [4, 3, 2, 1, 0, 4]
    output = 4 * 5^0 + 3 * 5^1 + 2 * 5^2 + 1 * 5^3 + 0 * 5^4 + 4 * 5^5 = 3124

    Args:
        n (int): Base of the integer value.
        base_n_list (List[int]): List of integers to convert.
    
    Returns:
        int: Base-n integer value.
    """
    integer_value = sum(val * (n ** idx) for idx, val in enumerate(base_n_list[::-1]))
    return integer_value


def base_n_integer_to_list(integer_value:int, n:int, list_length:int)->List[int]:
    """Convert an integer to its base-n list representation. This function serves as the inverse
    of `encode_base_n_list`.
    EXAMPLE:
    n, integer_value, list_length = 5, 3124, 6
    output = [4, 3, 2, 1, 0, 4]

    Args:
        integer_value (int): Integer value to convert.
        n (int): Base of the integer value.
        list_length (int): Length of the output list.
    
    Returns:
        List[int]: Base-n list representation of the input integer value.
    """
    base_n_list = []
    
    while integer_value > 0:
        remainder = integer_value % n
        base_n_list.append(remainder)
        integer_value //= n

    # If the number is shorter than expected, pad with zeros
    while len(base_n_list) < list_length:
        base_n_list.append(0)

    # The list is constructed in reverse order, so reverse it
    return base_n_list[::-1]