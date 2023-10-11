from abc import abstractmethod, abstractproperty, ABC
import json
from typing import (
    List, 
    Text, 
    Dict, 
    Tuple
)
from numpy.typing import NDArray
from abc import ABC
from .utils import validate_search_space

class Base_Interface(ABC):
    """
    Base class for Search Space Interface.
    This class defines the core methods that child classes should be overriding.
    """
    def __init__(self, searchspace_info:str):
        with open(searchspace_info, "r") as searchspace_file:
            self._data = json.load(searchspace_file)

        # this validates the input search space file to make sure it contains all the required keys
        validate_search_space(self._data)

    def contains(self, architecture_list:List[Text])->bool:
        """This function checks whether or not a given architecture is contained
        in the search space.

        Args:
            architecture_list (List[Text]): Architecture list.
        
        Raises:
            ValueError: If the input architecture is not a list. Use the `list_to_architecture` method
                        to convert a string to a list.

        Returns:
            bool: True if the architecture is contained, False otherwise.
        """
        if not isinstance(architecture_list, list):
            msg = """
            Input architecture must be a list to check if it is contained in searchpace! \n 
            Use the `list_to_architecture` method to convert a string to a list.
            """
            raise ValueError(msg)
        
        return all([op in self.all_ops for op in architecture_list])

    @property
    def name(self)->Text: 
        """Name of the considered search space."""
        return self._data["name"]

    def __len__(self)->int:
        """Number of architectures in considered search space."""
        return self._data["number_of_architectures"]
    
    def blocks_data(self, device:str, metric:str):
        return self._data[f"{device}_{metric}"]
    
    @property
    def architecture_len(self): 
        return self._data["architecture_length"]

    @property
    def all_ops(self): 
        return self._data["operations"]
    
    def __iter__(self):
        """Iterator method"""
        self.iteration_index = 0
        return self

    def __next__(self):
        if self.iteration_index >= self.__len__():
            raise StopIteration
        # access current element 
        net = self[self.iteration_index]
        # update the iteration index
        self.iteration_index += 1
        return net
    
    @abstractmethod
    def __getitem__(self, idx:int) -> Dict: 
        raise NotImplementedError("Abstract method!")
    
    @abstractmethod
    def list_to_accuracy(self, input_list:List[Text])->float:
        """This function returns the (test) accuracy related to the
        architecture represented with `input_list`.

        Args:
            input_list (List[Text]): Architecture string, represented as list.

        Raises:
            NotImplementedError: This is an abstract method!

        Returns:
            float: Test accuracy!
        """
        raise NotImplementedError("Abstract method!")
    
    @abstractmethod
    def list_to_architecture(self, input_list:List[Text])->Text:
        """This function maps an architecture list to the corresponding
        architecture string.

        Args:
            input_list (List[Text]): Architecture list.

        Raises:
            NotImplementedError: This is an abstract method!

        Returns:
            Text: Architecture string.
        """
        raise NotImplementedError("Abstract method!")
    
    @abstractmethod
    def architecture_to_list(self, architecture_string:Text)->List[Text]:
        """This function maps an architecture string to the corresponding
        architecture list.

        Args:
            input_list (Text): Architecture string.

        Raises:
            NotImplementedError: This is an abstract method!

        Returns:
            List[Text]: Architecture list.
        """
        raise NotImplementedError("Abstract method!")

    @abstractmethod
    def encode_architecture(self, 
                            architecture_string:Text,
                            verbose:bool=False)->NDArray:
        """
        This function represents a given architecture string with a numerical
        array. 
        Each architecture is represented through an `architecture_string` of lenght `m` (clearly 
        enough, `m = m(searchspace)`). Each operation in the base cell can be any of the `n` ops 
        in defined at the search-space level. In light of this, each individual can be represented 
        via a (very sparse) `m x n` array `{0,1}^{m x n}`.

        Args: 
            architecture_string (str): String used to actually represent the architecture currently 
                                       considered.

        Raises:
            NotImplementedError: This is an abstract method!
                                     
        Returns: 
            NDArray: Integer encoded representation of a given architecture string.
        """
        raise NotImplementedError("Abstract method!")

    @abstractmethod
    def decode_architecture(
            self, 
            architecture_encoded:NDArray)->Text:
        """
        This function decodes the numerical representation of a given architecture, producing an
        actual architecture string.
        Each architecture is represented through an `architecture_encoded` array whose first dimension
        always is `m` (clearly enough, `m = m(searchspace)`). Optionally on `onehot`, an architecture 
        is represented through a matrix (`onehot=True`) or an array (`onehot=False`). 

        Args: 
            architecture_encoded (NDArray): Numerical representation of a given architecture.
            
        Raises:
            NotImplementedError: This is an abstract method!
        
        Returns: 
            str: String used to actually represent the architecture currently considered.
        """
        raise NotImplementedError("Abstract method!")
    
    @abstractmethod
    def generate_random_samples(self, n_samples:int=20)->List[Tuple[List[Text], List[int]]]:
        """
        This function generate a random subset of architecture_lists alongside their indices.
        
        Args:
            n_samples (int, optional): Number of random architectures sampled.
        
        Returns: 
            List[Tuple[List[Text], List[int]]]: (Architecture list, index) tuple.
        """
        raise NotImplementedError("Abstract method!")
