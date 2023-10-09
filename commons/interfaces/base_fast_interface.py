from .base_interface import BaseInterface
from abc import abstractmethod, abstractproperty
import json
from typing import (
    List, 
    Text, 
    Dict, 
    Tuple
)
from numpy.typing import NDArray

class Base_FastInterface(BaseInterface):
    """
    Base class for Fast Search Space Interface.
    This class defines the core methods that child classes should be overriding.
    """
    def __init__(self, datapath:str, indexpath:str):
        # importing the performance data reading it from a json file.
        with open(datapath, "r") as datafile:
            self._data = {
                int(key): value for key, value in json.load(datafile).items()
            }
        # importing the "/"-architecture <-> index from a json file
        with open(indexpath, "r") as indexfile:
            self._architecture_to_index = json.load(indexfile)

        self.is_fast = True
    
    def __len__(self)->int:
        """Number of architectures in considered search space."""
        return len(self._data)
    
    def __getitem__(self, idx:int) -> Dict: 
        """Returns (untrained) network corresponding to index `idx`"""
        return self._data[idx]

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
    
    @property
    def data(self):
        return self._data

    @property
    def architecture_to_index(self):
        return self._architecture_to_index
    
    @property 
    @abstractproperty
    def name(self)->Text: 
        raise NotImplementedError("Abstract property!")

    @property
    @abstractproperty
    def architecture_len(self): 
        raise NotImplementedError("Abstract property!")
    
    @property
    @abstractproperty
    def all_ops(self): 
        raise NotImplementedError("Abstract property!")
    
    @property
    @abstractproperty
    def ordered_all_ops():
        raise NotImplementedError("Abstract property!")
    
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
                            onehot:bool=False,
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
            onehot (bool, optional): Boolean flag representing whether or not to use one hot encoding. 
                                     Defaults to True.

        Raises:
            NotImplementedError: This is an abstract method!
                                     
        Returns: 
            NDArray: Either a one-hot or integer encoded representation of a given architecture string.
        """
        raise NotImplementedError("Abstract method!")

    @abstractmethod
    def decode_architecture(
            self, 
            architecture_encoded:NDArray,
            onehot:bool=False
    )->Text:
        """
        This function decodes the numerical representation of a given architecture, producing an
        actual architecture string.
        Each architecture is represented through an `architecture_encoded` array whose first dimension
        always is `m` (clearly enough, `m = m(searchspace)`). Optionally on `onehot`, an architecture 
        is represented through a matrix (`onehot=True`) or an array (`onehot=False`). 

        Args: 
            architecture_encoded (NDArray): Numerical representation of a given architecture.
            onehot (bool, optional): Boolean flag representing whether or not one hot encoding has been
                                     used. Defaults to True.

        Raises:
            NotImplementedError: This is an abstract method!
        
        Returns: 
            str: String used to actually represent the architecture currently considered.
        """
        raise NotImplementedError("Abstract method!")
    
    @abstractmethod
    def generate_random_samples(self, n_samples:int=20)->Tuple[List[Text], List[int]]:
        """
        This function generate a random subset of architecture_lists alongside their indices.
        
        Args:
            n_samples (int, optional): Number of random architectures sampled.
        
        Returns: 
            Tuple[List[Text], List[int]]: (Architecture list, index) tuple.
        """
        raise NotImplementedError("Abstract method!")