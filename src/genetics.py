from typing import Callable, List, Dict
from .network_utils import *
from abc import abstractmethod, abstractproperty


class BaseIndividual: 
    """
    Base Individual class, just for typing purposes. 
    Implements the minimal methods that should be provided when dealing with individuals.
    """
    @abstractmethod
    def genotype(self):
        raise NotImplementedError("Abstract class!")
    
    @abstractmethod
    def update_idx(self): 
        raise NotImplementedError("Abstract class!")

    @abstractmethod
    def update_genotype(self): 
        raise NotImplementedError("Abstract class!")
    
    @abstractproperty
    def fitness(self): 
        raise NotImplementedError("Abstract class!")
    
    @abstractmethod
    def update_fitness(self, metric:Callable, attribute:str="net"): 
        raise NotImplementedError("Abstract class!")


class Individual(BaseIndividual): 
    def __init__(
        self,
        genotype:List[str],
        architecture_to_index:Dict[str, int],
        index:int, 
        age:int=0):
        
        self._genotype = genotype
        self.index=index
        self.age = age

        self._fitness = None
        self.architecture_to_index = architecture_to_index
    
    @property
    def genotype(self): 
        return self._genotype
    
    @property
    def fitness(self): 
        return self._fitness
    
    def update_idx(self):
        self.index = self.architecture_to_index["/".join(self._genotype)]

    def update_genotype(self, new_genotype:List[str]): 
        """Update current genotype with new one. When doing so, also the network field is updated"""
        self._genotype = new_genotype
        self.update_idx()
    
    def update_fitness(self, metric:Callable, attribute:str="net"): 
        """Update the current value of fitness using provided metric"""
        self._fitness = metric(getattr(self, attribute))
    
    def overwrite_fitness(self, new_fitness:float):
        """Overwrite current value of fitness"""
        if isinstance(new_fitness, float) or isinstance(new_fitness, int): 
            self._fitness = new_fitness
        else: 
            raise ValueError(f"New fitness value ({new_fitness}) is not a number!")

