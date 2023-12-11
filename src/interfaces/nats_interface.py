from .base_interface import Base_Interface
from ..utils import get_project_root
from typing import (
    Text, 
    Optional, 
    Tuple,
    List
)
import json
from numpy.typing import NDArray
from itertools import chain
import numpy as np
from tqdm import tqdm


class NATS_Interface(Base_Interface):
    """
    NATS-specific search space interface.
    """
    def __init__(self, 
                 datapath:str=str(get_project_root()) + "/searchspaces/nats_blocks.json",
                 dataset:str="cifar10",
                 use_lookup_table:bool=True,
                 path_to_lookup:Optional[str]=str(get_project_root()) + "/searchspaces/nats_interface_lookuptable.json", 
                 path_to_lookup_index:Optional[str]=str(get_project_root()) + "/searchspaces/nats_arch_index.json",
                 target_device:Optional[Text]=None):
        
        # parent init, loading the datapath
        super().__init__(datapath)

        self._dataset = dataset
        self.target_device = target_device
        self.using_lookup = use_lookup_table

        if self.using_lookup and path_to_lookup is None:
            raise ValueError(
                "If using lookup table, path to lookup table must be provided! Provided: {}".format(path_to_lookup)
            )
        elif path_to_lookup is not None:
            with open(path_to_lookup, "r") as lookup_file:
                self.lookup_table = {int(k): v for k, v in json.load(lookup_file).items()}
            with open(path_to_lookup_index, "r") as lookup_index_file:
                self.architecture_to_index = json.load(lookup_index_file)

        # routing the number of classes based on datasets
        if dataset == "cifar10": 
            self.network_numclasses = 10
        elif dataset == "cifar100": 
            self.network_numclasses = 100
        elif dataset == "ImageNet16-120":
            self.network_numclasses = 120

    @property
    def dataset(self)->str:
        return self._dataset

    def __getitem__(self, index:int)->str:
        """Retrives the architecture string which is associated with a given index.

        Args:
            index (int): Numerical index, between 0 and self.__len__()

        Returns:
            str: Architecture string associated with index.
        """
        return "{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|".format(*(self.index_to_list(architecture_index=index)))
    
    def index_to_lookup_index(self, index:int)->int:
        """Retrieves the lookup index associated with a given index.

        Args:
            index (int): Numerical index, between 0 and self.__len__()

        Returns:
            int: Lookup index associated with index.
        """
        if not (index >= 0 and index <= len(self)):
            raise ValueError(f"Index out of bounds! Must be between 0 and {len(self)}.")
        
        return self.architecture_to_index[
            "{}~0/{}~0/{}~1/{}~0/{}~1/{}~2".format(*(self.index_to_list(architecture_index=index)))
        ]

    def get_config_dictionary(self, index:int)->dict:
        """Retrieves the configuration dictionary associated with a given index.
        
        Args: 
            index (int): Index associated to the architecture of interest.
        
        Returns: 
            dict: Confiugration dict (for TinyNetwork usage)
        """
        return {
            "name": "infer.tiny", 
            "C": 16,
            "N": 5, 
            "num_classes": self.network_numclasses,
            "arch_str": self[index]
        }
        
    def compute_score(self, score_name:Text, index:int)->float:
        """
        Compute the score value for the given score name and network index.

        Args:
            score_name (Text): The name of the score for which to calculate the score value.
            index (int): The index of the architecture for which to calculate the score.

        Returns:
            float: The score value.
        """
        if self.using_lookup:
            lookup_index = self.index_to_lookup_index(index=index)
            return self.lookup_table[lookup_index][self.dataset][score_name]
        else:
            # return scores_router[score_name](network_idx=index, dataset=self.dataset)
            raise NotImplementedError("Online scoring not yet implemented for NATS space.")

    def get_score_mean_and_std(self, score_name:Text, sample_size:Optional[int]=None)->Tuple[float, float]:
        """
        Calculate the mean and standard deviation of the score value across (a fraction of) th dataset 
        for the given score name.

        Args:
            score_name (Text): The name of the score for which to calculate the mean.
            sample_size (Optional[int]): The number of items to use to calculate the mean. 
                                        If None, the mean is calculated across the entire dataset. 
                                        Defaults to None.

        Returns:
            Tuple[float, float]: The mean and standard deviation of the score values.

        Note:
            The score values are retrieved from each data point in the dataset and averaged.
        """
        
        # when not using lookup table, one needs to compute the score values
        if not self.using_lookup:
            # when sample_size is None, the mean is calculated across the entire dataset
            n_networks = sample_size if sample_size is not None else len(self)

            if not hasattr(self, f"{score_name}_values"):
                # randomly sample n_networks indices, and compute the score value once only for each of them
                random_indices = np.random.choice(len(self), size=n_networks, replace=False)
                # computing the score value for each of the randomly sampled indices
                score_values = np.zeros(n_networks)
                for idx, i in tqdm(enumerate(random_indices), desc=f"Computing {score_name} values..."):
                    score_values[idx] = self.compute_score(score_name=score_name, index=i)
                
                # store these new values in the interface object
                setattr(self, f"{score_name}_values", score_values)

            # computing the mean and std of the score values
            if not hasattr(self, f"mean_{score_name}"):
                setattr(self, 
                        f"mean_{score_name}", 
                        getattr(self, f"{score_name}_values").mean()
                        )
            
            if not hasattr(self, f"std_{score_name}"):
                setattr(self, 
                        f"std_{score_name}", 
                        getattr(self, f"{score_name}_values").std()
                        )
        
        else:
            # training-free scores statistics can't be retrieved from the lookup table and must be computed instead
            if score_name in ["naswot_score", "logsynflow_score", "skip_score"]:
                if not hasattr(self, f"mean_{score_name}"):
                    setattr(self, 
                    f"mean_{score_name}", 
                    np.array([self.lookup_table[i][self.dataset][score_name] for i in range(len(self))]).mean()
                    )
                if not hasattr(self, f"std_{score_name}"):
                    setattr(self, 
                    f"std_{score_name}", 
                    np.array([self.lookup_table[i][self.dataset][score_name] for i in range(len(self))]).std()
                    )

            # when using the lookup table, the mean and std can be easily retrieved
            if not hasattr(self, f"mean_{score_name}"):
                setattr(self, 
                        f"mean_{score_name}", 
                        self._data.get(f"mean_{score_name}", -float("inf"))
                        )
            if not hasattr(self, f"std_{score_name}"):
                setattr(self, 
                        f"std_{score_name}", 
                        self._data.get(f"std_{score_name}", -float("inf"))
                        )
        
        return getattr(self, f"mean_{score_name}"), getattr(self, f"std_{score_name}")

    def generate_random_samples(self, n_samples:int=10)->Tuple[List[Text], List[int]]:
        """Generate a group of architectures chosen at random"""
        idxs = np.random.choice(len(self), size=n_samples, replace=False)
        cell_structures = [self[i] for i in idxs]
        # return tinynets, cell_structures_string and the unique indices of the networks
        return cell_structures, idxs
    
    def list_to_architecture(self, input_list:List[str])->str:
        """
        Reformats genotype as architecture string. 
        This function clearly is specific for this very search space.
        """
        return "|{}|+|{}|{}|+|{}|{}|{}|".format(*input_list)
    
    def architecture_to_list(self, architecture_string:Text)->List[Text]: 
        """Turn architectures string into genotype list

        Args: 
            architecture_string(str): String characterising the cell structure only. 

        Returns: 
            List[str]: List containing the operations in the input cell structure.
                       In a genetic-algorithm setting, this description represents a genotype. 
        """
        # divide the input string into different levels
        subcells = architecture_string.split("+")
        # divide into different nodes to retrieve ops
        ops = chain(*[subcell.split("|")[1:-1] for subcell in subcells])
        
        return list(ops)
    
    def encode_architecture(
            self, 
            architecture_string:str,
            onehot:bool=False, 
            verbose:bool=False
    )->NDArray: 
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
                                     Defaults to False.

        Returns: 
            NDArray: Either a one-hot or integer encoded representation of a given architecture string.
        """
        if architecture_string == "": 
            return ""
        
        # turn architecture string into the list of operations in each cell
        architecture_list = self.architecture_to_list(architecture_string=architecture_string)
        # mapping each operation to the corresponding integer value according to the ordered ops available
        try: 
            architecture_integer = np.fromiter(
                map(lambda op: self.all_ops.index(op.split("~")[0]), architecture_list), 
                dtype=int
            )
        except ValueError:
            if verbose:
                print(f"architecture {architecture_string} contains operations not in {self.all_ops}")
            return "conversion error"
        
        if onehot:
            # initialize a zeroed-vector
            onehot_architecture = np.zeros((architecture_integer.size, len(self.all_ops)))
            # integer encoding -> one hot encoding
            onehot_architecture[np.arange(architecture_integer.size), architecture_integer] = 1
            
            return onehot_architecture
        else:
            return architecture_integer
    
    def decode_architecture(
            self, 
            architecture_encoded:NDArray,
            onehot:bool=False
    )->str:
        """
        This function decodes the numerical representation of a given architecture, producing an
        actual architecture string.
        Each architecture is represented through an `architecture_encoded` array whose first dimension
        always is `m` (clearly enough, `m = m(searchspace)`). Optionally on `onehot`, an architecture 
        is represented through a matrix (`onehot=True`) or an array (`onehot=False`). 

        Args: 
            architecture_encoded (NDArray): Numerical representation of a given architecture.
            onehot (bool, optional): Boolean flag representing whether or not one hot encoding has been
                                     used. Defaults to False.

        Returns: 
            str: String used to actually represent the architecture currently considered.
        """
        # Find the indices of the operations optionally on the use of onehot encoding.
        indices = np.argmax(architecture_encoded, axis=1) if onehot else architecture_encoded.tolist()
        levels = ["~0", "~0", "~1", "~0", "~1", "~2"]
        # Map the indices to the corresponding operations
        architecture_list = [self.all_ops[index] + level for index, level in zip(indices, levels)]
        # Concatenate the operations to form the architecture string
        architecture_string = self.list_to_architecture(input_list=architecture_list)

        return architecture_string
    
    def list_to_accuracy(self, input_list:List[str], training_config:Optional[dict]=None)->float: 
        """Returns the test accuracy of an input list representing the architecture. 
        This list contains the operations.

        Args:
            input_list (List[str]): List of operations inside the architecture.
            training_config (Optional[dict]): Training configuration object. Defaults to None.

        Returns:
            float: Test accuracy (after 200 training epochs).
        """
        # retrieving the architecture index associated to an input list, ideally something like
        arch_index = self.architecture_to_index["/".join(input_list)]
        if self.using_lookup:
            # retrieving the index associated to this particular architecture
            return self.lookup_table[arch_index][self.dataset]["test_accuracy"]

        else:
            """
            return training_router(search_space=self, 
                                   network_idx=arch_index, 
                                   dataset=self.dataset, 
                                   training_config=training_config)
            """
            raise NotImplementedError("Online training not yet implemented for NATS space.")

    def architecture_to_accuracy(self, architecture_string:str, training_config:Optional[dict]=None)->float:
        """Returns the test accuracy of an architecture string.
        The architecture <-> index map is normalized to be as general as possible, hence some (minor) 
        input processing is needed.

        Args:
            architecture_string (str): Architecture string.
            training_config (Optional[dict]): Training configuration object. Defaults to None.

        Returns:
            float: Test accuracy (after 200 training epochs).
        """
        # retrieving the index associated to this particular architecture
        return self.list_to_accuracy(self.architecture_to_list(architecture_string), training_config=training_config)

    def list_to_score(self, input_list:List[Text], score:Text)->float:
        """Returns the value of `score` of an input list representing the architecture. 
        This list contains the operations.

        Args:
            input_list (List[Text]): List of operations inside the architecture.
            score (Text): Score of interest.

        Returns:
            float: Score value for `input_list`.
        """
        # splitting the input list to retrieve the operations only -- this is then fed into list_to_index
        levels_split = [op.split("~")[0] for op in input_list]
        return self.compute_score(score_name=score, index=self.list_to_index(architecture_list=levels_split))

    def architecture_to_score(self, architecture_string:Text, score:Text)->float:
        """Returns the value of `score` of an architecture string.
        The architecture <-> index map is normalized to be as general as possible, hence some (minor) 
        input processing is needed.

        Args:
            architecture_string (Text): Architecture string.
            score (Text): Score of interest.

        Returns:
            float: Score value for `architecture_string`.
        """
        # retrieving the architecture list out of the architecture string
        architecture_list = self.architecture_to_list(architecture_string=architecture_string)

        return self.compute_score(score_name=score, index=self.list_to_index(architecture_list=architecture_list))

