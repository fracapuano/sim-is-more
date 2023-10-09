from nats_bench import create
from ..utils import *
from xautodl.models import get_cell_based_tiny_net
from xautodl.models.cell_infers.tiny_network import TinyNetwork
from typing import Union, Tuple, List, Set, Optional, Dict
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
import json

class NATSInterface:
    def __init__(
        self, 
        path:Optional[str]=None,
        fast_mode:bool=True,
        dataset:str="cifar10", 
        verbose:bool=False
        ):
        if path is None: 
            # load either a folder or a file according to the value of fast_mode
            path_to_nats = str(get_project_root()) + "/archive/"
            path_to_nats += "NATS-tss-v1_0-3ffb9-simple" if fast_mode else "NATS-tss-v1_0-3ffb9.pickle.pbz2"
        else:
            path_to_nats = path

        self._api = create(file_path_or_dict=path_to_nats, 
                           search_space="topology", 
                           fast_mode=fast_mode, 
                           verbose=verbose)
        
        # sanity check on the given dataset
        self.NATS_datasets = ["cifar10", "cifar100", "ImageNet16-120"]
        if dataset.lower() not in self.NATS_datasets: 
            if 'imagenet' in dataset.lower():
                dataset = 'ImageNet16-120'
            else:
                raise ValueError(f"Dataset '{dataset}' not in {self.NATS_datasets}!")
        
        self._dataset = dataset
        self.is_fast = False

    @property 
    def name(self)->str: 
        return "nats"
    
    @property
    def ordered_all_ops(self)->List[str]: 
        """NASTS Bench available operations, ordered (without any precise logic)"""
        return ['skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'none', 'avg_pool_3x3']

    @property
    def architecture_len(self)->int: 
        """Returns the number of different operations that uniquevoly define a given architecture"""
        return 6
    
    @property
    def all_ops(self)->Set[str]:
        """NASTS Bench available operations."""
        return {'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'none', 'avg_pool_3x3'}
    
    @property
    def dataset(self)->str: 
        return self._dataset
    
    @dataset.setter
    def change_dataset(self, new_dataset:str): 
        """
        Updates the current dataset with a new one. 
        Raises ValueError when new_dataset is not one of ["cifar10", "cifar100", "imagenet16-120"]
        """
        if new_dataset.lower() in self.NATS_datasets: 
            self._dataset = new_dataset
        else: 
            raise ValueError(f"New dataset {new_dataset} not in {self.NATS_datasets}")
    
    def __len__(self)->int:
        """Number of architectures in considered search space."""
        return len(self._api)
    
    def __getitem__(self, idx:int) -> TinyNetwork: 
        """Returns (untrained) network corresponding to index `idx`"""
        return self.query_with_index(idx=idx, trained_weights=False)

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
    
    def query_with_index(
        self, 
        idx:int, 
        trained_weights:bool=False, 
        return_cell_structure:bool=False) -> Union[TinyNetwork, str]: 
        """This function returns the TinyNetwork object asssociated to index `idx`. The returned network
        is either trained or not with respect to `trained_weights`. 
        Returning trained models require the additional download of 15K+ architectures, so make sure it is actually essential.

        Args:
            idx (int): Numerical index of the network to be returned.
            trained_weigths (bool, optional): Whether or not to load the state_dict for the returned network. Defaults to False.
            return_cell_structure (bool, optional): Whether or not to return the cell-structure for the considered network. Defaults to False.
        
        Returns:
            Union[TinyNetwork, str]: Either untrained or trained network corresponding to index idx. 
                                     Optionally, the string representing the network cell structure is returned too.
        """
        net_config = self._api.get_net_config(index=idx, dataset=self.dataset)
        tinynet = get_cell_based_tiny_net(config=net_config)

        if return_cell_structure:
            if trained_weights: 
                # dictionary in which the key is the random seed for training and the values are the parameters                                                         
                params = self._api.get_net_param(index=idx, dataset=self._dataset, seed=None)
                return tinynet.load_state_dict(next(iter(params.values()))), net_config["arch_str"]
            else: 
                return tinynet, net_config["arch_str"] # untrained network
        else: 
            if trained_weights:                                                       
                params = self._api.get_net_param(index=idx, dataset=self._dataset, seed=None)
                return tinynet.load_state_dict(next(iter(params.values()))) 
            else: 
                return tinynet, net_config["arch_str"] # untrained network
    
    def query_with_architecture(
        self, 
        architecture_string:str, 
        trained_weights:bool=False, 
        return_cell_structure:bool=True) -> Union[TinyNetwork, list]: 
        """This function returns the TinyNetwork object associated to `architecture_string` architecture. The returned network
        is either trained or not with respect to `trained_weights`.

        Args:
            architecture_string (str): String representing a given architecture.
            trained_weigths (bool, optional): Whether or not to load the state_dict for the returned network. Defaults to False.
            return_cell_structure (bool, optional): Whether or not to return the cell-structure for the considered network. Defaults to False.

        Returns:
            TinyNetwork: Either untrained or trained network corresponding to index idx.
        """
        if not cellstructure_isvalid(input_str=architecture_string): 
            raise ValueError(f"Architecture {architecture_string} is not valid in NATS search space!")

        architecture_idx = self._api.query_index_by_arch(arch=architecture_string)
        net_config = self._api.get_net_config(index=architecture_idx, dataset=self._dataset)
        tinynet = get_cell_based_tiny_net(config=net_config)

        if return_cell_structure:
            if trained_weights: 
                # dictionary in which the key is the random seed for training and the values are the parameters                                                         
                params = self._api.get_net_param(index=architecture_idx, dataset=self._dataset, seed=None)  # must specify `seed=None`
                return tinynet.load_state_dict(next(iter(params.values()))), net_config["arch_str"]
            else: 
                return tinynet, net_config["arch_str"]  # untrained network
        else: 
            if trained_weights:                                                     
                params = self._api.get_net_param(index=architecture_idx, dataset=self._dataset, seed=None)
                return tinynet.load_state_dict(next(iter(params.values())))
            else: 
                return tinynet

    def query_index_by_architecture(
        self, 
        architecture_string:str) -> float:
        """Query the index of an architecture in the search space.

        Args:
            architecture_string (str): string of the architecture 

        Returns:
            float: The index of the architcture
        """
        if not cellstructure_isvalid(input_str=architecture_string): 
            raise ValueError(f"Architecture {architecture_string} is not valid in NATS search space!")

        architecture_idx = self._api.query_index_by_arch(arch=architecture_string)

        return architecture_idx
    
    def query(
        self, 
        input_query:Tuple[int, str], 
        trained_weights:bool=False, 
        return_cell_structure:bool=False) -> Union[TinyNetwork, List]: 
        """This function unified query with index and query with architecture in one single `query` method.
        
        Args: 
            input_query (Tuple[int, str]): Either an integer or a string indicating, respectively, an index for the
                                           considered search space or a given cell structure
            trained_weigths (bool, optional): Whether or not to load the state_dict for the returned network. Defaults to False.
            return_cell_structure (bool, optional): Whether or not to return the cell-structure for the considered network. Defaults to False.
        
        Returns:
            Union[TinyNetwork, str]: Either untrained or trained network corresponding to index idx. Optionally, the
                                      string representing the network cell structure is returned too.
        """
        if isinstance(input_query, int): 
            return self.query_with_index(
                idx=input_query,
                trained_weights=trained_weights, 
                return_cell_structure=return_cell_structure
                )

        elif isinstance(input_query, str): 
            return self.query_with_architecture(
                architecture_string=input_query, 
                trained_weights=trained_weights, 
                return_cell_structure=return_cell_structure
                )
        else: 
            raise ValueError("{:} is not a string or an index indicating an architecture in NATS bench".format(input_query))

    def query_training_performance(self, architecture_idx:int, n_epochs:Tuple[int, int]=200) -> dict:
        """Returns accuracy, per-epoch and for n_epochs time for the training process of architecture `architecture_idx`"""
        result = dict()
        metrics = self._api.query_meta_info_by_index(
            arch_index=architecture_idx, 
            hp=str(n_epochs)
        ).get_metrics(dataset=self._dataset, setname="train")
        
        # only storing some of the metrics saved in the architecture
        result["accuracy"] = metrics["accuracy"]
        result["per-epoch_time"] = metrics["cur_time"]
        result["total_time"] = metrics["all_time"]

        return result
    
    def query_test_performance(
        self, 
        architecture_idx:int, 
        n_epochs:int=200
        ) -> dict:
        """Returns accuracy, per-epoch and for n_epochs time related to testing of architecture `architecture_idx`"""
        result = dict()
        metrics = self._api.query_meta_info_by_index(
            arch_index=architecture_idx, 
            hp=str(n_epochs)
        ).get_metrics(dataset=self._dataset, setname="ori-test")
        
        # only storing some of the metrics saved in the architecture
        result["accuracy"] = metrics["accuracy"]
        result["per-epoch_time"] = metrics["cur_time"]
        result["total_time"] = metrics["all_time"]

        return result
    
    def encode_architecture(
            self, 
            architecture_string:str,
            onehot:bool=True, 
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
                                     Defaults to True.

        Returns: 
            NDArray: Either a one-hot or integer encoded representation of a given architecture string.
        """
        if architecture_string == "": 
            return ""
        
        # turn architecture string into the list of operations in each cell
        architecture_list = architecture_to_genotype(architecture_string)
        # mapping each operation to the corresponding integer value according to the ordered ops available
        try: 
            architecture_integer = np.fromiter(
                map(lambda op: self.ordered_all_ops.index(op.split("~")[0]), architecture_list), 
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
            onehot:bool=True
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
                                     used. Defaults to True.

        Returns: 
            str: String used to actually represent the architecture currently considered.
        """
        # Find the indices of the operations optionally on the use of onehot encoding.
        indices = np.argmax(architecture_encoded, axis=1) if onehot else architecture_encoded.tolist()
        levels = ["~0", "~0", "~1", "~0", "~1", "~2"]
        # Map the indices to the corresponding operations
        architecture_list = [self.ordered_all_ops[index] + level for index, level in zip(indices, levels)]
        # Concatenate the operations to form the architecture string
        architecture_string = self.list_to_architecture(input_list=architecture_list)

        return architecture_string
        
    def generate_random_samples(
        self, 
        n_samples:int=10) -> Tuple[List[nn.Module], List[str], List[int]]:
        """Generate a group of architectures chosen at random"""
        idxs = np.random.choice(self.__len__(), size=n_samples, replace=False)
        tinynets = [self.query_with_index(i)[0] for i in idxs]
        cell_structures = [self.query_with_index(i)[1] for i in idxs]
        # return tinynets, cell_structures_string and the unique indices of the networks
        return tinynets, cell_structures, idxs
    
    def list_to_architecture(self, input_list:List[str])->str:
        """
        Reformats genotype as architecture string. 
        This function clearly is specific for this very search space.
        """
        return "|{}|+|{}|{}|+|{}|{}|{}|".format(*input_list)
    
    def list_to_accuracy(self, input_list:List[str])->float: 
        """Returns the test accuracy of an input list representing the architecture. 
        This list contains the operations.

        Args:
            input_list (List[str]): List of operations inside the architecture.

        Returns:
            float: Test accuracy (after 200 training epochs).
        """
        # list of ops -> arch string -> arch index -> accuracy score
        return self.query_test_performance(
            self.query_index_by_architecture(
                self.list_to_architecture(input_list=input_list)
                )
            )["accuracy"]
    
    def architecture_to_list(self, architecture_string:str)->List[str]: 
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
    
    def architecture_to_accuracy(self, architecture_string:str)->float:
        """Returns the test accuracy of an architecture string.

        Args:
            architecture_string (str): Architecture string.

        Returns:
            float: Test accuracy (after 200 training epochs).
        """
        return self.query_test_performance(
            self.query_index_by_architecture(
                architecture_string=architecture_string
            )
        )["accuracy"]

    @property
    def architecture_to_index(self)->Dict[str, int]: 
        """Returns the architecture <-> index hash map. Here architectures are defined as
        "/" concatenations of lists.
        
        Note: 
            This function does not check index consistency.
        
        Returns: 
            Dict[str, int]: Hash map in which each key represent an architecture and values
                            the (integer) index relative to such architecture. 
        """
        if not hasattr(self, "_architecture_to_index"):
            # search already produced index in archive.
            try: 
                file_name = "/compressed_archive/nats_arch_index.json"
                path_to_dict = str(get_project_root()) + file_name
                with open(path_to_dict, "r") as json_index:
                    self._architecture_to_index = json.load(json_index)

            except FileNotFoundError:
                print(f"{file_name} does not exist! Creating it (might take some time)...")
                
                table = {}
                for idx in tqdm(range(len(self))):
                    left_index = "/".join(self.architecture_to_list(
                        self.query_with_index(idx=idx)[1])
                    )
                    right_index = idx
                    table[left_index] = right_index
                
                with open(path_to_dict, "w") as index_file:
                    json.dump(table, index_file)
                
            # recall dump_architecture_to_index once again
            self.architecture_to_index
        else:
            return self._architecture_to_index