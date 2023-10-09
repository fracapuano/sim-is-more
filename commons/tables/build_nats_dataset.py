"""
This script builds the NATS Interface dataset, dumping data from Nats Bench (https://arxiv.org/abs/2009.00437)
The dataset created has a dictionary-like structure. The metrics that have been presented in FreeREA are also
contained in this file.

The output of this script is a .json file that can be imported as data interface file. This is done in the
sake of obtaining a procedure with which accessing API information quickly enough to actually be able to leverage
Deep RL for Neural Architecture Search.
"""
from nats_bench import create
import json
import os
from tqdm import tqdm
from ..utils import get_project_root
import numpy as np

def main():
    """
    The archive folder stores: 
        1. The NATS api file (both for the fast and not fast versions).
        2. The data necessary to compute the training-free metrics without having to
           download the whole cifar10/cifar100/ImageNet16-120 datasets.
        3. The pre-computed training-free metrics for the NATS search-space. 
    
    *Note*: 
    In the following code we will use the slight abuse of notation: imagenet ~ ImageNet16-120
    But, clearly enough, we will always be referring to the downsampled and at lower-resolution
    version of the ImageNet dataset.
    """
    datasets = ["cifar10", "cifar100", "imagenet"]
    # this lambda function is only used to account for the difference in naming of "ImageNet16-120"
    convert_dataset = lambda d: "ImageNet16-120" if "imagenet" in d else d

    path_to_archive = str(get_project_root()) + "/archive/"
    path_to_nats = path_to_archive + "NATS-tss-v1_0-3ffb9-simple"
    path_to_metrics = path_to_archive + "cachedmetrics/"
    # loads to memory performance metrics
    cifar10_filename = path_to_metrics + "cifar10_cachedmetrics.txt"
    cifar100_filename = path_to_metrics + "cifar100_cachedmetrics.txt"
    imagenet_filename = path_to_metrics + "imagenet_cachedmetrics.txt"
    cifar10_metrics = np.loadtxt(cifar10_filename, skiprows=1)
    cifar100_metrics = np.loadtxt(cifar100_filename, skiprows=1)
    imagenet_metrics = np.loadtxt(imagenet_filename, skiprows=1)
    dataset_to_metrics = {
        "cifar10": cifar10_metrics, 
        "cifar100": cifar100_metrics, 
        "ImageNet16-120": imagenet_metrics
    }

    # creates a NATS api to iterate through
    api = create(
        file_path_or_dict=path_to_nats, 
        search_space="topology", 
        fast_mode=True, 
        verbose=False)
    data = {}
    """The data will be stored in a structure like:
    data = {
        Index (int): {
            architecture_string: str,
            dataset (str): 
                {
                    naswot_score: float, 
                    logsynflow_score: float,
                    skip_score: float, 
                    test_accuracy: float
                }
        }
    }
    """
    for index in tqdm(range(len(api))):
        data[index] = {
            # at this level the dataset does not really play a role on the output
            "architecture_string": api.get_net_config(index=index, 
                                                        dataset="cifar10")["arch_str"]
        }
        for dataset in datasets:
            dataset = convert_dataset(dataset)
            data[index][dataset] = {
                        "naswot_score": dataset_to_metrics[dataset][index, 1],
                        "logsynflow_score": dataset_to_metrics[dataset][index, 2], 
                        "skip_score": dataset_to_metrics[dataset][index, 3], 
                        "test_accuracy": api.query_meta_info_by_index(arch_index=index, 
                                                                     hp="200",
                                        ).get_metrics(dataset=dataset, setname="ori-test")["accuracy"]

                    }
        
    """Check if the `compressed_archive` directory exists, if not, create it"""
    MYDIR = str(get_project_root()) + "/compressed_archive"
    CHECK_FOLDER = os.path.isdir(MYDIR)
    # If folder doesn't exist, then create it (avoid failure, silently)
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)
        print("created folder : ", MYDIR)
    
    with open(f"{MYDIR}/nats_interface_dataset.json", "w") as data_file:
        json.dump(data, data_file, indent=2)

if __name__=="__main__": 
    main()