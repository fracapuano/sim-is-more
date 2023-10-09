"""
This script builds the NATS Interface dataset, combining data from Nats Bench (https://arxiv.org/abs/2009.00437)
and HW-NAS-Bench: Hardware-Aware Neural Architecture Search Benchmark (https://openreview.net/pdf?id=_0kaDkv3dVf). 

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
    Moreover, we will make the assumption that the compressed_archive/hw_nats_interface_dataset.json file
    exists. From this file, we will be pulling out the hardware-aware metrics that we will "join" with
    training-free and performance metrics.
    """
    datasets = ["cifar10", "cifar100", "imagenet"]
    # this lambda function is only used to account for the difference in naming of "ImageNet16-120"
    convert_dataset = lambda d: d if "imagenet" not in d else "ImageNet16-120"

    path_to_archive = str(get_project_root()) + "/archive/"
    path_to_nats = path_to_archive + "NATS-tss-v1_0-3ffb9-simple"
    path_to_metrics = path_to_archive + "cachedmetrics/"
    path_to_hw = path_to_archive + "hw_nats_data.json"

    with open(path_to_hw, "r") as hardware_file: 
        hardware_dict = json.load(hardware_file)

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
                    test_accuracy: float, 
                    edgegpu_latency: float,
                    edgegpu_energy: float,
                    raspi4_latency: float,
                    edgetpu_latency: float,
                    pixel3_latency: float,
                    eyeriss_latency: float,
                    eyeriss_energy: float,
                    eyeriss_arithmetic_intensity: float,
                    fpga_latency: float,
                    fpga_energy: float,
                    average_hw_metric: float             
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
            # add hardware-aware metrics from hardware_dict via update
            data[index][dataset].update(hardware_dict[str(index)][dataset])
            
    """Check if the `compressed_archive` directory exists, if not, create it"""
    MYDIR = str(get_project_root()) + "/compressed_archive"
    CHECK_FOLDER = os.path.isdir(MYDIR)
    # If folder doesn't exist, then create it (avoid failure, silently)
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)
        print("created folder : ", MYDIR)
    
    with open(f"{MYDIR}/hw_nats_interface_dataset.json", "w") as data_file:
        json.dump(data, data_file, indent=2)

if __name__=="__main__":
    main()