# %%
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

paretodf = pd.read_csv("paretoindices.csv", index_col=0)
# %%
paretodf
# %%
devices = [
    "edgegpu",
    "raspi4",
    "edgetpu",
    "pixel3",
    "eyeriss",
    "fpga",
    "titanx",
    "samsung_a50",
    "titan_rtx",
    "titanxp",
    "pixel2",
    "silver_4210r",
    "essential_ph",
    "silver_4114",
    "samsung_s7",
    "2080ti",
    "1080ti",
    "gold_6226",
    "gold_6240",
]
# %%
paretodf.columns = devices
# %%

paretodf.head()
from src.interfaces import NATS_Interface

searchspace = NATS_Interface()
# %%
"""
what is the threshold for the pairs 45 archis x 19 devices -> latency, val acc
rank 
"""
# benchmark
device = "raspi4"
searchspace.architecture_to_score(searchspace[1], "naswot_score")
searchspace.architecture_to_score(searchspace[1], f"{device}_latency")


# %%
from collections import defaultdict
from tqdm import tqdm
import numpy as np

performance_dict = defaultdict(list)
for device in tqdm(paretodf.columns):
    for archi_idx in paretodf[device].values:
        if np.isnan(archi_idx):
            continue
        archi_idx = int(archi_idx)
        archi_obj = searchspace[int(archi_idx)]
        naswot_score = searchspace.architecture_to_score(archi_obj, "naswot_score")
        logsynflow_score = searchspace.architecture_to_score(
            archi_obj, "logsynflow_score"
        )
        skip_score = searchspace.architecture_to_score(archi_obj, "skip_score")

        # what latency threshold above which all of the archis are the same pareto score
        # how to compute the pareto score
        latency = searchspace.architecture_to_score(archi_obj, f"{device}_latency")
        accuracy = searchspace.architecture_to_accuracy(archi_obj)
        performance_dict["device"].append(device)
        performance_dict["archi_idx"].append(archi_idx)
        performance_dict["architecture"].append(archi_obj)
        performance_dict["accuracy"].append(accuracy)
        performance_dict["latency"].append(latency)


# %%
performance_df = pd.DataFrame(performance_dict)
performance_df.shape
performance_df["latency_pct"] = performance_df.groupby("device").latency.rank(pct=True)
performance_df["latency_pct"] = (performance_df["latency_pct"] * 100).round(0)

# %%
# overlaps
from itertools import product, islice

for device0, device1 in islice(product(devices, devices), 6, None, 5):
    if device0 == device1:
        continue
    # %%
    perfpair = performance_df.loc[performance_df.device.isin([device0, device1])]
    # fig = px.scatter(
    #     performance_df.loc[performance_df.device.isin([device0, device1])],
    #     x="latency_pct",
    #     y="accuracy",
    #     color="device",
    #      hover_data=["archi_idx"]
    # )
    # fig.update_layout({"title": f"{device0}-{device1}"})
    # fig.show()
    archigroups = iter(perfpair.groupby("archi_idx"))
    # %%
    archigroups =perfpair.groupby("archi_idx").size().value_counts()
    archicounts = perfpair.groupby("archi_idx").size().reset_index()
    archicounts
    # %% 
    perfpair = perfpair.merge(archicounts.reset_index().rename(
        columns= {0:"archi_counts"}
    ), on = ["archi_idx"])
    perfpair.loc[perfpair.archi_counts>1]

    # %% 
    break 



# %%
searchspace[0]

# %%
