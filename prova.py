import json
from hwnats_api import HWNASBenchAPI

api = HWNASBenchAPI(file_path_or_dict="HW-NAS-Bench-v1_0.pickle", search_space="fbnet")

with open("fbnet_cached_metrics.json", "w") as f:
    json.dump(api.HW_metrics["fbnet"], f)
