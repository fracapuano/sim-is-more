from .base_env import * 
from .renas_env import *
from .freerenas_env import *
from .recombination_env import *
from .HW_freerenas_env import *
from .HW_recombination_env import *
from .nas_env import *
from .HW_nas_env import *
from .oscar import *
from .marcella import *
from .utils import *

envs_dict = {
    "renas": RENASEnv, 
    "freerenas": FreeRENASEnv, 
    "freerenas-recombination": FreeRENAS_RecombinationEnv,
    "hwfreerenas": HW_FreeRENASEnv,
    "hwfreerenas-recombination": HW_FreeRENAS_RecombinationEnv,
    "nasenv": NASEnv,
    "hw-nasenv": HW_NASEnv,
    "oscar": OscarEnv,
    "marcella": MarcellaEnv
}

hardware_agnostic_search = [
    "renas", 
    "freerenas", 
    "freerenas-recombination", 
    "nasenv"]

hardware_aware_search = [
    "hwfreerenas", 
    "hwfreerenas-recombination",
    "hw-nasenv",
    "oscar",
    "marcella"
]

pure_search = [
    "nasenv", 
    "hw-nasenv", 
    "oscar", 
    "marcella"
]
