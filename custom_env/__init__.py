from .nas_env import *
from .oscar import *
from .marcella import *
from .marcella_plus import *
from .utils import *

envs_dict = {
    "nasenv": NASEnv,
    "oscar": OscarEnv,
    "marcella": MarcellaEnv,
    "marcella-plus": MarcellaPlusEnv
}

