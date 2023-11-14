from .base_env import *
from .nas_env import *
from .oscar import *
from .marcella import *
from .utils import *

envs_dict = {
    "nasenv": NASEnv,
    "oscar": OscarEnv,
    "marcella": MarcellaEnv
}

