from icecream import ic
from custom_env.oscar import OscarEnv
from custom_env.utils import NASIndividual
from src.interfaces.nats_interface import NATS_Interface as NATS_Searchspace

searchspace = NATS_Searchspace(target_device="edgegpu")
env = OscarEnv(searchspace_api=searchspace)

individual = NASIndividual(architecture=None, index=None, architecture_string_to_idx=searchspace.architecture_to_index)
individual = env.mount_architecture(individual, searchspace.encode_architecture(env.networks_pool[0]))

env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

ic(obs)
ic(reward)
ic(terminated)
ic(truncated)
ic(info)

