from custom_env.reward import Rewardv0
from custom_env.utils import NASIndividual
from src.interfaces.nats_interface import NATS_Interface as NATS_Searchspace

searchspace = NATS_Searchspace(target_device="edgegpu")

architecture, index = "|avg_pool_3x3~0|+|nor_conv_1x1~0|skip_connect~1|+|nor_conv_1x1~0|skip_connect~1|skip_connect~2|", 0
individual = NASIndividual(architecture=architecture, index=index, architecture_string_to_idx=searchspace.architecture_to_index)

reward_handler = Rewardv0(searchspace=searchspace)

reward = reward_handler.get_reward(individual)
print(reward)
