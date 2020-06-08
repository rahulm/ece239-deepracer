import gym
import numpy as np
from torch import nn


def get_generic_space_size(env_space: gym.spaces.space) -> int:
  if isinstance(env_space, gym.spaces.Discrete):
    return env_space.n
  elif isinstance(env_space, gym.spaces.Box):
    return np.prod(env_space.shape)
  else:
    raise ValueError(
      "Given space is not supported yet: {}".format(type(env_space))
    )


def set_requires_grad(model: nn.Module, requires_grad: bool) -> None:
  for p in model.parameters():
    p.requires_grad = requires_grad


def get_env_space_info(env_space, metadata_custom_space):
  if metadata_custom_space:
      space_discrete = (metadata_custom_space["type"] == "discrete")
  else:
      space_discrete = isinstance(env_space, gym.spaces.Discrete)
  
  if metadata_custom_space and ("size" in metadata_custom_space):
      space_size = metadata_custom_space["size"]
  else:
      space_size = get_generic_space_size(env_space)
  
  return space_discrete, space_size
