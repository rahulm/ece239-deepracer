import gym
import numpy as np


def get_generic_space_size(env_space: gym.spaces.space) -> int:
  if isinstance(env_space, gym.spaces.Discrete):
    return env_space.n
  elif isinstance(env_space, gym.spaces.Box):
    return np.prod(env_space.shape)
  else:
    raise ValueError(
      "Given space is not supported yet: {}".format(type(env_space))
    )
