import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_action_space_size(env):
  act_size = env.action_space.n  
  return act_size


def get_observation_space_size(env):
  obs_size = np.prod(env.observation_space.shape)
  return obs_size


class Actor(nn.Module):
  """
  A simple neural network whose input is an
  observation/state and output is the action
  probabilities.
  """

  def __init__(self, env):

    super(Actor, self).__init__()

    input_size = get_observation_space_size(env)
    output_size = get_action_space_size(env)

    self.fc1 = nn.Linear(input_size, 30)
    #self.fc2 = nn.Linear(10, 10)
    self.fc3 = nn.Linear(30, output_size)
    self.softmax = nn.Softmax(dim=-1)

  def forward(self, x):

    x = F.relu(self.fc1(x))
    #x = F.relu(self.fc2(x))
    x = self.fc3(x)
    x = self.softmax(x)

    return x

class Critic(nn.Module):
  """
  A simple neural network whose input is an
  observation/state and output is the value.
  """

  def __init__(self, env):

    super(Critic, self).__init__()

    input_size = get_observation_space_size(env)

    self.fc1 = nn.Linear(input_size, 30)
    #self.fc2 = nn.Linear(10, 10)
    self.fc3 = nn.Linear(30, 1)

  def forward(self, x):

    x = F.relu(self.fc1(x))
    #x = F.relu(self.fc2(x))
    x = self.fc3(x)

    return x
