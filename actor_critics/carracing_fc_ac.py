import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from actor_critics import ac_utils

class Actor(nn.Module):
  """
  A simple neural network whose input is an
  observation/state and output is the action
  probabilities.
  """

  def __init__(self, env):

    super(Actor, self).__init__()

    input_size = ac_utils.get_generic_space_size(env.observation_space)
    output_size = ac_utils.get_generic_space_size(env.action_space)

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

    input_size = ac_utils.get_generic_space_size(env.observation_space)

    self.fc1 = nn.Linear(input_size, 30)
    #self.fc2 = nn.Linear(10, 10)
    self.fc3 = nn.Linear(30, 1)

  def forward(self, x):

    x = F.relu(self.fc1(x))
    #x = F.relu(self.fc2(x))
    x = self.fc3(x)

    return x
