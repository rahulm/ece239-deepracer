import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from actor_critics import ac_utils
sys.path.append('..')
from envs.env_carracing_simple import CarRacingSimple


class Actor(nn.Module):
  """
  A simple neural network whose input is an
  observation/state and output is the action
  probabilities.

  NOTE: This model should only accept a CarRacingSimple env as input.
  """

  def __init__(self, env: CarRacingSimple):

    super(Actor, self).__init__()

    input_size = ac_utils.get_generic_space_size(env.observation_space)
    output_size = ac_utils.get_generic_space_size(env.action_space)


    self.conv1 = nn.Conv2d(3, 16, 8, stride=2)
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.conv2 = nn.Conv2d(16, 8, 8, stride=2)
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.fl = nn.Flatten()
    self.fc1 = nn.Linear(128, 30)
    #self.fc2 = nn.Linear(10, 10)
    #self.fc3 = nn.Linear(30, 30)

    self.fc4_1 = nn.Linear(30, 3)
    self.fc4_2 = nn.Linear(30, 3)
    self.fc4_3 = nn.Linear(30, 3)

    self.softmax1 = nn.Softmax(dim=-1)
    self.softmax2 = nn.Softmax(dim=-1)
    self.softmax3 = nn.Softmax(dim=-1)



  def forward(self, x):
    # basic manual mapping from [0, 255] to [0, 1]
    x = torch.true_divide(x, 255.0)
    x = torch.transpose(x, 1, 3)

    x = F.relu(self.conv1(x))
    x = self.pool1(x)

    x = F.relu(self.conv2(x))
    x = self.pool2(x)

    x = self.fl(x)
    x = F.relu(self.fc1(x))
    #x = F.relu(self.fc2(x))
    #x = F.relu(self.fc3(x))

    x1 = self.fc4_1(x)
    x2 = self.fc4_2(x)
    x3 = self.fc4_3(x)

    x1 = self.softmax2(x1)
    x2 = self.softmax2(x2)
    x3 = self.softmax3(x3)
    
    # NOTE: assumes that x is batched
    num_batches = len(x)
    xprods = []
    for b in range(num_batches):
      xprods.append(torch.cartesian_prod(x1[b], x2[b], x3[b]))
    xprods = torch.stack(xprods)

    x = xprods[:, :, 0] * xprods[:, :, 1] * xprods[:, :, 2]

    return x

class Critic(nn.Module):
  """
  A simple neural network whose input is an
  observation/state and output is the value.

  NOTE: This model should only accept a CarRacingSimple env as input.
  """

  def __init__(self, env: CarRacingSimple):

    super(Critic, self).__init__()

    input_size = ac_utils.get_generic_space_size(env.observation_space)

    self.conv1 = nn.Conv2d(3, 16, 8, stride=2)
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.conv2 = nn.Conv2d(16, 8, 8, stride=2)
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.fl = nn.Flatten()
    self.fc1 = nn.Linear(128, 30)
    #self.fc2 = nn.Linear(10, 10)
    self.fc3 = nn.Linear(30, 1)

  def forward(self, x):

    # basic manual mapping from [0, 255] to [0, 1]
    x = torch.true_divide(x, 255.0)
    x = torch.transpose(x, 1, 3)


    x = F.relu(self.conv1(x))
    x = self.pool1(x)

    x = F.relu(self.conv2(x))
    x = self.pool2(x)

    x = self.fl(x)
    x = F.relu(self.fc1(x))
    #x = F.relu(self.fc2(x))
    x = self.fc3(x)

    return x
