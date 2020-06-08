import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

from actor_critics import ac_utils
sys.path.append('..')
from envs.env_carracing_simple import CarRacingSimple


class Actor(nn.Module):
  """
  An actor model based on the pre-trained resnet-18 model.

  NOTE: This model should only accept a CarRacingSimple env as input.
  """

  def __init__(self, env: CarRacingSimple, pretrained: bool = True):

    super(Actor, self).__init__()

    _, self.output_size = ac_utils.get_env_space_info(
      env.action_space,
      env.metadata.get("custom_action_space", None)
    )

    # create the pretrained resnet, freeze all layers
    self.resnet = models.resnet18(pretrained=pretrained, progress=True)
    self.resnet_output_size: int = self.resnet.fc.in_features
    ac_utils.set_requires_grad(model=self.resnet, requires_grad=False)

    # create a small output FC neural network
    self.fc_1 = nn.Linear(self.resnet_output_size, 60)
    self.fc_2 = nn.Linear(60, 40)
    self.fc_3 = nn.Linear(40, self.output_size)
    self.softmax = nn.Softmax()
    self.seq_out = nn.Sequential(
      self.fc_1,
      nn.ReLU(),
      self.fc_2,
      nn.ReLU(),
      self.fc_3,
      self.softmax
    )
    ac_utils.set_requires_grad(model=self.seq_out, requires_grad=True)

    # replace resnet fc layer with seq_out
    self.resnet.fc = self.seq_out

    # TODO: figure out if we actually need upsample, based on:
    # https://pytorch.org/docs/stable/torchvision/models.html
    # # make upsampling
    # self.upsample = nn.Upsample(size=(3, 224, 224))
    # ac_utils.set_requires_grad(model=self.upsample, requires_grad=False)
    
    # put all the pieces together
    self.full_model = nn.Sequential(
      # self.upsample,
      self.resnet
    )

    # make the normalize transform needed
    self.normalize = transforms.Normalize(
      mean=[0.485, 0.456, 0.406],
      std=[0.229, 0.224, 0.225]
    )

  def forward(self, x):
    # basic manual mapping from [0, 255] to [0, 1]
    x = torch.true_divide(x, 255.0)

    # switch from (b, h, w, c) to (b, c, h, w)
    x = x.permute(0, 3, 1, 2)

    # normalize, as required for pytorch pretrained models
    x_norm = []
    for b in x:
      x_norm.append(self.normalize(b))
    x = torch.stack(x_norm)

    # run model
    x = self.full_model(x)

    return x

class Critic(nn.Module):
  """
  A critic model based on the pre-trained resnet-18 model.

  NOTE: This model should only accept a CarRacingSimple env as input.
  """

  def __init__(self, env: CarRacingSimple, pretrained: bool = True):

    super(Critic, self).__init__()

    # create the pretrained resnet, freeze all layers
    self.resnet = models.resnet18(pretrained=pretrained, progress=True)
    self.resnet_output_size: int = self.resnet.fc.in_features
    ac_utils.set_requires_grad(model=self.resnet, requires_grad=False)

    # create a small output FC neural network
    self.fc_1 = nn.Linear(self.resnet_output_size, 50)
    self.fc_2 = nn.Linear(50, 20)
    self.fc_3 = nn.Linear(20, 1)
    self.seq_out = nn.Sequential(
      self.fc_1,
      nn.ReLU(),
      self.fc_2,
      nn.ReLU(),
      self.fc_3
    )
    ac_utils.set_requires_grad(model=self.seq_out, requires_grad=True)

    # replace resnet fc layer with seq_out
    self.resnet.fc = self.seq_out    

    # TODO: figure out if we actually need upsample, based on:
    # https://pytorch.org/docs/stable/torchvision/models.html
    # # make upsampling
    # self.upsample = nn.Upsample(size=(3, 224, 224))
    # ac_utils.set_requires_grad(model=self.upsample, requires_grad=False)
    
    # put all the pieces together
    self.full_model = nn.Sequential(
      # self.upsample,
      self.resnet
    )

    # make the normalize transform needed
    self.normalize = transforms.Normalize(
      mean=[0.485, 0.456, 0.406],
      std=[0.229, 0.224, 0.225]
    )

  def forward(self, x):
    # basic manual mapping from [0, 255] to [0, 1]
    x = torch.true_divide(x, 255.0)

    # switch from (b, h, w, c) to (b, c, h, w)
    x = x.permute(0, 3, 1, 2)

    # normalize, as required for pytorch pretrained models
    x_norm = []
    for b in x:
      x_norm.append(self.normalize(b))
    x = torch.stack(x_norm)

    # run model
    x = self.full_model(x)

    return x
