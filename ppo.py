import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym


class Policy():

    def __init__(self, input_size, output_size):

        self.fc1 = nn.Linear(input_size, 40)
        self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(40, output_soze)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class PPO():


    def __init__(self, env, obs_size, act_size)

        self.env = env

        self.act_size = act_size
        self.obs_size = obs_size

        self.pi = Policy(obs_size, act_size)

    def sample(self, pi):
        """
        Evalulate the given policy using sampling
        """
        observation = self.env.reset()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        return observation, action, reward

    def prob(self, policy, observation, action):

        """
        Calculate the probability of taking a given action
        from a given observation using the given policy
        """

        return policy(observation)[action]

    def ratio(self, pi_new, pi_old):
        """
        Calculates the probability ration between the
        new and old policies

        Parameters
        ----------
        pi_new: Torch Tensor
            The new policy

        pi_old: Torch Tensor
            The old policy

        Returns
        -------

        """

    def loss(self, policy_new, reward):

        r = ratio(self, policy_new, self.pi)



    def step(self):







