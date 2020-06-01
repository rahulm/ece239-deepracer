import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import gym

class Policy(nn.Module):
    """
    A simple neural network whose input is an
    observation/state and output is the action
    probabilities.
    """

    def __init__(self, input_size, output_size):

        super(Policy, self).__init__()

        self.fc1 = nn.Linear(input_size, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)

        return x

class Critic(nn.Module):
    """
    A simple neural network whose input is an
    observation/state and output is the value.
    """

    def __init__(self, input_size):

        super(Critic, self).__init__()

        self.fc = nn.Linear(input_size, 1)

        self.fc1 = nn.Linear(input_size, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x



class PPO():

    """
    A class that runs Proximal Policy Optimization on an Openai gym
    envoriment. 
    """

    def __init__(self, env, obs_size, act_size, eps, gamma, alpha):

        self.env = env

        self.act_size = act_size
        self.obs_size = obs_size
        self.values = torch.zeros(obs_size)

        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps


        self.net = FCNet(obs_size)
        self.pi = Policy(obs_size, act_size, self.net)
        self.critic = Critic(obs_size, self.net)

        self.optimizer_pi = torch.optim.Adam(self.pi.parameters(), lr=0.005)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=0.005)

    def sample(self, pi):
        """
        Make an state, action, reward, state observation.
        """
        observation1 = self.env.reset()
        observation1 = Variable(torch.from_numpy(observation1), requires_grad=False).detach()
        action = torch.argmax(self.pi(observation1.float())) 
        observation2, reward, done, info = self.env.step(action.detach().numpy())

        return observation1, action, reward, observation2

    def prob(self, policy, observation, action):

        """
        Calculate the probability of taking a given action
        from a given observation using the given policy
        """

        return policy(observation.float())[action.detach()]

        ratio1 = prob(pi_new, obs, act) 
        ratio2 = prob(pi_old, obs, act)

        return ratio1 / ratio2


    def L_CLIP_loss(self, ratio, advantage):

        """
        Calculate CLIP loss from paper
        """

        loss1 = ratio * advantage
        loss2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * advantage

        return min(loss1, loss2)

    def calculate_loss(self, policy, advantage, obs, act, prob_old):

        """
        Calcuate the total actor loss
        """

        r = self.prob(policy, obs, act) / prob_old
        probs = self.pi(obs.float())
        return self.L_CLIP_loss(r, advantage) - 1 * torch.mean(probs * torch.log(probs + 1e-10))


    def step(self):
        """
        Take a single step of the PPO update
        """
        #sample stata, action, reward, state
        observation1, action, reward, observation2 = self.sample(self.pi)
        observation2 = Variable(torch.from_numpy(observation2)).detach()
        #calculate advantage and TD estimate
        val_1 = self.critic(observation1.float())
        val_2 = self.critic(observation2.float())
        advantage_est = reward + self.gamma * val_1 - val_2

        #critic loss
        critic_loss = (self.alpha * advantage_est)**2
        
        prob_old = Variable(self.prob(self.pi, observation1, action)).detach()
        actor_loss = self.calculate_loss(self.pi, advantage_est.detach(), observation1, action, prob_old)
        
        self.optimizer_pi.zero_grad()
        actor_loss.backward()

        self.optimizer_critic.zero_grad()
        critic_loss.backward()

