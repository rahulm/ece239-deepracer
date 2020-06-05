from typing import Dict

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

from config_utils import Config


class Policy(nn.Module):
    """
    A simple neural network whose input is an
    observation/state and output is the action
    probabilities.
    """

    def __init__(self, input_size, output_size):

        super(Policy, self).__init__()

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

    def __init__(self, input_size):

        super(Critic, self).__init__()

        self.fc1 = nn.Linear(input_size, 30)
        #self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(30, 1)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x



class PPO():

    """
    A class that runs Proximal Policy Optimization on an Openai gym
    envoriment. 
    """

    def __init__(self, env, config: Config):
        
        self.env = env

        self.act_size = self.env.action_space.n
        self.obs_size = np.prod(self.env.observation_space.shape)
        self.values = torch.zeros(self.obs_size)

        self.config: Config = config

        self.horizon: int = self.config.ppo.get("horizon", 100)
        self.gamma: float = self.config.ppo["discount_factor"]
        self.eps: float = self.config.ppo["epsilon"]

        self.entropy_coef: float = self.config.ppo.get("entropy_coef", 1)
        self.vf_coef: float = self.config.ppo.get("vf_coef", 1)

        self.delta: float = self.config.ppo.get("delta", 1)
        
        self.alpha: float = self.config.actor_critic["critic"]["alpha"]
        self.pi = Policy(self.obs_size, self.act_size)
        self.critic = Critic(self.obs_size)
        self.optimizer_pi = torch.optim.Adam(
            self.pi.parameters(),
            lr=self.config.actor_critic["actor"]["learning_rate"]
        )
        self.optimizer_critic = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.config.actor_critic["critic"]["learning_rate"]
        )

    def sample(self, pi, n):
        """
        Make an state, action, reward, state observation.
        """
        obs = []
        rewards = []
        actions = []
        observation = self.env.reset()
        #observation = Variable(torch.from_numpy(observation), requires_grad=False).detach()
        obs.append(observation)

        for i in range(n):

            #action = torch.argmax(self.pi(observation.float())) 
            action = torch.argmax(self.pi(Variable(torch.from_numpy(observation), requires_grad=False).float())) 

            actions.append(action)
            observation, reward, done, info = self.env.step(action.detach().numpy())
            #observation = Variable(torch.from_numpy(observation), requires_grad=False).detach()
            obs.append(observation)
            rewards.append(reward)

            if done:
                break

        rewards = Variable(torch.tensor(rewards).float(), requires_grad=False)

        return Variable(torch.tensor(obs).float(), requires_grad=False), rewards, Variable(torch.tensor(actions).float(), requires_grad=False)


    def get_advantages(self, values, rewards):

        #est_value = []
        #advantages = torch.tensor([])
        advantages = []
        returns = [1]
        ret = 1
        gae = 0
        for i in reversed(range(rewards.shape[0])):
            delta = rewards[i] + self.gamma * values[i+1] - values[i]
            gae = delta + self.gamma * self.delta * gae
            ret = rewards[i] + self.gamma * ret
            returns.insert(0, ret)

            #if i == 0:
            #    advantages = gae
            #else:
            #    advantages = torch.cat([advantages, gae], dim=1)
            #advantages.append(gae)
            advantages.insert(0, gae)
            #returns.append(gae + values[i])
            #advantages = torch.cat((advantages, torch.tensor([gae])))
            #est_value.insert(0, gae + values[i])
       
        returns = torch.tensor(returns)
        advantages = torch.transpose(torch.stack(advantages, dim=0), 0, 1)

        return advantages, returns #returns#torch.tensor(advantages)#, torch.FloatTensor(est_value)

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

        return torch.min(loss1, loss2)

    def calculate_loss(self, advantage, obs, actions, log_probs_old):

        """
        Calcuate the total actor loss
        """

        new_probs = Categorical(self.pi(obs[:-1]))
        log_probs = new_probs.log_prob(actions)

        ratio = torch.exp(log_probs - log_probs_old)
        
        clip_loss = self.L_CLIP_loss(ratio, advantage)
        entropy = new_probs.entropy()
        #print(log_probs)
        #print(entropy)
        #print(clip_loss)
        return (clip_loss + self.entropy_coef * entropy).mean()#torch.mean(probs * torch.log(probs + 1e-10))


    def step(self):
        """
        Take a single step of the PPO update
        """
        #sample stata, action, reward, state
        #observation1, action, reward, observation2 = self.sample(self.pi)
        obs, rewards, actions = self.sample(self.pi, self.horizon)
        #print(actions)
        #print(obs)

        #print(self.pi(obs[:-1]))
        rewards = rewards.detach()
        probs = Categorical(self.pi(obs[:-1]))
        log_probs = probs.log_prob(actions)
        values = self.critic(obs)
        advantages, returns = self.get_advantages(values.detach(), rewards)


        for _ in range(10):

            #observation2 = Variable(torch.from_numpy(observation2)).detach()
            #calculate advantage and TD estimate
            #val_1 = self.critic(observation1.float())
            #val_2 = self.critic(observation2.float())
            #advantage_est = reward + self.gamma * val_1 - val_2

            #probs = Categorical(self.pi(obs[:-1]))
            #log_probs = probs.log_prob(actions)
            values = self.critic(obs)
            __, returns = self.get_advantages(values.detach(), rewards)
            values = torch.reshape(values, (-1,))


            #if _ == 0:
            #    print("values: ", values)
            #    print("advantages: ", advantages)
            #    print("returns: ", returns)
            #advantages = Variable(torch.Tensor(advantages))
            #values = Variable(values, requires_grad=False)

            #critic loss
            #critic_loss = self.alpha * torch.norm(advantages)**2
            critic_loss = self.alpha *  torch.square((values-returns.detach())).mean()

            #if _ == 0:
            #    print(critic_loss)

            
            #prob_old = Variable(self.prob(self.pi, obs, action)).detach()

            #advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            #print(advantages)
            

            actor_loss = -self.calculate_loss(advantages.detach(), obs, actions, log_probs.detach())
            #if _ == 0:
            #print(actor_loss)
            #print(actor_loss)
            #print(actor_loss)
            

            self.optimizer_pi.zero_grad()
            actor_loss.backward()
            #print(self.pi.fc1.weight.grad)
            self.optimizer_pi.step()

            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            self.optimizer_critic.step()

