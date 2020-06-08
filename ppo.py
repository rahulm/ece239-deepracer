from typing import Dict, List, Text

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Categorical

from config_utils import Config
from actor_critics import ac_utils

class PPO():

    """
    A class that runs Proximal Policy Optimization on an Openai gym
    envoriment. 
    """

    def __init__(self,
        env,
        config: Config,
        actor: nn.Module,
        critic: nn.Module,
        no_cuda: bool = False
    ):
        
        # Setting up cuda, if needed
        self.device_name: Text
        if no_cuda or (not torch.cuda.is_available()):
            self.device_name = "cpu"
        else:
            self.device_name = "cuda:0"
        self.torch_device = torch.device(self.device_name)
        
        self.env = env

        self.act_discrete, self.act_size = ac_utils.get_env_space_info(
            self.env.action_space,
            self.env.metadata.get("custom_action_space", None)
        )

        self.obs_discrete, self.obs_size = ac_utils.get_env_space_info(
            self.env.observation_space,
            self.env.metadata.get("custom_observation_space", None)
        )

        self.values = torch.zeros(self.obs_size)

        self.config: Config = config

        self.horizon: int = self.config.ppo.get("horizon", 100)
        self.gamma: float = self.config.ppo["discount_factor"]
        self.eps: float = self.config.ppo["epsilon"]

        self.entropy_coef: float = self.config.ppo.get("entropy_coef", 1)
        self.vf_coef: float = self.config.ppo.get("vf_coef", 1)

        self.delta: float = self.config.ppo.get(
            "gae_parameter",
            self.config.ppo.get("delta", 1)
        )
        
        self.alpha: float = self.config.actor_critic["critic"]["alpha"]
        self.pi = actor.to(self.torch_device)
        self.critic = critic.to(self.torch_device)
        self.optimizer_pi = torch.optim.Adam(
            self.pi.custom_parameters
            if hasattr(self.pi, "custom_parameters")
            else self.pi.parameters(),
            lr=self.config.actor_critic["actor"]["learning_rate"]
        )
        self.optimizer_critic = torch.optim.Adam(
            self.critic.custom_parameters
            if hasattr(self.critic, "custom_parameters")
            else self.critic.parameters(),
            lr=self.config.actor_critic["critic"]["learning_rate"]
        )

    def sample(self, n):
        """
        Make an state, action, reward, state observation.
        """
        obs = []
        rewards = []
        actions = []
        observation = self.env.reset()
        obs.append(observation)
        observation = observation.copy()

        for i in range(n):

            if self.act_discrete:
                action = torch.argmax(
                    self.pi(torch.unsqueeze(
                            torch.from_numpy(observation.copy()),
                            0
                        ).to(self.torch_device).float()
                    )
                )
            else:
                action = self.pi(
                    torch.unsqueeze(
                        torch.from_numpy(observation.copy()),
                        0
                    ).to(self.torch_device).float()
                )
            
            action = torch.squeeze(action, 0)

            actions.append(action)
            observation, reward, done, info = self.env.step(action.cpu().detach().numpy())
            obs.append(observation)
            rewards.append(reward)

            if done:
                break

        rewards_var = torch.tensor(rewards, requires_grad=False).to(self.torch_device)
        obs_var = torch.tensor(obs, requires_grad=False).float().to(self.torch_device)
        actions_var = torch.stack(actions).to(self.torch_device)

        return obs_var, rewards_var, actions_var


    def get_advantages(self, values, rewards):

        advantages = []
        returns = [1]
        ret = 1
        gae = 0
        for i in reversed(range(rewards.shape[0])):
            delta = rewards[i] + self.gamma * values[i+1] - values[i]
            gae = delta + self.gamma * self.delta * gae
            ret = rewards[i] + self.gamma * ret
            returns.insert(0, ret)
            advantages.insert(0, gae)
       
        returns = torch.tensor(returns).to(self.torch_device)
        advantages = torch.transpose(torch.stack(advantages, dim=0), 0, 1).to(self.torch_device)

        return advantages, returns

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

        return (clip_loss + self.entropy_coef * entropy).mean()


    def step(self):
        """
        Take a single step of the PPO update
        """
        obs, rewards, actions = self.sample(self.horizon)

        rewards = rewards.detach()
        probs = Categorical(self.pi(obs[:-1]))
        log_probs = probs.log_prob(actions)
        values = self.critic(obs)
        advantages, returns = self.get_advantages(values.detach(), rewards)

        loss_vals: List[float] = []

        for _ in range(10):
            values = self.critic(obs)
            __, returns = self.get_advantages(values.detach(), rewards)
            values = torch.reshape(values, (-1,))

            critic_loss = self.alpha * torch.square(values-returns).mean()
            actor_loss = -self.calculate_loss(advantages.detach(), obs, actions, log_probs.detach())
            loss_vals.append(-(actor_loss.item()))

            self.optimizer_pi.zero_grad()
            actor_loss.backward()
            self.optimizer_pi.step()

            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            self.optimizer_critic.step()
        
        return loss_vals
