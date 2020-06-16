# ece239-deepracer
An implementation of the AWS DeepRacer PPO algo, for Reinforcement Learning (ECE 239AS) under Prof. Lin Yang at UCLA.

By Rahul Malavalli, Glen Meyerowitz, and Joshua Vendrow.


### Required Functions
1. Generate environment (race track of the form y = f(x) over some domain x = [x_min, x_max]
2. Generate agent (vehicle that moves in the environment based on the state space)
3. Create state space (table of angles and speeds)
4. Reward function (determine reward given to the agent at each time step t)
5. PPO implementation (train the agent over many iterations)
6. Train agent (simulate many episodes of the agent in the environment)
7. Collect quantitative performance data of the agent during training
