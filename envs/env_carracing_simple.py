from typing import List, Tuple

from gym.envs.box2d import CarRacing


class CarRacingSimple(CarRacing):
  """This extends OpenAI gym's CarRacing environment.
Changes:
- Replaces the observation space with an arbitrarily chosen set of simpler
values, without an image. (TODO IMPLEMENTATION)
- Replaces the continuous action space with a discrete one.
  """

  def __init__(self, verbose=1):
    super().__init__(verbose=verbose)

    self.VALUES_STEERING: List[float] = [-1.0, 0.0, +1.0]
    self.VALUES_GAS: List[float] = [0.0, 0.5, +1.0]
    self.VALUES_BRAKE: List[float] = [0.0, 0.5, +1.0]

    self.ACTION_MAP: List[Tuple[float, float, float]] = [
      (s, g, b)
      for s in self.VALUES_STEERING
      for g in self.VALUES_GAS
      for b in self.VALUES_BRAKE
    ]

    self.NUM_ACTIONS: int = len(self.ACTION_MAP)

  def __is_valid_discrete_action(self, action: int) -> bool:
    return (
      isinstance(action, int)
      and (action >= 0)
      and (action < self.NUM_ACTIONS)
    )   

  def step(self, action):
    """Performs a custom step, converting a discrete action into a continuous
one that the CarRacing environment can accept.

Parameters:
  - action (int): An integer, in the range [0, 26], corresponding to the
                  discrete action to take.
    """

    if action is None:
      return super().step(action)

    if not self.__is_valid_discrete_action(action):
      raise ValueError("Not a valid discrete action: {}".format(action))
    
    continuous_action = self.ACTION_MAP[action]
    return super().step(continuous_action)
