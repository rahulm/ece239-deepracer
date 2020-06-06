from gym.envs.box2d import CarRacing

class CarRacingSimple(CarRacing):
  """This extends OpenAI gym's CarRacing environment.
Only replaces the observation space with an arbitrarily chosen set of simpler
values, without an image.
  """
  pass