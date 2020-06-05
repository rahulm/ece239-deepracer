import json
from typing import Any, Dict, Text


class Config(object):
  """Wrapper of configuration, including:
- Model hyperparameters (for PPO, Actor-Critic, etc.)
- Training hyperparameters (planned)

These are loaded from a config_dict that is read from a JSON file, like the 
one found in "configs/config-2020_06_02.json". So, should use the helper
function get_config below to automatically read into this Config object.

An example of how to use this Config class:
  config = get_config("configs/config-2020_06_02.json")

  config.has_ppo    # True if the JSON has a "ppo_hyperparameters" key
  config.ppo        # If available, a dictionary of the "ppo_hyperparameters"

  config.has_actor_critic    # True if the ppo has a "actor_critic" key
  config.actor_critic        # If available, a dictionary of the "actor_critic"
  """
  KEY_PPO = "ppo_hyperparameters"
  KEY_ACTOR_CRITIC = "actor_critic"

  def __init__(self, config_dict: Dict) -> None:
    self.config_dict: Dict = config_dict
    
    if self.KEY_PPO in self.config_dict:
      self.__add_field("ppo", self.config_dict[self.KEY_PPO])

      if self.KEY_ACTOR_CRITIC in self.config_dict[self.KEY_PPO]:
        self.__add_field("actor_critic", self.config_dict[self.KEY_PPO][self.KEY_ACTOR_CRITIC])
      else:
        self.__mark_field_nonexistent("actor_critic")

    else:
      self.__mark_field_nonexistent("ppo")

  def __add_field(self, field_name: Text, field_value: Any) -> None:
    setattr(self, field_name, field_value)
    setattr(self, "has_{}".format(field_name), True)
  
  def __mark_field_nonexistent(self, field_name: Text) -> None:
    setattr(self, "has_{}".format(field_name), False)

  def __repr__(self) -> Text:
    return str(self.config_dict)
  
  # Start: for mypy
  has_ppo: bool
  ppo: Dict
  has_actor_critic: bool
  actor_critic: Dict
  has_training: bool
  training: Dict
  # End: for mypy


def get_config_dict(filename: Text) -> Dict:
  config_dict: Dict
  with open(filename, 'r') as config_file:
    config_dict = json.load(config_file)
  return config_dict


def get_config(filename: Text) -> Config:
  return Config(get_config_dict(filename))
