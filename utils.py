import os
from typing import Any, Dict, Text

import torch

from ppo import PPO


def save_ppo_checkpoint(
  exp_dir: Text,
  epoch: int,
  loss: float,
  ppo: PPO
) -> None:
  os.makedirs(exp_dir, exist_ok=True)
  checkpoint_file = os.path.join(exp_dir, "checkpoint-{}.tar".format(epoch))

  checkpoint: Dict[Text, Any] = {
    "epoch": epoch,
    "env": str(ppo.env),
    "loss": loss
  }

  for model_name, model, model_opt in [
    ("actor", ppo.pi, ppo.optimizer_pi),
    ("critic", ppo.critic, ppo.optimizer_critic)
  ]:
    checkpoint["{}-model_state_dict".format(model_name)] = model.state_dict()
    checkpoint["{}-optimizer_state_dict".format(model_name)] = model_opt.state_dict()

  torch.save(checkpoint, checkpoint_file)
