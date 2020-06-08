import os
from typing import Any, Dict, List, Optional, Text, Tuple, Union

import matplotlib.pyplot as plt
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


def save_loss(
  loss_dir: Text,
  loss_vals: Union[
    List[float],
    List[Tuple[int, float]],
    Dict[int, float]
  ],
  save_default_plot: bool = True
) -> None:
  
  os.makedirs(loss_dir, exist_ok=True)

  # try to iterate over loss_vals
  inds: List[int]
  vals: List[float]
  if isinstance(loss_vals, list):
    if isinstance(loss_vals[0], tuple):
      inds = [lv[0] for lv in loss_vals]
      vals = [lv[1] for lv in loss_vals]
    else:
    # elif isinstance(loss_vals[0], float):
      inds = list(range(len(loss_vals)))
      vals = loss_vals
  elif isinstance(loss_vals, dict):
    inds = sorted(loss_vals.keys())
    vals = [loss_vals[k] for k in inds]
  else:
    # if we don't know what to do, just try something
    inds = list(range(len(loss_vals)))
    vals = loss_vals
  
  line_format = "{},{:.8f}\n"
  loss_vals_path = os.path.join(loss_dir, "loss_vals.csv")
  with open(loss_vals_path, "w") as loss_vals_file:
    loss_vals_file.write("epoch,loss\n")
    for ind, val in zip(inds, vals):
      loss_vals_file.write(line_format.format(ind, val))
    loss_vals_file.flush()
  
  if save_default_plot:
    plt_file_path = os.path.join(loss_dir, "loss_curve-default.png")
    plt.plot(inds, vals)
    plt.title("Loss curve")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.savefig(plt_file_path)

  

  
  
