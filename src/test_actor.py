import gymnasium as gym
from network import FeedForwardNN
from HP import _HP, get_script_arguments
import torch
import numpy as np
import os
import re
import glob


def find_actor(path):
    actor_files = glob.glob(os.path.join(path, "actor_epoch_*.pt"))
    if not actor_files:
        raise FileNotFoundError("No actor files found in the specified dlirectory.")

    actor_files.sort(
        key=lambda x: int(re.search(r"actor_epoch_(\d+)", x).group(1)),
        reverse=True,
    )
    return actor_files[0]


HP = _HP("src/HP.yaml")
HP.update(get_script_arguments(HP.keys()))

if HP["RANDOM"]:
    threshold = HP["THRESHOLD"]
else:
    threshold = 1


if HP["FILE"] == "" or HP["FILE"] == "":
    path = find_actor(HP["FOLDER"])
else:
    path = os.path.join(HP["FOLDER"], HP["FILE"])

env = gym.make(HP["ENV_NAME"])
observation, info = env.reset()
observation = torch.tensor(np.array([observation]), dtype=torch.float32)
observation = observation.to(HP["DEVICE"])
actor = FeedForwardNN(
    in_dim=env.observation_space.shape[0],
    out_dim=env.action_space.n,
    alpha=0.0003,
    training=False,
    actor=True,
    HP=HP,
)
actor.load_network(path)

episode_over = False
hist_score = []
for i in range(1, HP["NB_EPISODES"] + 1):
    score = 0
    episode_over = False
    env = gym.make(HP["ENV_NAME"], render_mode="human")
    observation, info = env.reset(seed=i)
    observation = torch.tensor(np.array([observation]), dtype=torch.float32)
    observation = observation.to(HP["DEVICE"])
    while not episode_over:
        if np.random.rand() <= threshold:
            action = torch.argmax(actor(observation.clone().detach())).item()
        else:
            action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        observation = torch.tensor(np.array([observation]), dtype=torch.float32)
        observation = observation.to(HP["DEVICE"])
        score += reward

        episode_over = terminated or truncated
    print(f"Episode {i} - Score: {score}")
    hist_score.append(score)

    env.close()
print(f"Average score: {np.mean(hist_score)}")
env.close()
