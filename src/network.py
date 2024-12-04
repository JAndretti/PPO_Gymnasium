"""
    This file contains a neural network module for us to
    define our actor and critic networks in PPO.
"""

import torch
from torch import nn
import numpy as np
from torch.distributions.categorical import Categorical
import os
import torch.optim as optim


class FeedForwardNN(nn.Module):
    """
    A standard in_dim-64-64-out_dim Feed Forward Neural Network.
    """

    def __init__(self, in_dim, out_dim, alpha, training=True, actor=True, HP=None):
        """
        Initialize the network and set up the layers.

        Parameters:
                in_dim - input dimensions as an int
                out_dim - output dimensions as an int

        Return:
                None
        """
        super(FeedForwardNN, self).__init__()
        torch.manual_seed(1)
        self.HP = HP
        self.actor = actor
        self.training = training
        if self.HP["DEVICE"] == "cuda" and torch.cuda.is_available():
            # Warm up CUDA or MPS
            for _ in range(10):
                a = torch.randn(5000, 5000, device=self.device)
                b = torch.randn(5000, 5000, device=self.device)
                torch.matmul(a, b)
            self.device = torch.device("cuda")
        elif self.HP["DEVICE"] == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            # Warm up CUDA or MPS
            for _ in range(10):
                a = torch.randn(5000, 5000, device=self.device)
                b = torch.randn(5000, 5000, device=self.device)
                torch.matmul(a, b)
        else:
            self.device = torch.device("cpu")
        str = "actor" if actor else "critic"
        print(f"Using device for {str} : {self.device}")
        self.device = self.HP["DEVICE"]
        self.neural = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, out_dim),
        )
        if self.actor:
            self.neural.append(nn.Softmax(dim=-1))

        self.optimizer = optim.Adam(
            self.parameters(), lr=alpha, weight_decay=1e-2, betas=(0.9, 0.999)
        )
        self.to(self.device)

    def forward(self, obs):
        """
        Runs a forward pass on the neural network.

        Parameters:
                obs - observation to pass as input

        Return:
                output - the output of our forward pass
        """
        # Convert observation to tensor if it's a numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        dist = self.neural(obs)
        if self.training and self.actor:
            dist = Categorical(dist)

        return dist

    def save_network(self, file_path: str) -> None:
        """
        Saves the model to a file, and the HP parameters concerning
        the model architecture
        """
        if self.actor:
            file_path = os.path.join(
                os.path.dirname(file_path), "actor_" + os.path.basename(file_path)
            )
        else:
            file_path = os.path.join(
                os.path.dirname(file_path), "critic_" + os.path.basename(file_path)
            )
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "HP": {k: v for k, v in self.HP.get_config().items()},
            },
            file_path,
        )

    def load_network(self, file_path: str) -> None:
        """
        Loads the model from a file
        """
        checkpoint = torch.load(file_path, weights_only=True)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.HP = checkpoint["HP"]
        self.to(self.device)
        self.eval()
        print(f"Loaded model from {file_path}")
