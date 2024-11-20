import torch
import numpy as np
from network import FeedForwardNN
from memory import PPOMemory


class Agent:

    def __init__(
        self,
        HP,
        input_dims,
        n_actions,
        batch_size,
        n_epochs,
        alpha,
        gamma,
        gae_lambda,
        policy_clip,
    ):
        self.HP = HP
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.gae_lambda = gae_lambda
        self.alpha = alpha
        self.actor = FeedForwardNN(input_dims, n_actions, self.alpha, HP=self.HP)
        self.critic = FeedForwardNN(input_dims, 1, self.alpha, actor=False, HP=self.HP)
        self.memory = PPOMemory(batch_size)
        self.n_epochs = n_epochs

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self, path):
        # print("... saving models ...")
        self.actor.save_network(path)
        self.critic.save_network(path)

    def choose_action(self, observation):
        state = torch.tensor(np.array([observation]), dtype=torch.float32)
        state = state.to(self.actor.device)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, probs, value

    def learn(self):
        """
        Perform the PPO optimization process. This function updates the actor (policy)
        and critic (value function) networks using the collected experiences stored
        in the replay memory.

        The update is done over multiple epochs,
        processing the data in mini-batches to stabilize the training.
        """
        for _ in range(self.n_epochs):
            # Generate batches of data from memory
            (
                state_arr,  # States observed during interaction
                action_arr,  # Actions taken
                old_prob_arr,  # Log-probabilities of the action
                # (from the policy at the time of collection)
                vals_arr,  # Value function estimates at the time of collection
                reward_arr,  # Rewards received
                dones_arr,  # Flags indicating episode termination
                batches,  # Mini-batches indices for training
            ) = self.memory.generate_batches()

            values = vals_arr  # Value estimates from the critic
            advantage = np.zeros(
                len(reward_arr), dtype=np.float32
            )  # Initialize advantage array

            # Compute Generalized Advantage Estimation (GAE)
            # GAE advantage formula:
            # A_t = sum_{k=t}^T (γ^k-t * λ^(k-t) * δ_k),
            # where δ_k = r_k + γ * V(s_{k+1}) * (1 - done_k) - V(s_k)
            for t in range(len(reward_arr) - 1):
                discount = 1  # Initialize discount factor γ^k-t
                a_t = 0  # Initialize advantage estimate for time step t
                for k in range(t, len(reward_arr) - 1):
                    # Compute the temporal difference error δ_k
                    delta_k = (
                        reward_arr[k]
                        + self.gamma * values[k + 1] * (1 - int(dones_arr[k]))
                        - values[k]
                    )
                    a_t += discount * delta_k  # Add discounted TD error to advantage
                    discount *= (
                        self.gamma * self.gae_lambda
                    )  # Update discount factor with λ
                advantage[t] = a_t  # Store computed advantage for time step t

            # Convert advantage to tensor for PyTorch computations
            advantage = torch.tensor(advantage, dtype=torch.float32).to(
                self.actor.device
            )
            values = torch.tensor(values, dtype=torch.float32).to(
                self.actor.device
            )  # Convert values to tensor in float32

            # Process data in mini-batches
            for batch in batches:
                # Prepare mini-batch tensors
                states = torch.tensor(state_arr[batch], dtype=torch.float32).to(
                    self.actor.device
                )
                old_probs = torch.tensor(old_prob_arr[batch], dtype=torch.float32).to(
                    self.actor.device
                )
                actions = torch.tensor(action_arr[batch], dtype=torch.float32).to(
                    self.actor.device
                )

                # Forward pass through the actor to get the new action distribution
                dist = self.actor(states)

                # Forward pass through the critic to get value estimates
                critic_value = self.critic(states)
                critic_value = torch.squeeze(
                    critic_value
                )  # Remove unnecessary dimensions

                # Calculate new log-probabilities of the actions from the current policy
                new_probs = dist.log_prob(actions)

                # Compute the probability ratio: r_t = π_θ(a|s) / π_θ_old(a|s)
                prob_ratio = new_probs.exp() / old_probs.exp()
                # Alternatively: prob_ratio = (new_probs - old_probs).exp()

                # Compute the actor loss (surrogate objective with clipping)
                # L_clip = min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = (
                    torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                    * advantage[batch]
                )
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                # Compute the critic loss
                # (squared error between returns and value estimates)
                # L_value = (R_t - V(s))^2
                returns = advantage[batch] + values[batch]  # R_t = A_t + V(s)
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                # Total loss combines actor and critic losses
                # Total loss = L_clip + c1 * L_value
                # (with c1=0.5 as weighting factor for critic loss)
                total_loss = actor_loss + 0.5 * critic_loss

                # Backpropagation and optimization step
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        # Clear the memory after each optimization step
        self.memory.clear_memory()
