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

    def split_episodes(self, rewards, values, dones):
        """
        Split data into individual episodes based on the `dones` flags.
        """
        episodes_rewards, episodes_values, episodes_dones = [], [], []
        start = 0

        for i, done in enumerate(dones):
            if done:  # End of an episode
                episodes_rewards.append(rewards[start : i + 1])
                episodes_values.append(values[start : i + 1])
                episodes_dones.append(dones[start : i + 1])
                start = i + 1

        return episodes_rewards, episodes_values, episodes_dones

    def calculate_gae(self, rewards, values, dones):
        """
        Calculate Generalized Advantage Estimation (GAE) for multiple episodes.
        """
        all_advantages = []

        for ep_rews, ep_vals, ep_dones in zip(rewards, values, dones):
            advantages = []
            last_advantage = 0

            # Backward pass for GAE calculation
            for t in reversed(range(len(ep_rews))):
                if t + 1 < len(ep_rews):
                    delta = (
                        ep_rews[t]
                        + self.gamma * ep_vals[t + 1] * (1 - ep_dones[t + 1])
                        - ep_vals[t]
                    )
                else:
                    delta = ep_rews[t] - ep_vals[t]

                advantage = (
                    delta
                    + self.gamma * self.gae_lambda * (1 - ep_dones[t]) * last_advantage
                )
                last_advantage = advantage
                advantages.insert(0, advantage)

            all_advantages.extend(advantages)  # Concatenate advantages

        return torch.tensor(all_advantages, dtype=torch.float)

    def learn(self):
        """
        Perform the PPO optimization process. This function updates the actor (policy)
        and critic (value function) networks using the collected experiences stored
        in the replay memory.

        The update is done over multiple epochs,
        processing the data in mini-batches to stabilize the training.
        """
        batch_loss = []
        batch_actor_loss = []
        batch_critic_loss = []
        for _ in range(self.n_epochs):
            epoch_loss = []
            epoch_actor_loss = []
            epoch_critic_loss = []
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

            values = torch.tensor(values, dtype=torch.float32).to(
                self.actor.device
            )  # Convert values to tensor in float32

            # Split into episodes
            episodes_rewards, episodes_values, episodes_dones = self.split_episodes(
                reward_arr, vals_arr, dones_arr
            )

            # Calculate advantages
            advantages = self.calculate_gae(
                episodes_rewards, episodes_values, episodes_dones
            )
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

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
                prob_ratio = (new_probs - old_probs).exp()

                # Compute the actor loss (surrogate objective with clipping)
                # L_clip = min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)
                weighted_probs = advantages[batch] * prob_ratio
                weighted_clipped_probs = (
                    torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                    * advantages[batch]
                )
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                # Compute the critic loss
                # (squared error between returns and value estimates)
                # L_value = (R_t - V(s))^2
                returns = advantages[batch] + values[batch]  # R_t = A_t + V(s)
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
                epoch_loss.append(total_loss.item())
                epoch_actor_loss.append(actor_loss.item())
                epoch_critic_loss.append(critic_loss.item())
            batch_loss.append(np.mean(epoch_loss))
            batch_actor_loss.append(np.mean(epoch_actor_loss))
            batch_critic_loss.append(np.mean(epoch_critic_loss))
        # Clear the memory after each optimization step
        self.memory.clear_memory()
        return batch_loss, batch_actor_loss, batch_critic_loss
