import gymnasium as gym


class GYM_ENV:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.observation, self.info = self.env.reset()
        self.episode_over = False
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def step(self, action):
        self.observation, self.reward, self.terminated, self.truncated, self.info = (
            self.env.step(action)
        )
        self.episode_over = self.terminated or self.truncated
        return (
            self.observation,
            self.reward,
            self.terminated,
            self.truncated,
            self.info,
            self.episode_over,
        )

    def reset(self):
        self.observation, self.info = self.env.reset()
        self.episode_over = False
        return self.observation, self.info

    def close(self):
        self.env.close()

    def print_info(self):
        print(
            f"Observation: {self.observation}, Reward: {self.reward}, "
            f"Terminated: {self.terminated}, Truncated: {self.truncated}, "
            f"Info: {self.info}, Episode Over: {self.episode_over}"
        )


# env = gym.make("CartPole-v1", render_mode="human")
# observation, info = env.reset()

# episode_over = False
# while not episode_over:
#     action = (
#         env.action_space.sample()
#     )  # agent policy that uses the observation and info
#     observation, reward, terminated, truncated, info = env.step(action)

#     episode_over = terminated or truncated

# env.close()
