from env import GYM_ENV
from agent import Agent
import numpy as np
from HP import _HP, get_script_arguments
from Logger import WandbLogger
from tqdm import tqdm
from collections import deque

HP = _HP("src/HP.yaml")
HP.update(get_script_arguments(HP.keys()))

if HP["LOG"]:
    WandbLogger.init(None, 3, HP)

if __name__ == "__main__":
    env = GYM_ENV(HP["ENV_NAME"])
    N = HP["UPDATA_EVERY"]  # Number of steps before updating the policy
    batch_size = HP["BATCH_SIZE"]  # Size of mini-batches for PPO updates
    n_epochs = HP["EPOCHS_TRAIN"]  # Number of epochs to train on the collected data
    alpha = HP["ALPHA"]  # Learning rate for both actor and critic
    gamma = HP["GAMMA"]  # Discount factor
    gae_lambda = HP["GAE_LAMBDA"]  # Generalized Advantage Estimation lambda
    policy_clip = HP["POLICY_CLIP"]  # Clipping parameter for the PPO policy loss
    agent = Agent(
        HP=HP,
        input_dims=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        batch_size=batch_size,
        n_epochs=n_epochs,
        alpha=alpha,
        gamma=gamma,
        gae_lambda=gamma,
        policy_clip=gamma,
    )

    n_games = HP["N_EPISODES"]  # Total number of episodes to train

    figure_file = "plots/cartpole.png"

    # Initialize variables to track performance
    score_history = deque(maxlen=50)  # List to store scores for each episode
    learn_iters = 0  # Counter for number of learning steps
    n_steps = 0  # Total number of steps taken
    avg_score = deque(maxlen=50)  # List to store average scores over 50 episodes
    best_score = -np.inf  # Best score so far

    for i in tqdm(range(n_games)):
        if i % N == 0 and i > 0:
            tot_loss, actor_losses, critic_losses = agent.learn()
            if HP["LOG"]:
                for tot, ac_loss, cr_loss in zip(tot_loss, actor_losses, critic_losses):
                    learn_iters += 1  # Increment learning step counter
                    logs = {
                        "total_loss": tot,
                        "actor_loss": ac_loss,
                        "critic_loss": cr_loss,
                        "loss_step": learn_iters,
                    }
                    WandbLogger.log(logs)

        observation, _ = env.reset()
        done = False
        score = 0  # Accumulated reward for the current episode
        rews = []  # Rewards for the current episode
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info, done = env.step(action)
            n_steps += 1  # Increment step counter
            score += reward
            rews.append(reward)
            agent.remember(observation, action, prob, val, reward, done)
            # Update the policy and value networks every N steps
            observation = observation_
        avg_score.append(score)
        avg_score_50 = np.mean(avg_score)
        logs = {
            "score": score,
            "avg_score_50": avg_score_50,
            "mean_reward": np.mean(rews),
        }
        if HP["LOG"]:
            WandbLogger.log(logs)
        if avg_score_50 > best_score:
            best_score = avg_score_50
            WandbLogger.log_model(agent.save_models, logs["avg_score_50"], i)

    env.close()
    if HP["LOG"]:
        WandbLogger.close()
    print("... training complete ...")
