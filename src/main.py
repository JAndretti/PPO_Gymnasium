from env import GYM_ENV
from agent import Agent
import numpy as np
from HP import _HP, get_script_arguments
from Logger import WandbLogger
from tqdm import tqdm

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
    best_score = 0  # Best score achieved so far (initialized to minimum reward)
    score_history = []  # List to store scores for each episode
    learn_iters = 0  # Counter for number of learning steps
    avg_score = 0  # Average score over the last 100 episodes
    n_steps = 0  # Total number of steps taken
    max_avg_score = 0  # Maximum average score achieved so far

    for i in tqdm(range(n_games)):
        observation, _ = env.reset()
        done = False
        score = 0  # Accumulated reward for the current episode
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info, done = env.step(action)
            n_steps += 1  # Increment step counter
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            # Update the policy and value networks every N steps
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1  # Increment learning step counter
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-(int(HP["N_EPISODES"] * 0.1)) :])
        if i == 0:
            best_score = avg_score
            max_avg_score = avg_score
        if avg_score > max_avg_score:
            max_avg_score = avg_score
        logs = {
            "score": score,
            "avg_score": avg_score,
            "max_avg_score": max_avg_score,
        }
        if HP["LOG"]:
            WandbLogger.log(logs)
        if avg_score > best_score:
            best_score = avg_score
            WandbLogger.log_model(agent.save_models, logs["avg_score"], i)

        # print(
        #     "episode",
        #     i,
        #     "score %.1f" % score,
        #     "avg score %.1f" % avg_score,
        #     "time_steps",
        #     n_steps,
        #     "learning_steps",
        #     learn_iters,
        # )

    env.close()
    if HP["LOG"]:
        WandbLogger.close()
    print("... training complete ...")
