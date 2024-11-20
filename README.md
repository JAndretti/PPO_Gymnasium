# PPO Project with Gymnasium and Wandb

This project implements a reinforcement learning agent using the PPO (Proximal Policy Optimization) algorithm in Gymnasium simulation environments. Training curves and models are visualized and logged using Wandb.

## Project Structure
├── launch_HP_sweep.py
├── src/ │
    ├── agent.py │
    ├── env.py │
    ├── HP_sweep.yaml │
    ├── HP.py │
    ├── HP.yaml │
    ├── Logger.py │
    ├── main.py │
    ├── memory.py │
    ├── network.py │
    ├── test_actor.py

### Main Files

- **launch_HP_sweep.py**: Script to launch experiments with different hyperparameter configurations in parallel on multiple GPUs.
- **src/**: Directory containing the project's source files.

### Files in `src/`

- **agent.py**: Contains the `Agent` class that implements the PPO agent, with methods to choose actions, remember experiences, and learn from those experiences.
- **env.py**: Contains the `GYM_ENV` class that encapsulates the Gymnasium environment and provides methods to interact with it.
- **HP_sweep.yaml**: YAML configuration file for hyperparameters used in hyperparameter sweeps.
- **HP.py**: Contains the singleton `_HP` class for managing hyperparameters, as well as utility functions for script arguments.
- **HP.yaml**: YAML configuration file for default hyperparameters.
- **key.txt**: Contains the API key for Wandb.
- **Logger.py**: Contains the `WandbLogger` class for logging training curves and models to Wandb.
- **main.py**: Main script to initialize the environment and agent, and manage the training loop.
- **memory.py**: Contains the `PPOMemory` class for managing the PPO agent's memory.
- **network.py**: Contains the `FeedForwardNN` class that defines the neural network used as the actor and critic in the PPO algorithm.
- **test_actor.py**: Script to test the trained agent in the simulation environment.

## Usage

### Launch an Experiment

To launch an experiment with a specific hyperparameter configuration, use the `launch_HP_sweep.py` script:
