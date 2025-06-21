# Ultrakill LLM Bot

This project contains tools to train and run a reinforcement learning agent for the game **ULTRAKILL**. The scripts are designed for Windows and use a Conda environment.

## Setup
1. Install [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. Run the setup script to create and populate the `ultrakill` environment:

```bash
bash setup_conda_env.sh
conda activate ultrakill
```

The script installs Python 3.11 along with all required packages.

## Training
To train the agent, run:

```bash
python train_ultrakill.py
```

Training checkpoints are written to `checkpoints/`. While training runs you can control it with hotkeys:

- **Backspace** – pause or resume the environment. All movement keys are released while paused.
- **Esc** – stop training immediately. The current model is saved before exiting.

## Running the AI
Once you have a trained checkpoint you can start the driver:

```bash
python ultrakill_ai.py ppo path/to/your_checkpoint.zip
```

If no checkpoint path is supplied, the driver tries `ppo_ultrakill_curiosity.zip` by default. The game window is focused automatically and the AI begins controlling the player.

During gameplay you can use the following hotkeys:

- **Backspace** – pause or resume the AI.
- **Esc** – quit the program.
- **F1** – toggle debug logging.

All keys are released on exit so it is safe to type afterwards.
