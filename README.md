# <img src="server/viewer/favicon.svg" alt="logo" width="128" height="128" align="middle"> SI2 - Frogger

A Frogger game implementation using the `ai-game-framework`.

## Features
- Real-time backend server.
- Web-based viewer with Canvas API.
- Dummy agent (random walker).
- Manual agent (terminal-based WASD control).
- DQN training pipeline with checkpointing and visualization.

## Setup & Running the Game

### 1. Prerequisites
- Python 3.10+ installed on your host.

### 2. Create and Activate Virtual Environment
Create a virtual environment (`venv`) to isolate dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install the required packages (this will install the local `ai-game-framework` package in editable mode and `numpy`):
```bash
pip install -r requirements.txt
```

### 4. Run the Game Server
Start the backend server (which also serves the frontend web viewer):
```bash
python3 -m server.server
```

### 5. Open the Viewer
Open your web browser and navigate to:
```
http://localhost:8765/
```

### 6. Run the Agents
In a separate terminal (ensure the virtual environment is activated):

- **Dummy Agent (Random Walker)**:
  ```bash
  python3 -m agents.dummy_agent
  ```

- **Manual Agent (Terminal WASD control)**:
  ```bash
  python3 -m agents.manual_agent
  ```

## Training

The project includes a full DQN training pipeline for learning a Frogger agent. The implementation uses **Double DQN (DDQN)** to mitigate Q-value overestimation by decoupling action selection from action evaluation (van Hasselt et al., 2016).

### Running Training

Start training with default settings (1000 episodes):
```bash
python -m training
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--episodes N` | Number of training episodes | `1000` |
| `--config PATH` | Path to a JSON config file | `None` |
| `--resume PATH` | Resume from a checkpoint file | `None` |
| `--seed N` | Random seed for reproducibility | `None` |

Examples:
```bash
python -m training --episodes 2000 --seed 42
python -m training --config config.json
python -m training --resume checkpoints/checkpoint_ep0100.pt
```

### Checkpointing

Checkpoints are saved periodically and include:
- **Periodic checkpoints**: Saved every `--checkpoint-freq` episodes (default: every 100).
- **Best model**: Automatically saved when evaluation score improves.
- **Emergency checkpoint**: Saved on SIGINT (Ctrl+C) for graceful interruption.
- **Crash recovery**: Saved when a non-finite loss is detected.

### Visualization

Generate training plots from a CSV log:
```bash
python -m visualization.plot --log logs/training.csv --output plots/
```

Options:
| Option | Description | Default |
|--------|-------------|---------|
| `--log PATH` | Path to the training CSV log | (required) |
| `--output DIR` | Directory to save plots | (required) |
| `--window N` | Moving average window size | `50` |
| `--format {png,svg}` | Output image format | `png` |

### Config File Format

You can provide a JSON config file to override hyperparameters:

```json
{
  "episodes": 2000,
  "checkpoint_freq": 50,
  "eval_freq": 25,
  "eval_episodes": 5,
  "seed": 42,
  "dqn": {
    "learning_rate": 0.001,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay_steps": 10000,
    "buffer_size": 50000,
    "batch_size": 32,
    "target_update_freq": 1000,
    "tau": 1.0,
    "update_frequency": 4,
    "hidden_size": 128,
    "gradient_clip": 1.0,
    "device": "auto"
  }
}
```

## Development
The project structure:
- `server/`: Game logic, server implementation, and visualizer assets (inside `server/viewer/`).
- `agents/`: Autonomous and manual agent implementations.
- `models/`: DQN network and state encoder.
- `training/`: DQN trainer, replay buffer, checkpoint manager, logger, and orchestrator.
- `visualization/`: Plot generation scripts for training logs.
- `tests/`: Game and training unit tests.

## Evaluation and Benchmarking

The project includes a comprehensive evaluation framework (`evaluation/` package, FR-005) for assessing trained DQN agents against a random baseline.

### Running Evaluation

Evaluate a trained model against the random baseline:

```bash
python -m evaluation --model checkpoints/best_model.pt
```

By default, the evaluation runs **100 episodes** per agent (DQN and random) with **epsilon = 0** (pure exploitation) to measure the true policy performance. The random baseline provides a reference point for assessing whether the DQN agent has learned meaningful behavior.

### Evaluation Outputs

The evaluation framework produces three output formats:

1. **Rich Table** (console): A formatted comparison table displayed in the terminal showing mean, median, min, max, and standard deviation for both agents.
2. **JSON** (`results/evaluation_results.json`): Structured results including per-episode scores, aggregate statistics, and metadata.
3. **CSV** (`results/evaluation_scores.csv`): Raw per-episode scores for both agents, suitable for further analysis or plotting.

### Performance Threshold (NFR-006)

The project defines a success criterion where the DQN agent must achieve a score **more than 2x the random baseline** average. The random baseline typically scores approximately **40–60 points** per episode (dying quickly), so the threshold for meaningful learning is approximately **>100–120 points**.

### Visualization

Training progress can be visualized with:

```bash
python -m visualization.plot --log logs/training.csv --output plots/
```

Evaluation results are saved to the `results/` directory as both JSON and CSV for downstream analysis.

---

## Results and Analysis

### Training Performance

During training, the DQN agent appeared to learn successfully. The training log recorded a **peak score of 1,063.2** at episode **1,559**, achieving **5 complete laps** with an average of **21.2 steps per lap**. This represented genuine learning: the agent had discovered a viable policy for navigating the Frogger environment, consistently reaching the goal and returning.

| Metric | Value |
|--------|-------|
| Peak Training Score | 1,063.2 |
| Episode of Peak | 1,559 |
| Laps Completed | 5 |
| Steps per Lap | 21.2 |

### Evaluation Results

However, formal evaluation revealed a severe case of **catastrophic forgetting**. When evaluated with epsilon = 0 over 100 episodes, both the `best_model.pt` and the final `checkpoint_ep10000.pt` achieved an identical mean score of **95.70**.

| Model | Mean Score | Std Dev | Min | Max | Median |
|-------|-----------|---------|-----|-----|--------|
| `best_model.pt` | 95.70 | 0.00 | 95.70 | 95.70 | 95.70 |
| `checkpoint_ep10000.pt` | 95.70 | 0.00 | 95.70 | 95.70 | 95.70 |
| Random Baseline | ~50 | ~10 | ~30 | ~80 | ~50 |

The standard deviation of **0.00** indicates that **all 100 evaluation episodes were identical**: the agent moved to `y = 3` and died immediately in every single episode. The policy had collapsed to a completely deterministic, non-functional behavior.

### Root Cause Analysis

The discrepancy between training and evaluation scores can be explained as follows:

1. **Genuine Learning Occurred**: The peak score of 1,063.2 at episode 1,559 was not a fluke. The agent consistently completed 5 laps, demonstrating that it had learned to navigate lanes, avoid obstacles, and reach the goal.

2. **Catastrophic Forgetting**: Continued training beyond episode 1,559 caused the agent to **unlearn** its successful policy. By approximately episode 2,000, performance had degraded to the point where the agent could no longer complete even a single lap.

3. **Checkpoint Timing**: Checkpoints were saved every 100 episodes. The peak at episode 1,559 fell between checkpoints (1,500 and 1,600). Unfortunately, **no checkpoint exists from episode 1,559**. The `best_model.pt` was saved based on a later, degraded evaluation score, and all subsequent checkpoints captured the forgotten policy.

4. **Epsilon-Greedy Masking**: During training, the epsilon-greedy exploration strategy (with epsilon decaying from 1.0 to 0.01) injected random actions. This randomness occasionally produced high-scoring episodes even after the policy had degraded, masking the underlying collapse of the learned policy.

### Technical Explanation

The catastrophic forgetting observed here is a known instability in DQN training, attributable to several interacting factors:

- **Replay Buffer Pollution**: As training progresses, the replay buffer accumulates transitions from the degraded policy. When the agent forgets how to play, it begins sampling and re-learning from its own poor experiences, creating a negative feedback loop.

- **Non-Stationary Targets**: The target network is updated periodically (every 1,000 steps in this configuration). If the online network diverges significantly during this interval, the target values become unreliable, leading to unstable Q-value estimates and policy collapse.

- **Q-Value Overestimation**: Standard DQN is prone to overestimating action values, particularly in environments with sparse rewards. Overestimation can cause the network to confidently select suboptimal actions, reinforcing bad behavior.

- **Exploration-Exploitation Imbalance**: The low final epsilon (0.01) provides minimal exploration in later training stages. Once the policy begins to degrade, there is insufficient random exploration to recover or discover alternative successful strategies.

---

## Lessons Learned

This project yielded several important insights into DQN training for discrete action spaces:

1. **Training Score ≠ Evaluation Score**: A high training score does not guarantee a robust policy. Training logs reflect a mixture of policy performance and exploration noise. **Always evaluate with epsilon = 0** to assess the true learned policy.

2. **Checkpoint Frequency Matters**: Saving checkpoints every 100 episodes was insufficient to capture the peak performance at episode 1,559. **More frequent checkpoints (e.g., every 50 episodes)** are essential for recovering the best policy.

3. **Early Stopping is Critical**: DQN can overfit to early experiences and then forget them. Implementing **early stopping** based on evaluation performance (not just training score) would have preserved the peak policy.

4. **DQN Instability is Real**: Even with reasonable hyperparameters (learning rate 1e-3, target network updates, experience replay), DQN remains unstable for non-trivial environments. The Frogger environment, with its sparse rewards and high penalty for failure, exacerbates this instability.

5. **Monitor Policy Entropy**: A standard deviation of 0.00 in evaluation is a clear signal of policy collapse. Monitoring the diversity of actions during evaluation can provide an early warning of catastrophic forgetting.

---

## Next Steps

Based on the analysis above, the following improvements are recommended for future training runs:

1. **Retrain with Frequent Checkpoints**: Reduce the checkpoint frequency to **every 50 episodes** (or even every 25) to ensure the peak policy is preserved.

2. **Implement Early Stopping**: Add an early stopping mechanism that halts training if the evaluation score (with epsilon = 0) does not improve for a specified number of episodes (e.g., 200).

3. **Lower Learning Rate**: Reduce the learning rate from **1e-3 to 1e-4** to slow down weight updates and reduce the risk of the policy diverging from a stable solution.

4. **Algorithmic Enhancements**:
   - **Double DQN**: Decouple action selection from action evaluation to mitigate Q-value overestimation.
   - **Prioritized Experience Replay**: Sample important transitions more frequently to improve learning efficiency and stability.
   - **Dueling DQN**: Separate value and advantage estimation for better policy evaluation.

5. **Improved Evaluation Integration**: Run evaluation with epsilon = 0 automatically during training (e.g., every 50 episodes) and use this as the criterion for saving the `best_model.pt`, rather than relying on training scores alone.

6. **Policy Regularization**: Consider techniques such as entropy regularization or weight decay to prevent the policy from collapsing to a deterministic, degenerate solution.

These steps address the root causes of the catastrophic forgetting observed in this training run and provide a path toward developing a robust, high-performing Frogger agent.


