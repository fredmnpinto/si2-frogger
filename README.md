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

The project includes a full DQN training pipeline for learning a Frogger agent.

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


