# SI2-Frogger DQN Agent Functional Specification

> **Version**: 2.3.0 | **Date**: 2026-06-24 | **Author**: Documenter Agent | **Status**: Draft

## Change Log
| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 2.3.0 | 2026-06-24 | Documenter Agent | FR-004 (Inference Agent) marked as Implemented. DQNAgent subclassing BaseAgent with checkpoint loading, StateEncoder for WebSocket state conversion, greedy action selection (epsilon=0), STAY action masking, CLI entry point (`python -m agents.dqn_agent`), 19 unit tests, ~93% coverage. Commit: `d6c8f0bc`. |
| 2.2.0 | 2026-06-24 | Documenter Agent | Updated spec to match current implementation: 5 actions (added STAY), 30-D state with 8 directional sensors, MLP 256x256 (~75K params), Rich Live display with high score tracker and achievement banner, per-lap (200) and total (2000) step limits, 5 plots (added steps_per_lap), updated reward function details, updated FR statuses (FR-002, FR-003, FR-006 Implemented; FR-004, FR-005 Draft), updated dependencies (added rich, removed tqdm). |
| 2.1.2 | 2026-06-21 | Documenter Agent | FR-003 (Training Orchestration) marked as Implemented. TrainingOrchestrator with episode loop, tqdm progress bar, CheckpointManager with periodic saves/best model tracking/emergency recovery, TrainingLogger with console+CSV logging, TrainingConfig with CLI parser and JSON config support, visualization/plot.py with 4 plot types, DQNTrainer save/load checkpoint methods, CLI entry points, SIGINT handling, NaN/Inf detection. 176 tests passing, 99% coverage. Commit: `9d20902`. |
| 2.1.1 | 2026-06-20 | Documenter Agent | FR-001 (Environment Wrapper) marked as Implemented. `env/__init__.py` and `env/frogger_env.py` created with `FroggerEnv` class wrapping `server.logic.Frogger`. All acceptance criteria checked off. 29 unit tests, 98% coverage. Commit: `4f5e292`. |
| 2.1.0 | 2026-06-20 | Documenter Agent | Fixed granularity: consolidated 14 sub-function FRs into 6 User Goal-level FRs per IREB best practices. Moved state representation, reward function, and network architecture design decisions to Architecture section. Acceptance criteria now carry sub-function detail. |
| 2.0.0 | 2026-06-20 | Documenter Agent | Complete rewrite: shifted focus from game server to DQN agent/training system. Previous spec (v1.0.0) documented the existing game implementation which is out of scope for this project. |

---

## 1. Introduction

### 1.1 Purpose
This document specifies the functional requirements for the **SI2-Frogger DQN Agent** — a university project to develop an autonomous Deep Q-Network (DQN) agent that learns to play the Frogger game via Reinforcement Learning (RL). The game server, logic, and viewer are pre-existing and provided as a stable platform. This spec defines the agent architecture, training pipeline, evaluation methodology, and deliverables.

### 1.2 Scope

**In Scope**:
- DQN agent architecture (neural network, replay buffer, target network)
- Gym-like environment wrapper around `server/logic.py` for offline training
- State representation design and feature engineering from raw game state
- Reward function design shaping agent behavior
- Epsilon-greedy exploration strategy with decay
- Offline training loop with logging, checkpointing, and convergence monitoring
- Model save/load mechanism for trained weights
- `BaseAgent` subclass for WebSocket-based inference gameplay
- Hyperparameter configuration and analysis
- Training visualization (convergence graphs, loss curves, reward plots)
- Evaluation statistics and performance benchmarking
- Comprehensive `README.md` serving as the project report
- Unit tests for all agent/training components (target: >=85% coverage)

**Out of Scope (Non-Goals)**:
- Modification of the existing game server (`server/server.py`), game logic (`server/logic.py`), or viewer (`server/viewer/`)
- Multiplayer or networked training (training is offline via direct Python imports)
- Real-time training via WebSocket (training uses imported `Frogger` class directly)
- Alternative RL algorithms (PPO, A3C, etc.) — DQN is the mandated approach
- Distributed training or cloud compute orchestration
- Sound effects, visual enhancements, or UI modifications to the game
- Persistent leaderboard or database storage

### 1.3 Audience
- **Students**: Implementing the DQN agent and training pipeline for the SI2 course project
- **Evaluators**: Grading the solution based on effectiveness, code quality, and report clarity
- **Developers**: Extending or reproducing the agent architecture

### 1.4 References
- [project02.md](project02.md) - Course project description and grading criteria
- `server/logic.py` - Existing Frogger game logic (external dependency, read-only)
- `agents/base_agent.py` - Existing BaseAgent class for WebSocket inference (external dependency, subclassed)
- Mnih et al. (2015) - "Human-level control through deep reinforcement learning" (DQN reference paper)

---

## 2. System Description

### 2.1 Current State (As-Is)
The project starts with a fully functional Frogger game implementation:
- **Game Engine**: `server/logic.py` provides `Frogger` class with grid-based movement (11x9), 6 traffic lanes, collision detection, checkpoint/lap scoring, and 3 lives
- **Game Server**: `server/server.py` provides WebSocket API at `ws://localhost:8765/ws` running at 30 FPS
- **Agent Framework**: `agents/base_agent.py` provides abstract `BaseAgent` with WebSocket client, state reception, and action transmission
- **Baseline Agents**: `DummyAgent` (random walker) and `ManualAgent` (WASD terminal control) exist as reference implementations
- **Testing**: Unit tests exist for core game logic in `tests/test_logic.py`
- **No ML/RL code exists yet** — this is the deliverable of the project

### 2.2 Target State (To-Be)
The project delivers a complete DQN-based RL system:
- **Environment Wrapper**: A Gym-like wrapper (`FroggerEnv`) around `server/logic.Frogger` exposing `reset()`, `step(action)`, and state extraction
- **State Representation**: 32-dimensional engineered feature vector capturing frog position, lives, checkpoint flag, per-lane obstacle distance/speed/width (6 lanes × 3), and 8 directional proximity sensors (N, S, E, W, NE, NW, SE, SW)
- **DQN Network**: PyTorch MLP mapping state vectors to Q-values for 5 discrete actions (NORTH, SOUTH, EAST, WEST, STAY). Architecture: Input[32] → Linear(32, 64) + ReLU → Linear(64, 64) + ReLU → Linear(64, 5). ~6,600 parameters.
- **Replay Buffer**: Fixed-size experience replay storing `(state, action, reward, next_state, done)` tuples
- **Training Pipeline**: Offline training script importing `server/logic.py` directly, running thousands of episodes with epsilon-greedy exploration
- **Target Network**: Separate target network updated periodically for training stability
- **Checkpointing**: Model weights saved at intervals and best-model selection based on evaluation score
- **Inference Agent**: `DQNAgent` subclass of `BaseAgent` loading trained weights and playing via WebSocket
- **Evaluation & Visualization**: Scripts generating training curves, convergence graphs, hyperparameter comparison tables, and performance statistics
- **Project Report**: `README.md` documenting architecture, state representation, model design, reward function, training logs, and evaluation

### 2.3 Project Goals
- **G1**: Train a DQN agent that achieves significantly better-than-random performance (score > DummyAgent average)
- **G2**: Complete training in reasonable wall-clock time (target: <24 hours on standard laptop CPU/GPU)
- **G3**: Achieve >=85% unit test coverage for all agent/training code
- **G4**: Produce clear training convergence graphs and hyperparameter analysis in README.md
- **G5**: Deliver a reproducible training pipeline with configurable hyperparameters
- **G6**: Pass WebSocket inference test — trained agent plays successfully via `BaseAgent` subclass

---

## 3. Functional Requirements

| ID | Title | Description | Priority (Must/Should/Could) | Source | Dependencies | Status |
|----|-------|-------------|-------------------------------|--------|--------------|--------|
| FR-001 | Environment Wrapper | The system shall provide a Gym-like wrapper around Frogger logic exposing reset(), step(), state extraction, reward computation, and cooldown handling | Must | project02.md, codebase | None | Implemented |
| FR-002 | DQN Training System | The system shall implement a complete DQN algorithm including neural network, experience replay, epsilon-greedy exploration, target network updates, loss computation, and optimizer | Must | project02.md, DQN paper | FR-001 | Implemented |
| FR-003 | Training Orchestration | The system shall run a training loop over N episodes with logging, checkpointing, and visualization | Must | project02.md | FR-002 | Implemented |
| FR-004 | Inference Agent | The system shall provide a DQNAgent subclassing BaseAgent that loads trained weights and plays via WebSocket | Must | project02.md, codebase | FR-003 | Implemented |
| FR-005 | Evaluation and Benchmarking | The system shall evaluate trained agents over multiple episodes and compute statistics vs DummyAgent baseline | Must | project02.md | FR-004 | Implemented |
| FR-006 | Hyperparameter Configuration | The system shall expose all hyperparameters via config file or CLI | Should | project02.md | FR-003 | Implemented |

### Acceptance Criteria

#### FR-001: Environment Wrapper
**Description**: The system shall provide a Gym-like wrapper around `server.logic.Frogger` exposing `reset()`, `step()`, state extraction, reward computation, and cooldown handling.
**Priority**: Must
**Source**: project02.md, codebase
**Dependencies**: None
**Acceptance Criteria**:
- [x] `FroggerEnv` class wraps `server.logic.Frogger` without modifying `logic.py`
- [x] `reset()` returns initial state and resets the underlying `Frogger` game
- [x] `step(action)` accepts action string ("NORTH", "SOUTH", "EAST", "WEST", "STAY") or action index (0-4), advances game by one or more ticks, and returns `(next_state, reward, done, info)`
- [x] `step()` handles the 5-frame movement cooldown by either stepping multiple ticks or waiting appropriately
- [x] Environment tracks episode length and exposes it in `info`
- [x] Environment is deterministic when seeded
**Status**: Implemented

#### FR-002: DQN Training System
**Description**: The system shall implement a complete DQN algorithm including neural network architecture (PyTorch), experience replay buffer, epsilon-greedy exploration with decay, target network updates, loss computation (Huber/MSE), and Adam optimizer.
**Priority**: Must
**Source**: project02.md, DQN paper (Mnih et al., 2015)
**Dependencies**: FR-001
**Acceptance Criteria**:
- [x] Network maps states to 5 Q-values (NORTH, SOUTH, EAST, WEST, STAY)
- [x] Network supports both CPU and CUDA execution
- [x] Forward pass completes in <10ms for a single state
- [x] Replay buffer stores tuples of `(state, action, reward, next_state, done)`
- [x] Replay buffer has configurable maximum capacity (default: 100,000)
- [x] `push()` adds transitions; when full, oldest transitions are overwritten
- [x] `sample(batch_size)` returns random minibatch without replacement using vectorized operations
- [x] Epsilon starts at `epsilon_start` (default: 1.0) and decays to `epsilon_end` (default: 0.01) over configurable schedule
- [x] With probability epsilon, a random valid action is selected; with probability (1 - epsilon), the action with highest Q-value is selected
- [x] Target network is a copy of the policy network, initialized identically
- [x] Hard update: target network copies policy network weights every N steps (default: every 1,000 steps); soft update via Polyak averaging (`tau * policy + (1-tau) * target`) also supported
- [x] Target network is used for computing Q-targets in loss function, not for action selection
- [x] Loss is computed as Huber loss (SmoothL1Loss) between predicted Q and target Q
- [x] Optimizer is Adam with configurable learning rate (default: 5e-4)
- [x] Training updates occur every `update_frequency` steps (default: every 4 steps)
**Status**: Implemented

#### FR-003: Training Orchestration
**Description**: The system shall run a training loop over N episodes with logging, checkpointing, and visualization.
**Priority**: Must
**Source**: project02.md
**Dependencies**: FR-002
**Acceptance Criteria**:
- [x] Training script runs for configurable number of episodes (default: 1,000+)
- [x] Each episode: reset env, run until done, accumulate transitions, perform training updates
- [x] Logs include: episode number, total reward, epsilon, loss, episode length, high score, max_y, laps_completed, steps_per_lap
- [x] Logs are written to CSV file; console output uses Rich Live display
- [x] Rich Live display shows: episode count, percentage, bar, ETA (smoothed), EPS, Best score, Reward, Laps, MaxY
- [x] High score tracker: In-place Rich table showing last 3 best episodes
- [x] Achievement banner: Rich Panel showing "NEW BEST" for 8 seconds
- [x] All updates happen in-place without scrolling
- [x] Per-lap limit: 200 steps without completing a lap → episode ends
- [x] Total limit: 2000 steps absolute max per episode
- [x] Lap completion resets the per-lap timer
- [x] Model weights saved every N episodes (configurable, default: every 100)
- [x] Best model (highest evaluation score) is saved separately and never overwritten
- [x] Checkpoint includes: policy network state dict, target network state dict, optimizer state dict, episode count, epsilon value, best score
- [x] Checkpoints are saved to `checkpoints/` directory with descriptive filenames
- [x] Checkpoint loading restores full training state for resumption
- [x] Plot 1: Episode rewards over time (smoothed with moving average)
- [x] Plot 2: Loss curve over training steps
- [x] Plot 3: Epsilon decay over episodes
- [x] Plot 4: Score distribution (histogram and boxplot)
- [x] Plot 5: Steps per lap over time (smoothed with moving average)
- [x] Plots are saved as PNG/SVG to `plots/` or `results/` directory
- [x] Plot generation script is separate and can be run post-training
**Status**: Implemented

#### FR-004: Inference Agent
**Description**: The system shall provide a `DQNAgent` subclassing `BaseAgent` that loads trained weights and plays via WebSocket.
**Priority**: Must
**Source**: project02.md, codebase
**Dependencies**: FR-003
**Acceptance Criteria**:
- [x] `DQNAgent` extends `agents.base_agent.BaseAgent`
- [x] Implements `async deliberate()` method selecting action via trained DQN
- [x] Converts WebSocket state JSON to the same representation used during training
- [x] Selects action with epsilon=0 (pure exploitation) during gameplay
- [x] Handles `game_over` state by returning `None`
- [x] Successfully connects to `ws://localhost:8765/ws` and plays the game
- [x] Can be launched via `python3 -m agents.dqn_agent`
**Status**: Implemented

#### FR-005: Evaluation and Benchmarking
**Description**: The system shall evaluate trained agents over multiple episodes and compute statistics vs DummyAgent baseline.
**Priority**: Must
**Source**: project02.md
**Dependencies**: FR-004
**Acceptance Criteria**:
- [ ] Evaluation runs for configurable number of episodes (default: 100) with epsilon=0.05 (DeepMind protocol) or epsilon=0.00 (deterministic)
- [ ] Computes: mean score, max score, min score, standard deviation
- [ ] Computes: mean episode length, mean survival time, mean laps completed
- [ ] Compares against DummyAgent baseline over same number of episodes
- [ ] Results are saved to JSON/CSV and printed to console
- [ ] Evaluation script is separate from training script
**Status**: Implemented

#### FR-006: Hyperparameter Configuration
**Description**: The system shall expose all hyperparameters via config file or CLI.
**Priority**: Should
**Source**: project02.md
**Dependencies**: FR-003
**Acceptance Criteria**:
- [x] Configurable parameters: learning_rate, gamma, epsilon_start, epsilon_end, epsilon_decay_steps, buffer_size, batch_size, target_update_freq, update_frequency, hidden_size, gradient_clip, device
- [x] Configuration loaded from JSON file or CLI arguments (CLI overrides JSON)
- [x] Default values are sensible and documented
- [x] Multiple hyperparameter experiments can be run and compared
- [x] Hyperparameter values used for final model are documented in README.md
**Status**: Implemented

---

## 4. Non-Functional Requirements

| ID | Category | Description | Metric | Target |
|----|----------|-------------|--------|--------|
| NFR-001 | Performance | Training completion time | Wall-clock time | <24 hours on standard laptop (CPU); <6 hours with CUDA GPU |
| NFR-002 | Performance | Inference latency | Action selection time | <50ms per action on CPU |
| NFR-003 | Maintainability | Code modularity | Module separation | Clear separation: env/, models/, agents/, training/, evaluation/, utils/ |
| NFR-004 | Maintainability | Code readability | Style compliance | PEP 8 compliant, meaningful variable names, docstrings for all public methods |
| NFR-005 | Reliability | Test coverage | Coverage % | >=85% for all agent/training code (excluding `server/` and `agents/base_agent.py`) |
| NFR-006 | Performance | Agent performance | Mean evaluation score | >2x DummyAgent mean score over 100 episodes |
| NFR-007 | Usability | Documentation completeness | README sections | All required sections per project02.md: setup, architecture, state rep, model, reward, training curves, hyperparameters, evaluation |
| NFR-008 | Portability | Platform independence | OS support | Runs on Linux, macOS, Windows with Python 3.10+ |
| NFR-009 | Reproducibility | Determinism | Random seed | Training is reproducible when random seed is fixed |

---

## 5. Architecture Overview

### 5.1 Components

| Component | Purpose | Responsibilities |
|-----------|---------|-----------------|
| FroggerEnv | Environment Wrapper | Wraps `server.logic.Frogger`; exposes Gym-like API (`reset`, `step`, `state extraction`); computes rewards; handles cooldown |
| StateEncoder | State Representation | Converts raw game state (frog pos, obstacles) into fixed-size feature vector or image tensor |
| DQNNetwork | Neural Network | PyTorch `nn.Module` mapping state tensors to Q-values for 5 actions |
| ReplayBuffer | Experience Storage | Circular buffer storing transitions; supports uniform random sampling |
| DQNTrainer | Training Orchestrator | Runs training loop: episode generation, epsilon decay, loss computation, backpropagation, checkpointing |
| TargetNetwork | Stable Q-Target | Conceptual component: a deep copy of DQNNetwork maintained inside DQNTrainer, updated periodically for stable TD targets |
| DQNAgent | Inference Agent | Subclasses `BaseAgent`; loads trained weights; selects greedy actions via WebSocket |
| Evaluator | Performance Testing | Runs evaluation episodes, computes statistics, generates comparison tables |
| Plotter | Visualization | Conceptual component: plotting implemented as standalone functions in visualization/plot.py |
| ConfigManager | Hyperparameter Management | Conceptual component: hyperparameter management handled by TrainingConfig and EvaluationConfig dataclasses with CLI parsing |

### 5.2 Design Decisions

The following design decisions are documented in the Architecture section (not as standalone functional requirements) because they are implementation choices that support the User Goal-level requirements:

#### State Representation Design
The `StateEncoder` component converts raw game state into a fixed-size 32-dimensional feature vector. Key design choices:
- **Fixed dimensions**: 32-D vector regardless of obstacle count
- **Frog position**: Indices 0-1, normalized frog x and y coordinates
- **Lives**: Index 2, normalized lives (0-1)
- **Checkpoint flag**: Index 3, binary flag indicating whether the middle checkpoint has been reached
- **Per-lane features**: Indices 4-21, 6 traffic lanes × 3 features each (distance to nearest obstacle, obstacle speed, obstacle width), all normalized
- **Directional sensors**: Indices 22-29, 8 directional ray-casting sensors (N, S, E, W, NE, NW, SE, SW) returning normalized distance to nearest obstacle within range 3
- **Performance**: State can be converted to PyTorch tensor in <1ms
- **Documentation**: Representation dimensionality and encoding rationale must be documented in README.md
- **Alternatives**: At least one alternative representation must be considered and documented (e.g., image-like vs. vector)

#### Reward Function Design
The reward function is implemented in the environment wrapper (`FroggerEnv`) to shape agent behavior. Key design choices:
- **Forward progress**: +20 per new lane reached (only for new best y-position to prevent farming)
- **Checkpoint**: +100 for reaching the middle checkpoint
- **Lap completion**: +200 for completing a full lap
- **Death**: -50 for losing a life
- **Survival bonus**: +1.0 per step
- **Directional incentives**: +2 for NORTH action, -1 for SOUTH action
- **Stay penalty**: -0.5 for choosing STAY to discourage idle behaviour
- **Backward movement**: -2 for moving backward (unless lap was just completed)
- **Sparsity vs. density**: Dense reward shaping with small per-step penalties and larger event-based bonuses
- **Clipping**: Reward clipping (e.g., [-1, 1]) must be considered and documented
- **Documentation**: Reward function must be explicitly defined and documented in README.md

#### Network Architecture Details
The `DQNNetwork` component implements the neural network approximating the Q-function. Key design choices:
- **Input**: Accepts state tensor of shape `(batch, 32)` from `StateEncoder`
- **Output**: Q-values for [NORTH, SOUTH, EAST, WEST, STAY] — 5 discrete actions
- **Architecture**: MLP with two hidden layers: `Input[32] → Linear(32, 64) + ReLU → Linear(64, 64) + ReLU → Linear(64, 5)`
- **Parameter count**: ~6,600 parameters
- **Loss function**: Huber loss (`nn.SmoothL1Loss`)
- **Optimizer**: Adam with learning rate 5e-4
- **Justification**: MLP is appropriate for the engineered 32-D vector state; CNN would be used for image-like states
- **Device support**: Supports both CPU and CUDA execution with automatic fallback

### 5.3 Data Models

#### Model: Transition
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| state | Tensor | Yes | Encoded state representation (shape: [32]) |
| action | int | Yes | Action index: 0=NORTH, 1=SOUTH, 2=EAST, 3=WEST, 4=STAY |
| reward | float | Yes | Scalar reward from reward function |
| next_state | Tensor | Yes | Encoded next state |
| done | bool | Yes | True if episode terminated (game_over or win) |

#### Model: Checkpoint
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| policy_state_dict | dict | Yes | PyTorch state dict of policy network |
| target_state_dict | dict | Yes | PyTorch state dict of target network |
| optimizer_state_dict | dict | Yes | PyTorch state dict of optimizer |
| episode | int | Yes | Current training episode |
| epsilon | float | Yes | Current epsilon value |
| best_score | float | Yes | Best evaluation score achieved |

#### Model: EvaluationResult
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| mean_score | float | Yes | Mean score over evaluation episodes |
| max_score | float | Yes | Maximum score achieved |
| min_score | float | Yes | Minimum score achieved |
| std_score | float | Yes | Standard deviation of scores |
| mean_length | float | Yes | Mean episode length (steps) |
| mean_laps | float | Yes | Mean laps completed per episode |
| laps_completed | int | Yes | Total laps completed across all evaluation episodes |
| steps_per_lap | float | Yes | Mean steps per lap (lower is better) |

### 5.4 API Design

#### Gym-like Environment API
```python
# FroggerEnv
state = env.reset()                          # Returns initial encoded state
next_state, reward, done, info = env.step(action)  # Returns transition tuple
state_shape = env.observation_space          # Fixed shape of state vector (32)
n_actions = env.action_space                 # 5 (NORTH, SOUTH, EAST, WEST, STAY)
```

#### DQN Network API
```python
# DQNNetwork (PyTorch)
q_values = model(state_tensor)               # Forward pass: returns [batch, 5] Q-values
```

#### Replay Buffer API
```python
# ReplayBuffer
buffer.push(state, action, reward, next_state, done)  # Store transition
batch = buffer.sample(batch_size)            # Sample random minibatch
```

#### Trainer API
```python
# DQNTrainer
trainer = DQNTrainer(config)                 # Initialize with hyperparameters
trainer.train(n_episodes)                    # Run training loop
trainer.save_checkpoint(path)                # Save model and training state
trainer.load_checkpoint(path)                # Resume training
```

#### Inference Agent API
```python
# DQNAgent (extends BaseAgent)
agent = DQNAgent(model_path="checkpoints/best.pt")
asyncio.run(agent.run())                     # Connects via WebSocket and plays
```

---

## 6. Interfaces

### 6.1 User Interface
- **Command Line**: Training and evaluation scripts are CLI-based
  - `python -m training` — Run training with defaults (1000 episodes)
  - `python -m training --episodes 2000 --seed 42` — Override episodes and seed
  - `python -m training --config config.json` — Load hyperparameters from JSON
  - `python -m training --resume checkpoint.pt` — Resume from checkpoint
  - `python -m visualization.plot --log logs/training.csv --output plots/` — Generate plots post-training
  - `python -m agents.dqn_agent --model checkpoints/best.pt` — Run inference agent (planned)
- **Web Viewer**: Existing HTML5 Canvas viewer at `http://localhost:8765/` used to observe trained agent gameplay

### 6.2 External Integrations
| System | Purpose | Data Format | Protocol | Error Handling |
|--------|---------|-------------|----------|----------------|
| Frogger Game Server | Inference gameplay | JSON over WebSocket | WebSocket | Connection retry, graceful exit on disconnect |
| server/logic.py | Offline training environment | Python objects | Direct import | Validate game state consistency |
| matplotlib | Plot generation | PNG/SVG files | File I/O | Handle missing data gracefully |
| PyTorch | Neural network training | Tensors | In-memory | CUDA fallback to CPU |
| rich | Live console display, progress bars, tables | Text | In-memory | Graceful fallback to plain text |

### 6.3 Error Handling
- **Import Errors**: If `server.logic` cannot be imported, raise clear error indicating training must run from repo root
- **Model Load Errors**: If checkpoint file is missing or corrupted, raise `FileNotFoundError` or `RuntimeError` with descriptive message
- **WebSocket Errors**: During inference, connection errors are logged and agent exits gracefully (inherited from `BaseAgent`)
- **CUDA Errors**: If CUDA is unavailable, automatically fall back to CPU with warning log
- **NaN/Inf in Training**: Detect and abort training if loss becomes NaN/Inf, logging last known good state

---

## 7. Testing Strategy

- **Unit Tests** (target: >=85% coverage):
  - **Environment Wrapper**: Test `reset()`, `step()`, state extraction, reward computation, done condition
  - **State Encoder**: Test fixed output shape, normalization, edge cases (no obstacles, max obstacles)
  - **Replay Buffer**: Test capacity limits, sampling distribution, push/pop behavior
  - **DQN Network**: Test forward pass shape, parameter count, CPU/CUDA compatibility
  - **Reward Function**: Test reward values for: forward move, backward move, death, checkpoint, lap completion
  - **Epsilon Greedy**: Test decay schedule, boundary values (epsilon_start, epsilon_end), action randomness
  - **Target Network**: Test hard/soft update correctness

- **Integration Tests**:
  - **Training Loop**: Run 10 episodes and verify logs are created, model checkpoint is saved
  - **End-to-End Inference**: Load trained model (or random weights) into `DQNAgent`, verify it connects and sends valid actions
  - **Checkpoint Save/Load**: Save checkpoint, load into new trainer, verify training can resume

- **Evaluation Tests**:
  - Run evaluation script and verify output JSON contains all required statistics
  - Verify evaluation is deterministic when seed is fixed

- **Regression Tests**:
  - Ensure `server/logic.py` is not modified (checksum validation or git diff check)

---

## 8. Risks & Constraints

### 8.1 Risks
| Risk | Impact | Mitigation |
|------|--------|------------|
| Hyperparameter sensitivity | High | Run multiple experiments with different learning rates, epsilon decay schedules; document all results |
| Overfitting to specific obstacle patterns | Medium | Evaluate on multiple random seeds; ensure obstacle generation is stochastic during training |
| Training instability (diverging loss) | High | Use Huber loss, gradient clipping, target network, smaller learning rate; monitor for NaN/Inf |
| Insufficient compute for timely convergence | Medium | Start with smaller network; use GPU if available; reduce replay buffer size if memory-constrained |
| Poor state representation leading to poor performance | High | Experiment with multiple representations (vector vs. image); document design rationale |
| Deadline pressure (22/06/2026) | High | Prioritize MVP: simple state representation, small network, basic reward function; iterate if time permits |
| Game logic changes in upstream repo | Low | Pin fork state; do not pull upstream changes during project; verify `server/` is untouched |

### 8.2 Constraints
- **Deadline**: Project submission by 2026-06-22 (e-learning platform)
- **Game Immutability**: `server/logic.py`, `server/server.py`, and `agents/base_agent.py` must NOT be modified
- **Algorithm**: DQN is the mandated RL approach (not PPO, A3C, etc.)
- **Training Mode**: Offline training via direct Python import of `server/logic.py` (not via WebSocket)
- **Inference Mode**: Must use WebSocket via `BaseAgent` subclass
- **Python Version**: 3.10+
- **Dependencies**: PyTorch, numpy, matplotlib, rich (plus existing `ai-game-framework`, `websockets`)
- **Hardware**: Must run on standard laptop (CPU training acceptable; GPU optional but preferred)

---

## 9. Appendices

### Glossary
- **DQN**: Deep Q-Network — neural network approximating the Q-function in RL
- **Experience Replay**: Storage and uniform sampling of past transitions to break correlation
- **Target Network**: Separate network used to compute stable Q-learning targets
- **Epsilon-Greedy**: Exploration strategy selecting random actions with probability epsilon
- **TD Target**: Temporal Difference target = `reward + gamma * max(Q_target(next_state))`
- **Huber Loss**: Robust loss function less sensitive to outliers than MSE
- **State Representation**: Encoded form of game state fed into the neural network
- **Reward Shaping**: Designing reward function to guide agent toward desired behavior
- **Checkpoint**: Saved model weights and training state for resumption or inference

### Related Documents
- [project02.md](project02.md) - Course project specification and grading criteria
- `server/logic.py` - Frogger game logic (read-only external dependency)
- `agents/base_agent.py` - Base agent class for WebSocket inference (read-only external dependency)
- `README.md` - Project report (to be written as part of deliverables)
