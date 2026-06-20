# SI2-Frogger DQN Agent Functional Specification

> **Version**: 2.1.0 | **Date**: 2026-06-20 | **Author**: Documenter Agent | **Status**: Draft

## Change Log
| Version | Date | Author | Changes |
|---------|------|--------|---------|
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
- **State Representation**: Engineered feature vector or image-like representation capturing frog position, obstacle positions/speeds, and lane danger levels
- **DQN Network**: PyTorch neural network approximating Q-values for 4 discrete actions (NORTH, SOUTH, EAST, WEST)
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
| FR-001 | Environment Wrapper | The system shall provide a Gym-like wrapper around Frogger logic exposing reset(), step(), state extraction, reward computation, and cooldown handling | Must | project02.md, codebase | None | Draft |
| FR-002 | DQN Training System | The system shall implement a complete DQN algorithm including neural network, experience replay, epsilon-greedy exploration, target network updates, loss computation, and optimizer | Must | project02.md, DQN paper | FR-001 | Draft |
| FR-003 | Training Orchestration | The system shall run a training loop over N episodes with logging, checkpointing, and visualization | Must | project02.md | FR-002 | Draft |
| FR-004 | Inference Agent | The system shall provide a DQNAgent subclassing BaseAgent that loads trained weights and plays via WebSocket | Must | project02.md, codebase | FR-003 | Draft |
| FR-005 | Evaluation and Benchmarking | The system shall evaluate trained agents over multiple episodes and compute statistics vs DummyAgent baseline | Must | project02.md | FR-004 | Draft |
| FR-006 | Hyperparameter Configuration | The system shall expose all hyperparameters via config file or CLI | Should | project02.md | FR-003 | Draft |

### Acceptance Criteria

#### FR-001: Environment Wrapper
**Description**: The system shall provide a Gym-like wrapper around `server/logic.Frogger` exposing `reset()`, `step()`, state extraction, reward computation, and cooldown handling.
**Priority**: Must
**Source**: project02.md, codebase
**Dependencies**: None
**Acceptance Criteria**:
- [ ] `FroggerEnv` class wraps `server.logic.Frogger` without modifying `logic.py`
- [ ] `reset()` returns initial state and resets the underlying `Frogger` game
- [ ] `step(action)` accepts action string ("NORTH", "SOUTH", "EAST", "WEST"), advances game by one or more ticks, and returns `(next_state, reward, done, info)`
- [ ] `step()` handles the 5-frame movement cooldown by either stepping multiple ticks or waiting appropriately
- [ ] Environment tracks episode length and exposes it in `info`
- [ ] Environment is deterministic when seeded
**Status**: Draft

#### FR-002: DQN Training System
**Description**: The system shall implement a complete DQN algorithm including neural network architecture (PyTorch), experience replay buffer, epsilon-greedy exploration with decay, target network updates, loss computation (Huber/MSE), and Adam optimizer.
**Priority**: Must
**Source**: project02.md, DQN paper (Mnih et al., 2015)
**Dependencies**: FR-001
**Acceptance Criteria**:
- [ ] Network maps states to 4 Q-values (NORTH, SOUTH, EAST, WEST)
- [ ] Network supports both CPU and CUDA execution
- [ ] Forward pass completes in <10ms for a single state
- [ ] Replay buffer stores tuples of `(state, action, reward, next_state, done)`
- [ ] Replay buffer has configurable maximum capacity (default: 10,000 - 100,000)
- [ ] `push()` adds transitions; when full, oldest transitions are overwritten
- [ ] `sample(batch_size)` returns random minibatch without replacement using vectorized operations
- [ ] Epsilon starts at `epsilon_start` (default: 1.0) and decays to `epsilon_end` (default: 0.01) over configurable schedule
- [ ] With probability epsilon, a random valid action is selected; with probability (1 - epsilon), the action with highest Q-value is selected
- [ ] Target network is a copy of the policy network, initialized identically
- [ ] Hard update: target network copies policy network weights every N steps (default: every 1,000 steps); OR soft update via Polyak averaging (`tau * policy + (1-tau) * target`)
- [ ] Target network is used for computing Q-targets in loss function, not for action selection
- [ ] Loss is computed as Huber or MSE between predicted Q and target Q
- [ ] Optimizer is Adam with configurable learning rate
- [ ] Training updates occur every `update_frequency` steps (default: every 4 steps)
**Status**: Draft

#### FR-003: Training Orchestration
**Description**: The system shall run a training loop over N episodes with logging, checkpointing, and visualization.
**Priority**: Must
**Source**: project02.md
**Dependencies**: FR-002
**Acceptance Criteria**:
- [ ] Training script runs for configurable number of episodes (default: 1,000+)
- [ ] Each episode: reset env, run until done, accumulate transitions, perform training updates
- [ ] Logs include: episode number, total reward, epsilon, loss, episode length, high score
- [ ] Logs are written to console and optionally to file (CSV or JSON)
- [ ] Model weights saved every N episodes (configurable, default: every 100)
- [ ] Best model (highest evaluation score) is saved separately and never overwritten
- [ ] Checkpoint includes: policy network state dict, target network state dict, optimizer state dict, episode count, epsilon value, best score
- [ ] Checkpoints are saved to `checkpoints/` directory with descriptive filenames
- [ ] Checkpoint loading restores full training state for resumption
- [ ] Plot 1: Episode rewards over time (smoothed with moving average)
- [ ] Plot 2: Loss curve over training steps
- [ ] Plot 3: Epsilon decay over episodes
- [ ] Plot 4: Evaluation score distribution (histogram or boxplot)
- [ ] Plots are saved as PNG/SVG to `plots/` or `results/` directory
- [ ] Plot generation script is separate and can be run post-training
**Status**: Draft

#### FR-004: Inference Agent
**Description**: The system shall provide a `DQNAgent` subclassing `BaseAgent` that loads trained weights and plays via WebSocket.
**Priority**: Must
**Source**: project02.md, codebase
**Dependencies**: FR-003
**Acceptance Criteria**:
- [ ] `DQNAgent` extends `agents.base_agent.BaseAgent`
- [ ] Implements `async deliberate()` method selecting action via trained DQN
- [ ] Converts WebSocket state JSON to the same representation used during training
- [ ] Selects action with epsilon=0 (pure exploitation) during gameplay
- [ ] Handles `game_over` state by returning `None`
- [ ] Successfully connects to `ws://localhost:8765/ws` and plays the game
- [ ] Can be launched via `python3 -m agents.dqn_agent`
**Status**: Draft

#### FR-005: Evaluation and Benchmarking
**Description**: The system shall evaluate trained agents over multiple episodes and compute statistics vs DummyAgent baseline.
**Priority**: Must
**Source**: project02.md
**Dependencies**: FR-004
**Acceptance Criteria**:
- [ ] Evaluation runs for configurable number of episodes (default: 100) with epsilon=0
- [ ] Computes: mean score, max score, min score, standard deviation
- [ ] Computes: mean episode length, mean survival time, mean laps completed
- [ ] Compares against DummyAgent baseline over same number of episodes
- [ ] Results are saved to JSON/CSV and printed to console
- [ ] Evaluation script is separate from training script
**Status**: Draft

#### FR-006: Hyperparameter Configuration
**Description**: The system shall expose all hyperparameters via config file or CLI.
**Priority**: Should
**Source**: project02.md
**Dependencies**: FR-003
**Acceptance Criteria**:
- [ ] Configurable parameters: learning rate, gamma (discount factor), epsilon_start, epsilon_end, epsilon_decay, buffer_size, batch_size, target_update_freq, hidden_layer_sizes
- [ ] Configuration loaded from JSON/YAML file or CLI arguments
- [ ] Default values are sensible and documented
- [ ] Multiple hyperparameter experiments can be run and compared
- [ ] Hyperparameter values used for final model are documented in README.md
**Status**: Draft

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
| DQNNetwork | Neural Network | PyTorch `nn.Module` mapping state tensors to Q-values for 4 actions |
| ReplayBuffer | Experience Storage | Circular buffer storing transitions; supports uniform random sampling |
| DQNTrainer | Training Orchestrator | Runs training loop: episode generation, epsilon decay, loss computation, backpropagation, checkpointing |
| TargetNetwork | Stable Q-Target | Copy of DQNNetwork updated periodically; used for computing TD targets |
| DQNAgent | Inference Agent | Subclasses `BaseAgent`; loads trained weights; selects greedy actions via WebSocket |
| Evaluator | Performance Testing | Runs evaluation episodes, computes statistics, generates comparison tables |
| Plotter | Visualization | Generates matplotlib plots: reward curves, loss curves, epsilon decay, score distributions |
| ConfigManager | Hyperparameter Management | Loads hyperparameters from JSON/YAML/CLI; provides defaults |

### 5.2 Design Decisions

The following design decisions are documented in the Architecture section (not as standalone functional requirements) because they are implementation choices that support the User Goal-level requirements:

#### State Representation Design
The StateEncoder component converts raw game state into a fixed-size representation suitable for neural network input. Key design choices:
- **Fixed dimensions**: Representation must have fixed dimensions regardless of obstacle count (e.g., via grid occupancy, relative positions, or engineered features)
- **Frog position**: Included as normalized or absolute coordinates
- **Obstacle information**: Positions, speeds, and directions for relevant lanes
- **Performance**: State can be converted to PyTorch tensor in <1ms
- **Documentation**: Representation dimensionality and encoding rationale must be documented in README.md
- **Alternatives**: At least one alternative representation must be considered and documented (e.g., image-like vs. vector)

#### Reward Function Design
The reward function is implemented in the environment wrapper (not in game logic) to shape agent behavior. Key design choices:
- **Positive rewards**: Forward progress (e.g., +1 per new lane), reaching checkpoints (+10/+20), completing laps
- **Negative rewards**: Death/collision (e.g., -10), idle time (optional), moving backward (optional)
- **Sparsity vs. density**: Design choice must be justified in README.md
- **Clipping**: Reward clipping (e.g., [-1, 1]) must be considered and documented
- **Documentation**: Reward function must be explicitly defined and documented in README.md

#### Network Architecture Details
The DQNNetwork component implements the neural network approximating the Q-function. Key design choices:
- **Input**: Accepts state tensor from StateEncoder
- **Output**: Q-values for [NORTH, SOUTH, EAST, WEST]
- **Architecture**: Layer sizes and activation functions must be documented
- **Parameter count**: Number of parameters must be documented in README.md
- **Justification**: Architecture choice must be justified (e.g., MLP for vector states, CNN for image states)
- **Device support**: Supports both CPU and CUDA execution

### 5.3 Data Models

#### Model: Transition
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| state | Tensor | Yes | Encoded state representation (shape depends on encoder) |
| action | int | Yes | Action index: 0=NORTH, 1=SOUTH, 2=EAST, 3=WEST |
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

### 5.4 API Design

#### Gym-like Environment API
```python
# FroggerEnv
state = env.reset()                          # Returns initial encoded state
next_state, reward, done, info = env.step(action)  # Returns transition tuple
state_shape = env.observation_space          # Fixed shape of state vector
n_actions = env.action_space                 # 4 (NORTH, SOUTH, EAST, WEST)
```

#### DQN Network API
```python
# DQNNetwork (PyTorch)
q_values = model(state_tensor)               # Forward pass: returns [batch, 4] Q-values
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
  - `python3 -m training.train --config config.json`
  - `python3 -m evaluation.evaluate --model checkpoints/best.pt --episodes 100`
  - `python3 -m visualization.plot --log logs/training.csv`
  - `python3 -m agents.dqn_agent --model checkpoints/best.pt`
- **Web Viewer**: Existing HTML5 Canvas viewer at `http://localhost:8765/` used to observe trained agent gameplay

### 6.2 External Integrations
| System | Purpose | Data Format | Protocol | Error Handling |
|--------|---------|-------------|----------|----------------|
| Frogger Game Server | Inference gameplay | JSON over WebSocket | WebSocket | Connection retry, graceful exit on disconnect |
| server/logic.py | Offline training environment | Python objects | Direct import | Validate game state consistency |
| matplotlib | Plot generation | PNG/SVG files | File I/O | Handle missing data gracefully |
| PyTorch | Neural network training | Tensors | In-memory | CUDA fallback to CPU |

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
- **Dependencies**: PyTorch, numpy, matplotlib (plus existing `ai-game-framework`, `websockets`)
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
