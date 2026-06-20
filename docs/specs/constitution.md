# SI2-Frogger DQN Agent — Project Constitution

> **Version**: 1.0.0 | **Date**: 2026-06-20 | **Status**: Approved
>
> This document defines the fundamental principles, values, standards, and constraints governing the SI2-Frogger DQN Agent project. It serves as the "north star" for all development decisions. All contributors SHALL adhere to the mandatory rules ("SHALL") and SHOULD follow the recommendations ("SHOULD").

---

## 1. Project Vision

**What we are building**: An autonomous Deep Q-Network (DQN) agent that learns to play the Frogger game via Reinforcement Learning (RL), achieving significantly better-than-random performance through self-play and neural network-based value approximation.

**Why we are building it**: This is a university project for the SI2 course, demonstrating mastery of RL concepts (DQN, experience replay, target networks, epsilon-greedy exploration), software engineering best practices (modularity, testing, documentation), and scientific methodology (experimentation, benchmarking, reproducibility).

**Deadline**: 2026-06-22 (submission via e-learning platform).

**Success criteria**: The trained agent achieves >2x the mean score of the `DummyAgent` baseline over 100 evaluation episodes, with clear training convergence, comprehensive documentation, and >=85% test coverage.

---

## 2. Core Values

### 2.1 Game Server is Sacred
The existing game server, logic, and base agent framework are external dependencies provided as a stable platform. They SHALL NOT be modified. Our work builds *on top of* these components, never *inside* them.

### 2.2 Training is Offline, Inference is Online
Training SHALL use direct Python import of `server.logic.Frogger` (offline, no WebSocket). Inference (gameplay) SHALL use WebSocket via a `BaseAgent` subclass (online). These two modes SHALL remain strictly separated.

### 2.3 Reproducibility Over Randomness
All experiments SHALL be seeded (Python, NumPy, PyTorch) to ensure reproducible results. Randomness is a tool for exploration, not an excuse for unexplainable outcomes.

### 2.4 Simplicity Over Complexity
We SHALL start with the simplest viable solution (simple state representation, small network, basic reward function) and iterate based on evidence. Complexity is earned, not assumed.

### 2.5 Evidence Over Intuition
All design decisions (state representation, reward function, network architecture, hyperparameters) SHALL be justified with training curves, evaluation statistics, or benchmark comparisons. "It felt right" is not sufficient.

### 2.6 Documentation is Code
The `README.md` SHALL serve as the project report. It SHALL be treated with the same rigor as code: accurate, up-to-date, and comprehensive. Inline documentation (docstrings, comments) SHALL explain *why*, not just *what*.

### 2.7 Checkpoints are Insurance
Model checkpoints SHALL be saved frequently and automatically. Training is expensive; losing progress to a crash or poor hyperparameter choice is unacceptable.

---

## 3. Architecture Principles

### 3.1 Separation of Concerns
The codebase SHALL be organized into distinct modules with clear responsibilities:
- `env/` — Environment wrapper and state representation
- `models/` — Neural network architectures
- `agents/` — Inference agents (subclassing `BaseAgent`)
- `training/` — Training loop, replay buffer, epsilon-greedy
- `evaluation/` — Benchmarking and statistics
- `utils/` — Shared utilities (config, logging, plotting)

### 3.2 PyTorch for Neural Networks
All neural network code SHALL use PyTorch. The network SHALL support both CPU and CUDA execution with automatic fallback.

### 3.3 Gym-like Environment API
The environment wrapper (`FroggerEnv`) SHALL expose a Gym-like API: `reset()`, `step(action)`, `observation_space`, `action_space`. This ensures compatibility with standard RL tooling and testing patterns.

### 3.4 BaseAgent Subclass for Inference
All inference agents SHALL subclass `agents.base_agent.BaseAgent` and implement the `async deliberate()` method. The agent SHALL load trained weights and select actions with epsilon=0 (pure exploitation).

### 3.5 Config-Driven Hyperparameters
All hyperparameters SHALL be externalized to a configuration file (JSON/YAML) or CLI arguments. No magic numbers SHALL exist in training or model code. Default values SHALL be sensible and documented.

### 3.6 Checkpointing for Reproducibility
Checkpoints SHALL include: policy network state, target network state, optimizer state, episode count, epsilon value, and best score. Checkpoints SHALL enable full training resumption from any saved state.

---

## 4. Code Quality Standards

### 4.1 Style and Formatting
- All Python code SHALL be PEP 8 compliant.
- Line length SHOULD not exceed 100 characters.
- Imports SHALL be ordered: standard library, third-party, local.

### 4.2 Type Safety
- All public functions and methods SHOULD use type hints.
- Complex data structures (e.g., transition tuples, config dicts) SHOULD use `TypedDict` or dataclasses.

### 4.3 Documentation
- Every module SHALL have a module-level docstring explaining its purpose.
- Every public class SHALL have a class docstring.
- Every public method SHALL have a docstring describing parameters, return values, and exceptions.
- Docstrings SHOULD follow the Google or NumPy style.

### 4.4 Testing
- Unit test coverage SHALL be >=85% for all agent and training code.
- Tests SHALL cover: environment wrapper, state encoder, replay buffer, DQN network, reward function, epsilon-greedy, target network updates.
- Integration tests SHALL cover: training loop (10 episodes), checkpoint save/load, end-to-end inference.
- The `server/` directory and `agents/base_agent.py` SHALL be excluded from coverage targets.

### 4.5 Code Smells
- Magic numbers SHALL NOT exist in source code; use named constants from config.
- Global state and singletons SHALL NOT be used.
- Functions SHOULD be small and single-purpose (max ~50 lines).
- Variable names SHALL be meaningful (e.g., `epsilon_decay` not `ed`).

---

## 5. RL-Specific Principles

### 5.1 State Representation
We SHALL start with the simplest viable state representation (e.g., engineered feature vector) and iterate based on evaluation evidence. Alternative representations SHALL be considered and documented.

### 5.2 Reward Function
Reward shaping is an experiment, not a given. The reward function SHALL be explicitly defined, documented, and justified. We SHALL evaluate the impact of reward design on training convergence.

### 5.3 Baseline Benchmarking
Every trained agent SHALL be benchmarked against the `DummyAgent` baseline over the same number of episodes. No agent is "good" without statistical evidence of superiority.

### 5.4 Checkpoint Discipline
Model weights SHALL be saved at regular intervals (default: every 100 episodes). The best-performing model (by evaluation score) SHALL be saved separately and never overwritten.

### 5.5 Training Stability
Training SHALL monitor for instability (NaN/Inf loss, diverging Q-values). If detected, training SHALL abort and the last known good checkpoint SHALL be preserved. Gradient clipping and Huber loss SHOULD be used.

### 5.6 Epsilon Decay
Epsilon decay is part of the experiment. The decay schedule (linear, exponential, step) SHALL be configurable, documented, and justified with training curves.

### 5.7 Target Network Updates
The target network SHALL be updated periodically (hard copy or soft Polyak averaging) and SHALL be used exclusively for computing Q-targets, not for action selection.

---

## 6. Decision-Making Framework

When faced with a technical decision, apply the following hierarchy:

1. **When in doubt, prefer simplicity.** A simple solution that works is better than a complex solution that might work.
2. **When comparing approaches, use evaluation statistics.** Run both, measure mean score, variance, and convergence speed. Numbers decide.
3. **When changing hyperparameters, document the before/after.** Every hyperparameter experiment SHALL be logged with: old value, new value, rationale, and resulting performance change.
4. **When adding features, ask: "Does this improve the grade?"** Features SHALL align with the grading criteria (Solution, Code, Repository, Complexity, Report, Contributions).
5. **When stuck, consult the functional spec.** The functional spec (`docs/specs/functional-spec.md`) is the single source of truth for requirements.
6. **When the deadline approaches, cut scope, not quality.** A complete, well-documented simple solution scores higher than an incomplete complex one.

---

## 7. Forbidden Practices (Hard Constraints)

The following practices are STRICTLY PROHIBITED. Violation SHALL be considered a critical project breach.

| # | Constraint | Rationale |
|---|-----------|-----------|
| F-01 | **NEVER modify `server/logic.py`, `server/server.py`, or `agents/base_agent.py`.** | These are external dependencies. Modifying them breaks reproducibility and upstream compatibility. |
| F-02 | **NEVER train via WebSocket.** | Training SHALL use direct import of `server.logic.Frogger`. WebSocket is for inference only. |
| F-03 | **NEVER commit model checkpoints or large binary files to git.** | Checkpoints SHALL be saved to `checkpoints/` (gitignored). Use `.gitignore` to exclude `.pt`, `.pth`, `.ckpt`, and `checkpoints/`. |
| F-04 | **NEVER use global state or singletons.** | All components SHALL be instantiated and passed explicitly. Global state makes testing and reproducibility impossible. |
| F-05 | **NEVER ignore test failures.** | A failing test SHALL block progress until resolved or explicitly documented as a known issue with justification. |
| F-06 | **NEVER use an RL algorithm other than DQN.** | The project mandates DQN (Q-Learning with neural networks). PPO, A3C, and other algorithms are out of scope. |
| F-07 | **NEVER hardcode hyperparameters in training/model code.** | All hyperparameters SHALL be loaded from config or CLI arguments. |
| F-08 | **NEVER submit without benchmarking against DummyAgent.** | Every submission-worthy model SHALL have statistical comparison vs. the random baseline. |

---

## 8. Grading Alignment

This constitution SHALL guide development to maximize alignment with the grading criteria defined in `project02.md`.

| Criterion | Weight | Constitutional Alignment |
|-----------|--------|-------------------------|
| **Solution** (Effectiveness) | 30% | Benchmark against `DummyAgent` (Core Value 2.5, RL Principle 5.3). Achieve convergence (RL Principle 5.4). Demonstrate stable training (RL Principle 5.5). |
| **Code** (Readability, Modularity) | 20% | PEP 8 compliance (4.1). Separation of concerns (3.1). Type hints and docstrings (4.2, 4.3). >=85% test coverage (4.4). |
| **Repository** (Organization) | 20% | Clear directory structure (3.1). No committed binaries (F-03). Config-driven design (3.5). Clean `.gitignore`. |
| **Complexity** (Model Design, Reward Creativity) | 15% | Document state representation experiments (5.1). Justify reward function design (5.2). Document network architecture (3.2). |
| **Report** (Training Curves, Evaluation) | 10% | `README.md` as report (2.6). Matplotlib plots for rewards, loss, epsilon (FR-003). Evaluation statistics table (FR-005). |
| **Contributions** (Bug Fixes, PRs) | 5% | Fix bugs in separate branches. Submit PRs to upstream for bonus credit. Never modify upstream files directly in main branch (F-01). |

**Grading Strategy**: To maximize the grade, prioritize Solution (30%) and Code (20%) first — these are the highest-weight criteria and are directly supported by the constitution. Repository organization (20%) is table stakes. Complexity (15%) and Report (10%) are differentiation factors. Contributions (5%) are a bonus.

---

## 9. Amendment Process

This constitution is a living document, but changes SHALL NOT be made lightly.

### 9.1 Proposal
Any contributor MAY propose an amendment by creating a new section in the project discussion or by opening a dedicated amendment document.

### 9.2 Requirements for Amendment
A proposed amendment SHALL include:
- The specific text to be added, modified, or removed.
- The rationale for the change.
- The impact on existing requirements, code, and grading alignment.

### 9.3 Approval
An amendment SHALL be approved ONLY by:
- Explicit written approval from all team members, OR
- Explicit approval from the project evaluator/professor.

### 9.4 Versioning
Approved amendments SHALL increment the constitution version number (semantic: MAJOR.MINOR.PATCH) and SHALL be recorded in a change log at the bottom of this document.

### 9.5 Emergency Overrides
In exceptional circumstances (e.g., critical bug blocking submission), a team member MAY temporarily override a constitutional rule with:
- A clear justification documented in code comments and commit messages.
- A plan to revert or formally amend the constitution post-submission.

---

## 10. Change Log

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-06-20 | Documenter Agent | Initial constitution: Project Vision, Core Values, Architecture Principles, Code Quality Standards, RL-Specific Principles, Decision-Making Framework, Forbidden Practices, Grading Alignment, Amendment Process. |

---

*End of Constitution. All team members SHALL acknowledge and adhere to this document.*
