# DQN Policy Collapse at y=3: Diagnosis and Recommended Fixes

**Date:** 2026-06-24  
**Agent:** DQN on custom Frogger environment  
**Symptoms:**

| Metric | Training (ε = 0.01) | Evaluation (ε = 0) |
|--------|--------------------|--------------------|
| Mean Return | **695.8** | **95.5** |
| AvgLaps | **3.1** | **0.0** |
| Std | ≥ 0 (normal variance) | **0.0** (identical episodes) |
| Failure Mode | Survives >3 laps | **Always dies at y = 3** |

The evaluation standard deviation of **0.0** proves that the greedy policy is completely deterministic and reaches the exact same catastrophic state on every single episode. This document explains *why* the greedy policy collapses to dying at `y = 3`, and provides concrete code fixes.

---

## 1. Root Cause Analysis

The collapse is not caused by a single bug, but by a **fatal interaction of four design choices**:

1. **Death is "cheap" because the episode does not terminate on death.**
2. **The death penalty (-10) is dwarfed by forward-progress rewards (+11 per lane) and bootstrapped future value.**
3. **The state representation lacks an explicit "safe to move north" signal, forcing the network to infer collision timing from aggregated lane statistics.**
4. **ε-greedy exploration (1 % random actions) occasionally generates "lucky" trajectories that survive the dangerous `y = 3` transition. DQN over-generalises the high return of these rare successes to the greedy action.**

### 1.1 The Non-Terminal Death Trap (Most Critical)

In `env/frogger_env.py` line 240 the episode-termination flag is set as:

```python
self._done = self.game.game_over or self.game.win
```

`game_over` is only `True` when **all** lives are exhausted (`lives <= 0`). When the frog dies with lives remaining, `done = False`. This has a devastating effect on Q-learning.

In DQN the TD target for a transition is:

```
target = reward + γ * max_a Q_target(s', a) * (1 - done)
```

Because `done = False` on death, the target becomes:

```
target_death = -10 + γ * V(respawn_state)
```

A completed lap is worth roughly **+226** (forward progress + checkpoint + lap bonus). Even a conservative estimate of the respawn-state value `V(respawn)` is **~150–200** because the agent occasionally reaches the goal. With `γ = 0.99`:

```
target_death ≈ -10 + 0.99 * 175  =  +163
```

**The network learns that dying is only a minor inconvenience.** The effective penalty is not -10; it is the * opportunity cost* of the single step, which is easily masked by optimistic bootstrapping.

> **If death were terminal (`done = True`), the target would simply be `-10`, and the Q-value of a suicidal action would collapse immediately.**

### 1.2 Reward Imbalance

The default reward shaping in `FroggerEnv.__init__` (line 61) is:

| Event | Reward |
|-------|--------|
| Forward (new lane) | **+11** (+10 bonus + 1 progress) |
| Checkpoint (y = 4) | **+50** |
| Lap (y = 8) | **+100** |
| Death | **-10** |
| Time/step | -0.1 |
| Stay | -0.5 |

Because bootstrapping makes the *effective* death cost negligible (section 1.1), the agent perceives a **positive expected value** for moving forward even when the move is fatal. The +11 immediate reward for entering a new lane outweighs the tiny perceived penalty.

### 1.3 State Representation Blind Spots

The 30-D state vector in `models/dqn_network.py` encodes:

- **Per-lane features (indices 4–21):** `min_dist / width`, `speed / max_speed`, `width / width`.
- **Directional sensors (indices 22–29):** Ray-cast distance in 8 directions, but **only for cells *ahead* of the frog** (`check_y = frog_y + dy * dist`).

Three critical pieces of information are missing:

1. **Current-cell occupancy.** There is no sensor or feature that says "a car is currently overlapping the frog". The directional sensors look at `y+1`, `y-1`, etc., but not at `y`.
2. **Safe-to-enter-north flag.** The network cannot directly read "the cell at (`frog_x`, `frog_y + 1`) is free". It must infer this from the distance to the nearest car in lane `y+1`, but the distance is a *minimum circular distance* (`_distance_to_obstacle`, line 192-226) that loses directional information (left vs. right). A car 1 unit away could be approaching from the left or receding to the right.
3. **Time-to-collision (TTC).** The raw distance is meaningless without knowing the relative speed and direction. A car 2 units away moving toward the frog is dangerous; a car 2 units away moving away is safe. The speed sign is present, but the network must learn to combine speed, distance, and wrap-around geometry—a difficult spatial-reasoning task for a 2-layer MLP.

### 1.4 Exploration Camouflage: Why Training Scores Are Deceptive

The environment is **fully deterministic** (obstacle initial positions are fixed at `i * spacing`, and physics are fixed-dt). The only source of stochasticity during training is ε-greedy:

- 99 % of steps: greedy action.
- 1 % of steps: uniform random action over 5 choices (0.2 % chance for each direction).

A random **EAST** or **WEST** at `y = 0` shifts the frog's `x` coordinate. For the default start `x = 5.0`, moving **NORTH** through all southbound lanes hits a fatal car at `y = 3`. However, moving **WEST** to `x = 4.0` places the frog in a safe gap for the rest of the lap. A single random lateral move at any point before `y = 3` can change a death episode (+35 return) into a 3-lap episode (+695 return).

Because these lucky episodes are **extremely high reward**, they dominate the Q-value updates for the state `(y = 2, x = 5.0, action = NORTH)` even though they represent only ~3 % of transitions. The network learns:

```
Q(y=2, x=5.0, NORTH)  ≈  0.97 * (death_target)  +  0.03 * (survival_target)
                        ≈  0.97 * (+163)         +  0.03 * (+210)
                        ≈  +164
```

Meanwhile the Q-value for **STAY** is:

```
Q(y=2, x=5.0, STAY)   ≈  -0.5 + γ * V(y=2)
                        ≈  -0.5 + 0.99 * 164
                        ≈  +162
```

**Greedy policy therefore chooses NORTH** (164 > 162), condemning the frog to death. During training the 1 % random noise occasionally overrides this decision and saves the episode, creating the illusion of a competent policy. During evaluation (ε = 0) the mask disappears and the true greedy policy reveals itself: a deterministic death march to `y = 3`.

> This is the definition of an **exploration-dependent policy**: the agent has learned to rely on the exploration noise to survive, rather than learning a robust, self-sufficient strategy.

---

## 2. Why the Failure Is Specifically at y = 3

The game layout creates an adversarial phasing at `y = 3`:

- **Lane 1** (`small_fast`, speed 1.8): Cars are spaced 11/3 ≈ 3.67. The frog passes through safely at the default timing.
- **Lane 2** (`large_slow`, speed 0.6, width 2.5): Cars are spaced 11/2 = 5.5. The frog also passes through safely at `x = 5.0`.
- **Lane 3** (`small_slow`, speed 0.8, width 1.0): Cars are spaced 11/4 = 2.75.

For the start position `x = 5.0`, after 3 northward moves the frog arrives at `y = 3` at `t = 0.5 s`. The car that started at `x = 5.5` has moved to `x ≈ 5.1`, overlapping the frog body `[5.1, 5.9]`. This is a **guaranteed collision** in the deterministic evaluation.

The agent never learns to wait for the gap because:
1. Waiting ~8 steps (1.25 s) is astronomically unlikely under ε-greedy.
2. The Q-value of waiting is never higher than the over-inflated Q-value of moving forward (section 1.4).
3. The state vector gives no explicit feature saying "wait 1.25 s".

Thus `y = 3` becomes a deterministic wall for the greedy policy.

---

## 3. Recommended Fixes

The fixes are ordered by **impact** (highest first). All code references are to the current codebase.

### Fix 1: Make Death Terminal (or Drastically Increase the Penalty)

**File:** `env/frogger_env.py`

**Why:** Solves the bootstrapping problem described in §1.1. When `done=True` on death, the TD target is exactly `reward_death`. The network can no longer hide the cost of dying behind optimistic future value.

**Change:** Treat every life-loss as an episode end for the RL agent.

```python
# env/frogger_env.py  (around line 240)
# OLD:
# self._done = self.game.game_over or self.game.win

# NEW:
self._done = (
    self.game.game_over
    or self.game.win
    or (self.game.lives < prev_lives)   # <-- terminate on ANY death
)
```

**Trade-off:** This caps laps-per-episode at one per life (max 3). This is *desirable* for RL—it forces the agent to learn survival. If you need the agent to continue after death, use Fix 1b instead.

**Fix 1b (Alternative):** Keep the multi-life episode but increase the penalty to offset bootstrapping:

```python
# env/frogger_env.py  (around line 61)
# OLD: reward_death: float = -10.0,
# NEW:
reward_death: float = -100.0,
```

Even with `done=False`, a penalty of `-100` ensures `target_death = -100 + γ*V(respawn) < 0`, making death genuinely costly. **Recommended:** apply both Fix 1 and Fix 1b for maximum stability.

---

### Fix 2: Add Explicit Safety Features to the State Vector

**File:** `models/dqn_network.py`

**Why:** Removes the need for the MLP to perform implicit collision geometry. The agent gets a direct negative signal when it is in danger and a direct positive signal when moving north is safe.

**Changes:**

1. Increase state dimensionality:

```python
# models/dqn_network.py  (line 17)
# OLD: STATE_DIM = 30
STATE_DIM = 32
```

2. Add two helper methods to `StateEncoder`:

```python
def _current_lane_danger(
    self, frog_x: float, frog_y: int, obstacles: List[Dict[str, Any]]
) -> float:
    """Return 1.0 if a car overlaps the frog's current cell, else 0.0."""
    if frog_y not in TRAFFIC_LANES:
        return 0.0
    frog_left = frog_x + 0.1
    frog_right = frog_x + 0.9
    for obs in obstacles:
        if obs.get("y") != frog_y:
            continue
        left = float(obs["x"])
        right = left + float(obs["width"])
        if left < frog_right and right > frog_left:
            return 1.0
        if right > self.width:
            if 0.0 < frog_right and (right - self.width) > frog_left:
                return 1.0
        if left < 0:
            if (self.width + left) < frog_right and self.width > frog_left:
                return 1.0
    return 0.0

def _next_lane_safe(
    self, frog_x: float, frog_y: int, obstacles: List[Dict[str, Any]]
) -> float:
    """Return 1.0 if the cell directly north is free of cars, else 0.0."""
    next_y = frog_y + 1
    if next_y not in TRAFFIC_LANES:
        return 1.0  # checkpoint / goal are always safe
    frog_left = frog_x + 0.1
    frog_right = frog_x + 0.9
    for obs in obstacles:
        if obs.get("y") != next_y:
            continue
        left = float(obs["x"])
        right = left + float(obs["width"])
        if left < frog_right and right > frog_left:
            return 0.0
        if right > self.width:
            if 0.0 < frog_right and (right - self.width) > frog_left:
                return 0.0
        if left < 0:
            if (self.width + left) < frog_right and self.width > frog_left:
                return 0.0
    return 1.0
```

3. Populate the new features in `encode`:

```python
# models/dqn_network.py  (inside encode(), after the directional sensors block)
features[30] = self._current_lane_danger(frog_x, frog_y, obstacles)
features[31] = self._next_lane_safe(frog_x, frog_y, obstacles)
```

**Expected impact:** The Q-network can now learn a simple rule: *if feature 31 == 0, do not choose NORTH*. This breaks the fatal loop at `y = 3` within a few thousand updates.

---

### Fix 3: De-Randomise the Exploration (Noisy Nets)

**File:** `training/dqn_trainer.py`

**Why:** ε-greedy is the root cause of the "exploration camouflage". Uniform random noise occasionally saves the agent, allowing the Q-network to misattribute the success to the greedy action. Noisy Nets provide **state-dependent** exploration: the network learns *when* to explore, rather than relying on blind luck.

**High-level implementation:** Replace `select_action` with a NoisyNet layer.

```python
# training/dqn_trainer.py
# In DQNNetwork.__init__, replace the second layer with a NoisyLinear:
# (NoisyLinear is available in stable-baselines3 or can be implemented in ~20 lines)

# Example conceptual change in select_action:
def select_action(self, state: torch.Tensor, epsilon: float = 0.0) -> int:
    # epsilon is ignored; exploration comes from network noise
    with torch.no_grad():
        q_values = self.policy_net(state.unsqueeze(0).to(self.device))
        return int(q_values.argmax(dim=1).item())
```

If implementing Noisy Nets is out of scope for the project, a cheaper alternative is **Parameter Space Noise**: add adaptive Gaussian noise to the policy-network weights before each episode, scaling the noise so that the greedy action changes by a target percentage. This avoids the uniform random actions that break deterministic legs of the trajectory.

---

### Fix 4: Prioritise Death Transitions in Replay

**File:** `training/replay_buffer.py` (if upgrading) or `training/dqn_trainer.py`

**Why:** Death transitions are rare but carry the most information. In standard uniform sampling, the network sees hundreds of "driving forward" transitions for every death, so it never accurately learns the boundary between safe and unsafe.

**Implementation sketch:**

- Assign a priority `p = |δ| + ε` to each transition, where `δ` is the TD error.
- Sample minibatches proportional to `p^α`.
- Use importance-sampling weights to correct the bias.

This is the standard **Prioritized Experience Replay (PER)** algorithm (Schaul et al., 2016).

---

### Fix 5: Randomise Obstacle Initial Positions (Domain Randomisation)

**File:** `server/logic.py`

**Why:** The deterministic initial layout causes the agent to overfit to a single fatal car phasing. Randomising start positions forces the agent to learn a policy that works for *any* car configuration, preventing the `std = 0.0` collapse.

**Change:**

```python
# server/logic.py  (inside _add_lane)
def _add_lane(self, y, width, speed_mag, variant, direction, count,
              random_offset: bool = True):
    max_count = int(self.width / (width + 1.5))
    actual_count = min(count, max_count)
    if actual_count < 1:
        actual_count = 1

    spacing = self.width / actual_count
    speed = speed_mag * direction
    offset = random.uniform(0.0, spacing) if random_offset else 0.0
    for i in range(actual_count):
        x = (i * spacing + offset) % self.width
        self.obstacles.append(Obstacle(x, y, width, speed, "car", variant))
```

Then call `_init_obstacles` inside `reset_game()` so every episode has a different layout.

---

### Fix 6: Dueling Network Architecture

**File:** `models/dqn_network.py`

**Why:** The current MLP learns `Q(s,a)` directly. A dueling architecture learns `V(s)` and `A(s,a)` separately. This helps the network recognise that `y = 3` with a car approaching is a **low-value state regardless of action**, which pushes the Q-value of NORTH down more aggressively.

**Implementation sketch:**

```python
class DuelingDQNNetwork(nn.Module):
    def __init__(self, input_dim=STATE_DIM, hidden_dim=HIDDEN_SIZE):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, NUM_ACTIONS)
        )

    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Agregate: Q = V + (A - mean(A))
        return value + (advantage - advantage.mean(dim=1, keepdim=True))
```

---

## 4. Summary Table

| Rank | Fix | Target File(s) | Solves | Effort |
|------|-----|----------------|--------|--------|
| 1 | **Make death terminal** | `env/frogger_env.py` | §1.1 Bootstrapping illusion | 1 line |
| 2 | **Raise `reward_death`** | `env/frogger_env.py` | §1.2 Reward imbalance | 1 line |
| 3 | **Add `danger` / `safe_north` features** | `models/dqn_network.py` | §1.3 Missing spatial info | ~40 lines |
| 4 | **Randomise obstacle starts** | `server/logic.py` | Overfitting to fixed seed | ~5 lines |
| 5 | **Noisy Nets / PER** | `training/` | §1.4 Exploration camouflage | Medium |
| 6 | **Dueling DQN** | `models/dqn_network.py` | Stability / value estimation | ~20 lines |

---

## 5. Expected Impact

Applying **Fixes 1 + 2 + 3** together should produce the following behaviour:

1. **`reward_death = -100` & terminal death**: The TD target for moving into the car at `y = 3` becomes approximately `-100` (instead of `+163`). The Q-value of **NORTH** collapses below the Q-value of **STAY** or **WEST**.
2. **Safety features**: The network immediately sees that `feature[31] == 0` at the critical `y = 2` state, suppressing the NORTH advantage.
3. **Result**: The greedy policy learns to wait at `y = 2` or shift laterally (`x = 4.0`) to avoid the fatal car, then proceeds through the checkpoint and northbound lanes.
4. **Evaluation score**: Should rise from **95.5** to **≥ 600**, matching the training performance. `Std` may remain low (a stable, deterministic optimal policy is expected), but now it reflects *successful* lap completion rather than deterministic death.

---

## 6. How to Verify the Diagnosis

You can test the bootstrapping-illusion hypothesis without changing code:

1. Temporarily set `reward_death = -500.0` in `FroggerEnv` and run 200 training episodes.
2. Observe whether the greedy policy now learns to **STAY** at `y = 2` instead of rushing into `y = 3`.
3. If evaluation mean jumps from ~95 to >400, the diagnosis is confirmed: the original penalty was simply too small to overcome optimistic value bootstrapping.

If you need the agent to maintain multi-life episodes (no terminal death), you *must* keep `reward_death` at least as large as the expected return of a full lap (~200–300) to prevent the Q-network from treating respawns as valuable resets.
