# Training vs Evaluation Performance Discrepancy: A Critical Analysis

## Executive Summary

This document presents a detailed analysis of a significant performance discrepancy observed during the development of a Deep Q-Network (DQN) agent for the Frogger environment. While the agent appeared to achieve strong training performance—reaching a peak score of 1063.2 with 5 completed laps at episode 1559—subsequent evaluation of the saved model yielded catastrophically poor results (95.70 score, 0 laps). This report examines the root causes, presents empirical evidence, and offers recommendations for addressing this common but critical issue in reinforcement learning.

---

## 1. Observed Phenomenon

During the training phase, the DQN agent demonstrated what appeared to be successful learning. The training logs showed progressive improvement, with the agent eventually achieving multi-lap completion and high scores. However, when the trained model was evaluated in a deterministic setting (no exploration, fixed weights), the agent failed to complete even a single lap and achieved a score an order of magnitude lower than the training peak.

This discrepancy between training and evaluation performance is not merely a minor deviation—it represents a fundamental failure in the learning process that was masked by the training metrics.

---

## 2. Evidence

### 2.1 Training Performance Log

The following table summarises the key training milestones:

| Metric | Episode 1559 (Peak) | Episode 1800 | Episode 2000 (End) |
|--------|---------------------|--------------|--------------------|
| **Score** | 1063.2 | 518.5 | 29.2 |
| **Laps Completed** | 5 | — | 0 |
| **Steps per Lap** | 21.2 | — | — |
| **Agent State** | Successful | Degraded | Failed |

At episode 1559, the agent demonstrated robust performance: completing 5 laps with an efficient step count of 21.2 steps per lap. This represents genuine mastery of the environment.

By episode 2000, the agent was stuck at `y=3` (the starting row) and died immediately, achieving a score of only 29.2 with zero laps completed.

### 2.2 Evaluation Results

Evaluation was conducted using deterministic action selection (ε = 0) to assess the learned policy without exploration noise:

| Model Checkpoint | Evaluation Score | Standard Deviation | Laps Completed |
|------------------|------------------|--------------------|----------------|
| `best_model.pt` | 95.70 | 0.00 | 0 |
| `checkpoint_ep10000.pt` | 95.70 | 0.00 | 0 |

Both checkpoints produced identical, poor evaluation scores with zero variance, indicating the agent had converged to a deterministic but catastrophically suboptimal policy.

### 2.3 Checkpoint Metadata Analysis

The checkpoint metadata reveals a critical insight:

```
best_model.pt
├── Saved at episode: 1800
├── Best historical score: 518.5
├── Actual weights represent: Episode 1800 policy
└── Peak performance (ep 1559): NOT captured
```

The `best_model.pt` file was saved at episode 1800 with a score of 518.5—not at episode 1559 when the true peak of 1063.2 was achieved. By the time the checkpoint was saved, the agent had already begun to forget its previously learned successful strategy.

---

## 3. Root Cause Analysis: Catastrophic Forgetting in DQN

The primary cause of this failure is **catastrophic forgetting** combined with **checkpoint timing misalignment**.

### 3.1 The Learning Trajectory

1. **Episode ~1559**: The agent successfully learned an effective policy, achieving 5 laps and a score of 1063.2. The Q-network had converged to a representation that accurately valued state-action pairs leading to successful lane crossings.

2. **Episodes 1560–1800**: Continued training caused the network to overwrite its successful weights. The replay buffer, which increasingly contained transitions from suboptimal trajectories, polluted the learning signal. The agent began to "forget" the successful strategy it had discovered.

3. **Episode 1800**: The checkpoint was saved. While the metadata recorded a "best score" of 518.5, the actual network weights represented a degraded policy that was already significantly worse than the episode 1559 peak.

4. **Episodes 1801–2000**: The degradation accelerated. The agent became stuck at `y=3`, dying immediately. The Q-values for the successful actions had been overwritten by incorrect estimates derived from poor experiences.

5. **Episode 2000**: Training terminated. The final model was completely non-functional.

### 3.2 Why the Agent Got Stuck at y=3

The observation that the agent was stuck at `y=3` (the starting position) suggests the Q-network assigned higher values to staying put or moving horizontally rather than advancing forward. This can occur when:

- The reward signal for dying immediately became dominant in the replay buffer
- The target network propagated incorrect Q-value estimates
- The agent overfit to a local minimum where inaction appeared optimal

---

## 4. Why the "Best Score" is Misleading

A critical design flaw in the checkpointing mechanism contributed to this issue. The "best score" metric is a **historical maximum** that does not guarantee the saved weights correspond to that performance level.

```
Misleading Interpretation:
  "best_model.pt has best_score = 518.5"
  → Assumption: These weights achieve 518.5

Reality:
  best_score = 518.5 (historical maximum observed)
  saved_at_episode = 1800
  actual_weights_performance ≈ 95.70 (evaluation result)
```

The checkpoint saves the **current network weights** at the time the historical maximum is updated—not the weights from when the maximum was actually achieved. If performance degrades between the peak and the checkpoint save event, the saved model will reflect the degraded state.

In this case:
- The true peak (1063.2) at episode 1559 was never checkpointed
- The checkpoint at episode 1800 captured a model that had already forgotten the successful policy
- The historical maximum of 518.5 at episode 1800 was itself a degraded version of the true peak

---

## 5. Technical Explanation: Why This Happens in DQN

Several mechanisms inherent to DQN contribute to this instability:

### 5.1 Replay Buffer Pollution

DQN uses experience replay to break correlation between consecutive samples. However, as the agent's performance degrades, the replay buffer fills with poor-quality transitions (immediate deaths, zero rewards). When these transitions are sampled for training, they overwrite the network's knowledge of successful strategies.

```
Replay Buffer Composition Over Time:
Episode 1000:  [Good, Good, Good, Good, ...]  → Network improves
Episode 1559:  [Good, Good, Good, Good, ...]  → Peak performance
Episode 1800:  [Poor, Poor, Good, Poor, ...]  → Mixed, degradation begins
Episode 2000:  [Poor, Poor, Poor, Poor, ...]  → Network forgets
```

### 5.2 Target Network Instability

DQN uses a separate target network to compute Q-value targets. If the target network itself becomes unstable due to poor recent experiences, it propagates incorrect targets back to the primary network, creating a feedback loop of degradation.

### 5.3 Non-Stationary Targets

Unlike supervised learning where targets are fixed, DQN's targets depend on the network's own predictions. As the policy changes, the target Q-values shift. If the network enters a region of parameter space where targets consistently underestimate the value of good actions, it can spiral into a local minimum.

### 5.4 Overfitting to Local Minima

The agent may have overfit to a local minimum where staying at `y=3` appeared optimal. Without sufficient exploration (ε too low) or with a learning rate too high, the network can converge to this suboptimal policy and lose the ability to escape.

### 5.5 Epsilon-Greedy Exploration Masking Poor Learning

During training, epsilon-greedy exploration injects random actions. This exploration can artificially inflate training scores by allowing the agent to stumble into good states occasionally, even if the learned policy itself is poor. Evaluation with ε = 0 removes this crutch, revealing the true policy quality.

---

## 6. Lessons Learned

This failure provides several important lessons about reinforcement learning:

### 6.1 Training Score ≠ Evaluation Score

A high training score does not guarantee a good policy. Training metrics can be inflated by exploration, lucky trajectories, or temporary convergence. **Deterministic evaluation is the only reliable measure of policy quality.**

### 6.2 Need for Frequent Checkpoints

The default checkpointing frequency was insufficient. The true peak at episode 1559 was missed entirely. More frequent checkpoints (e.g., every 50 episodes) would have captured the successful model before forgetting occurred.

### 6.3 DQN Can Be Unstable

DQN is not guaranteed to converge monotonically. Performance can peak and then degrade catastrophically. Monitoring must include not just the current score but the stability of the learned policy.

### 6.4 Epsilon-Greedy Can Mask Poor Learning

Training with ε > 0 can produce scores that reflect exploration rather than exploitation. The agent's training score of 1063.2 may have included significant contributions from random lucky actions, while the learned policy itself was already degrading.

### 6.5 Historical Maximums Are Not Sufficient

Saving checkpoints based on historical maximum scores without ensuring the saved weights correspond to that performance is a critical error. The checkpointing mechanism must either:
- Save weights immediately when a new maximum is achieved, or
- Maintain a separate "best weights" buffer that is only updated when performance improves

---

## 7. Recommendations

### 7.1 Increase Checkpoint Frequency

Save checkpoints every 50 episodes (or even every 25) rather than relying on sparse saves. This increases the probability of capturing the true peak performance.

```python
# Recommended checkpointing strategy
checkpoint_frequency = 50  # Save every 50 episodes
save_best_separately = True  # Maintain best weights independently
```

### 7.2 Implement Early Stopping

Monitor evaluation performance and halt training when performance degrades for a sustained period (e.g., 100 episodes without improvement).

```python
# Early stopping pseudocode
patience = 100
best_eval_score = 0
episodes_without_improvement = 0

if eval_score > best_eval_score:
    best_eval_score = eval_score
    save_best_weights()
    episodes_without_improvement = 0
else:
    episodes_without_improvement += 1
    if episodes_without_improvement >= patience:
        stop_training()
```

### 7.3 Hyperparameter Tuning

- **Lower learning rate**: A smaller learning rate (e.g., 1e-4 instead of 1e-3) reduces the speed of catastrophic forgetting
- **More frequent target updates**: Update the target network every 500 steps instead of 1000 to reduce target instability
- **Larger replay buffer**: A larger buffer retains more diverse experiences, slowing the dominance of poor transitions

### 7.4 Algorithmic Improvements

The following algorithmic enhancements have been implemented or are recommended:

- **Double DQN (Implemented)**: Reduces overestimation bias by decoupling action selection from action evaluation. The policy network selects the best next action, while the target network evaluates its Q-value. This is a well-known fix for Q-value overestimation in DQN (van Hasselt et al., 2016).
- **Prioritized Experience Replay**: Samples important transitions more frequently, helping retain knowledge of successful trajectories
- **Dueling DQN**: Separates value and advantage estimation, potentially improving stability

### 7.5 Separate Best Weights Buffer

Maintain a dedicated `best_weights` variable that is only updated when a new evaluation maximum is achieved, independent of the training checkpoint:

```python
if eval_score > best_eval_score:
    best_eval_score = eval_score
    best_weights = copy.deepcopy(model.state_dict())
    torch.save(best_weights, 'true_best_model.pt')
```

### 7.6 Regular Evaluation During Training

Run deterministic evaluation every N episodes during training to detect forgetting early:

```python
eval_frequency = 50
if episode % eval_frequency == 0:
    eval_score = evaluate(model, epsilon=0)
    log_eval_score(eval_score)
```

---

## 8. Conclusion

This analysis reveals a classic reinforcement learning failure mode: **catastrophic forgetting masked by misleading training metrics**. The DQN agent did genuinely learn a successful policy around episode 1559, achieving a score of 1063.2 with 5 completed laps. However, continued training caused the network to overwrite this knowledge, and the checkpointing mechanism failed to preserve the peak performance model.

The evaluation score of 95.70 with 0 laps represents not a failure to learn, but a failure to **retain** what was learned. This distinction is crucial: the agent was capable of solving the environment, but the training process did not preserve that capability.

For the project report, this failure should be presented not as a simple mistake, but as a demonstration of the inherent challenges in deep reinforcement learning. DQN is known to be unstable, catastrophic forgetting is a well-documented phenomenon, and checkpointing strategies must be carefully designed to capture peak performance. The honest analysis of this failure—and the technical understanding required to diagnose it—represents genuine learning about the complexities of training neural networks in non-stationary environments.

The recommendations provided above, if implemented, would significantly improve the reliability of future training runs and ensure that peak performance is both achieved and preserved.

---

## Appendix: Summary of Key Data Points

| Data Point | Value | Significance |
|------------|-------|--------------|
| Training peak episode | 1559 | True best performance |
| Training peak score | 1063.2 | Best achieved score |
| Training peak laps | 5 | Successful multi-lap completion |
| Training end episode | 2000 | Training termination |
| Training end score | 29.2 | Complete policy failure |
| Checkpoint save episode | 1800 | After peak, during degradation |
| Checkpoint metadata score | 518.5 | Historical max at save time |
| `best_model.pt` eval score | 95.70 | Catastrophic evaluation failure |
| `checkpoint_ep10000.pt` eval | 95.70 | Consistent degradation |
| Evaluation standard deviation | 0.00 | Deterministic poor policy |

---

*Document prepared for MSc project report. This analysis demonstrates the importance of rigorous evaluation, frequent checkpointing, and understanding the instability inherent in deep reinforcement learning algorithms.*
