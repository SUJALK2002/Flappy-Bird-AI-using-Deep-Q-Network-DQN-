# Flappy Bird AI using Deep Q-Network (DQN)

## 🎮 Project Overview

This project implements a **Deep Reinforcement Learning** agent that learns to play **Flappy Bird** using the **Deep Q-Network (DQN)** algorithm. The agent interacts with the game environment, learns from its experiences, and gradually improves its performance through trial and error.

The implementation combines several key reinforcement learning techniques to achieve stable and efficient learning:

* **Deep Q-Network (DQN)**
* **Experience Replay**
* **Target Network**
* **Epsilon-Greedy Exploration**
* **PyTorch Neural Networks**

Over time, the agent learns when to **flap** and when to **glide**, maximizing its survival time and overall score.

---

## 🚀 Features

* Deep Reinforcement Learning using DQN
* Experience Replay Buffer
* Target Network Synchronization
* Epsilon-Greedy Exploration Strategy
* GPU Support (CUDA/MPS)
* Automatic Model Checkpoint Saving
* Training Log Generation
* Real-Time Gameplay Rendering
* Configurable Hyperparameters using YAML
* PyTorch-Based Neural Network Architecture

---

## 📂 Project Structure

```text
FlappyBird-DQN/
│
├── train.py                # Main training and testing script
├── dqn.py                  # Deep Q-Network architecture
├── experience_replay.py    # Replay memory implementation
├── parameters.yaml         # Hyperparameter configurations
│
├── runs/
│   ├── model.pt            # Best saved model
│   └── training.log        # Training logs
│
├── requirements.txt
└── README.md
```

---

## 🧠 Deep Q-Network Architecture

The Deep Q-Network approximates the action-value function:

```math
Q(s, a)
```

### Input Features

The neural network receives the environment state, including:

* Bird vertical position
* Bird vertical velocity
* Horizontal distance to the next pipe
* Vertical position of the pipe gap

### Output

The network predicts the Q-value for each possible action.

| Action | Description |
| ------ | ----------- |
| 0      | Do Nothing  |
| 1      | Flap        |

During exploitation, the agent selects the action with the highest predicted Q-value.

---

## 🔄 Training Process

The agent follows the standard DQN training pipeline:

1. Observe the current state.
2. Select an action using the epsilon-greedy policy.
3. Execute the action in the environment.
4. Receive a reward and observe the next state.
5. Store the experience in replay memory.
6. Sample a mini-batch of experiences.
7. Compute target Q-values.
8. Update the policy network using gradient descent.
9. Periodically synchronize the target network.
10. Repeat until convergence.

---

## 🏗 Reinforcement Learning Components

### State Space

The environment observation contains information about:

* Bird position
* Bird velocity
* Distance to upcoming pipes
* Pipe gap positions

These features allow the agent to understand its current situation and make informed decisions.

### Action Space

The agent can perform two actions:

| Action | Description |
| ------ | ----------- |
| 0      | No Flap     |
| 1      | Flap        |

### Reward Function

The reward mechanism encourages survival and successful navigation:

* Positive reward for staying alive
* Positive reward for passing pipes
* Negative reward for collisions

### Experience Replay

Experiences are stored as tuples:

```python
(state, action, next_state, reward, done)
```

Random mini-batches are sampled from memory to break correlations between consecutive experiences and improve learning stability.

### Target Network

A separate target network is maintained to generate stable learning targets. The target network is periodically synchronized with the policy network, reducing oscillations during training.

---

## 📈 Learning Objective

The agent aims to maximize the cumulative discounted reward by learning the optimal action-value function:

```math
Q(s,a)
```

Using Bellman updates, the network gradually improves its estimates and develops a strategy capable of surviving longer and achieving higher scores in Flappy Bird.
