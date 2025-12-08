import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DQNetwork(nn.Module):
    """
    Deep Q-Network with batch normalization and dropout for stability.

    Architecture:
    - Input layer (state_dim)
    - 2 hidden layers with batch norm + ReLU + dropout
    - Output layer (action_dim)
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        return self.network(state)

class DQNAgent:
    """
    - Double DQN (reduces overestimation bias)
    - Gradient clipping (prevents exploding gradients)
    - Huber loss (robust to outliers)
    - Lower learning rate (0.0005 instead of 0.001)
    - Larger batch size (64 instead of 32)
    - Batch normalization and dropout in network
    """

    def __init__(self, state_dim, action_dim, lr=0.0005, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64, grad_clip=1.0):
        """
        Initialize DQN agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            lr: Learning rate (reduced to 0.0005 for stability)
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Exploration decay rate
            buffer_size: Replay buffer size
            batch_size: Training batch size (increased to 64)
            grad_clip: Gradient clipping threshold
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.grad_clip = grad_clip

        # Create Q-networks
        self.q_network = DQNetwork(state_dim, action_dim)
        self.target_network = DQNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer and loss function
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss - more robust than MSE

        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)

    def select_action(self, state):
        """
        Select action using epsilon-greedy strategy.

        Args:
            state: Current state

        Returns:
            action: Selected action
        """
        if random.random() < self.epsilon:
            # Exploration: random action
            return random.randint(0, self.action_dim - 1)
        else:
            # Exploitation: greedy action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train_step(self):
        """
        Perform one training step with Double DQN and gradient clipping.

        Returns:
            loss: Training loss (or None if buffer too small)
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch from replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors (fix: use np.array first)
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        # Current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Double DQN: use current network to select action, target to evaluate
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Compute loss
        loss = self.loss_fn(current_q.squeeze(), target_q)

        # Backward pass with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_clip)

        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Copy weights from Q-network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
