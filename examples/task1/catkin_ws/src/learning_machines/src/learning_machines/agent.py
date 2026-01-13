
import numpy as np
import random
from typing import Optional, Tuple
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque

class ReplayBuffer:
    """Experience Replay Buffer for DQN Agent."""
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        return len(self.buffer)


def create_q_network(state_dim: int = 8, num_actions: int = 6) -> keras.Model:
    """
    Create Q-network neural network.
    
    Architecture:
    - Input: State vector [8 sensor values]
    - Hidden layers: 2 layers of 64 neurons each
    - Output: Q-values for each action [6 values]
    """
    model = keras.Sequential([
        layers.Input(shape=(state_dim,)),
        layers.Dense(64, activation='relu', name='hidden1'),
        layers.Dense(64, activation='relu', name='hidden2'),
        layers.Dense(num_actions, activation='linear', name='q_values')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )
    
    return model


class DQNAgent:
    """DQN Agent for discrete action control."""
    
    def __init__(
        self,
        state_dim: int = 8,
        num_actions: int = 6,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 32,
        replay_buffer_size: int = 10000,
        target_update_frequency: int = 100
    ):
        """Initialize DQN Agent."""
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        
        # Create Q-network (main network)
        self.q_network = create_q_network(state_dim, num_actions)
        
        # Create target network (copy of Q-network)
        self.target_network = create_q_network(state_dim, num_actions)
        self.update_target_network()  # Initialize target = Q-network
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        
        # Training step counter
        self.step_count = 0
    
    def update_target_network(self) -> None:
        """Copy weights from Q-network to target network."""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        With probability ε: Random action (exploration)
        Otherwise: Best action according to Q-network (exploitation)
        """
        if training and random.random() < self.epsilon:
            # Explore: random action
            return random.randint(0, self.num_actions - 1)
        else:
            # Exploit: best action according to Q-network
            state_batch = state.reshape(1, -1)
            q_values = self.q_network.predict(state_batch, verbose=0)
            return np.argmax(q_values[0])
    
    def decay_epsilon(self) -> None:
        """Decay epsilon for exploration."""
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step.
        
        Steps:
        1. Sample batch from replay buffer
        2. Compute current Q-values
        3. Compute target Q-values using target network
        4. Update Q-network to minimize MSE
        """
        # Check if buffer has enough samples
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        
        # Current Q-values from Q-network
        current_q_values = self.q_network.predict(states, verbose=0)
        
        # Next Q-values from target network
        next_q_values = self.target_network.predict(next_states, verbose=0)
        
        # Compute target Q-values (Bellman equation)
        target_q_values = current_q_values.copy()
        
        for i in range(self.batch_size):
            if dones[i]:
                # Terminal state: target = reward only
                target_q_values[i][actions[i]] = rewards[i]
            else:
                # Non-terminal: target = reward + γ * max Q(s', a')
                target_q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(
                    next_q_values[i]
                )
        
        # Train Q-network to match targets
        history = self.q_network.fit(
            states, 
            target_q_values, 
            verbose=0, 
            epochs=1,
            batch_size=self.batch_size
        )
        
        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.target_update_frequency == 0:
            self.update_target_network()
        
        return history.history['loss'][0] if history.history['loss'] else None
    
    def save_model(self, filepath: str) -> None:
        """Save the Q-network model."""
        self.q_network.save(filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load a saved Q-network model."""
        self.q_network = keras.models.load_model(filepath)
        self.update_target_network()  # Update target network too