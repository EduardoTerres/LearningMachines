import numpy as np
import random
from typing import Optional, Tuple
import tensorflow as tf
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


class SACAgent:
    """Soft Actor-Critic (SAC) agent adapted for discrete action spaces.
    
    This implementation uses:
    - Categorical policy with Gumbel-Softmax for reparameterization
    - Q-networks that output Q-values for each discrete action
    - Discrete entropy calculation
    
    Args:
        state_dim: Dimension of the state space
        action_dim: Number of discrete actions
        action_low: Not used (for compatibility)
        action_high: Not used (for compatibility)
        lr: Learning rate for actor and critic networks
        gamma: Discount factor
        tau: Soft update coefficient for target networks
        alpha: Entropy coefficient (exploration-exploitation balance).
               Higher values encourage more exploration through entropy.
        batch_size: Batch size for training
        replay_size: Size of replay buffer
    """
    def __init__(self, state_dim: int, action_dim: int, action_low=None, action_high=None,
                 lr=3e-3, gamma=0.99, tau=0.005, alpha=0.15, batch_size=64, replay_size=100000,
                 epsilon_start=0.1, epsilon_end=0.01, epsilon_decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha  # Entropy coefficient - can be modified during training
        self.batch_size = batch_size
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.replay_buffer = ReplayBuffer(replay_size)

        # networks
        self.actor = self._build_actor()
        self.critic1 = self._build_critic()
        self.critic2 = self._build_critic()
        self.target_critic1 = self._build_critic()
        self.target_critic2 = self._build_critic()
        self.target_critic1.set_weights(self.critic1.get_weights())
        self.target_critic2.set_weights(self.critic2.get_weights())

        self.actor_opt = keras.optimizers.Adam(lr)
        self.critic_opt = keras.optimizers.Adam(lr)

    def _build_actor(self):
        """Build actor network that outputs action probabilities (logits)."""
        inp = layers.Input(shape=(self.state_dim,))
        x = layers.Dense(64, activation='relu')(inp)
        x = layers.Dense(64, activation='relu')(x)
        logits = layers.Dense(self.action_dim)(x)  # No activation - raw logits
        model = keras.Model(inp, logits)
        return model

    def _build_critic(self):
        """Build critic network that outputs Q-values for all actions."""
        s = layers.Input(shape=(self.state_dim,))
        x = layers.Dense(64, activation='relu')(s)
        x = layers.Dense(64, activation='relu')(x)
        q_values = layers.Dense(self.action_dim)(x)  # Q-value for each action
        return keras.Model(s, q_values)

    def decay_epsilon(self) -> None:
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_end:
                self.epsilon = self.epsilon_end

    def select_action(self, state: np.ndarray, training: bool = True):
        """Select action using epsilon-greedy with categorical distribution."""
        s = state.reshape(1, -1).astype(np.float32)
        logits = self.actor(s, training=False)
        
        if training:
            # Epsilon-greedy exploration
            if random.random() < self.epsilon:
                action_idx = random.randint(0, self.action_dim - 1)
            else:
                # Sample from categorical distribution
                action_idx = tf.random.categorical(logits, 1)[0, 0].numpy()
        else:
            # Use greedy action during evaluation
            # action_idx = tf.argmax(logits[0]).numpy()
            action_idx = tf.random.categorical(logits, 1)[0, 0].numpy()
        
        return int(action_idx)

    def train_step(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.astype(np.float32)
        next_states = next_states.astype(np.float32)
        actions = actions.astype(np.int32)
        rewards = rewards.reshape(-1, 1).astype(np.float32)
        dones = dones.reshape(-1, 1).astype(np.float32)

        # Compute target Q-values
        next_logits = self.actor(next_states, training=False)
        next_probs = tf.nn.softmax(next_logits)
        next_log_probs = tf.nn.log_softmax(next_logits)
        
        # Target Q-values for next state
        target_q1_next = self.target_critic1(next_states, training=False)
        target_q2_next = self.target_critic2(next_states, training=False)
        target_q_next = tf.minimum(target_q1_next, target_q2_next)
        
        # Soft Q-value: E[Q] - alpha * H (entropy term)
        # For discrete: sum over actions of: prob * (Q - alpha * log(prob))
        next_value = tf.reduce_sum(
            next_probs * (target_q_next - self.alpha * next_log_probs),
            axis=1,
            keepdims=True
        )
        
        # TD target
        y = rewards + self.gamma * (1 - dones) * next_value

        # Update critics
        with tf.GradientTape() as tape:
            q1_all = self.critic1(states, training=True)
            q2_all = self.critic2(states, training=True)
            
            # Get Q-values for taken actions
            indices = tf.stack([tf.range(self.batch_size), actions], axis=1)
            q1 = tf.expand_dims(tf.gather_nd(q1_all, indices), 1)
            q2 = tf.expand_dims(tf.gather_nd(q2_all, indices), 1)
            
            c_loss = tf.reduce_mean((q1 - y)**2) + tf.reduce_mean((q2 - y)**2)
        
        critic_vars = self.critic1.trainable_variables + self.critic2.trainable_variables
        grads = tape.gradient(c_loss, critic_vars)
        self.critic_opt.apply_gradients(zip(grads, critic_vars))

        # Update actor
        with tf.GradientTape() as tape:
            logits = self.actor(states, training=True)
            probs = tf.nn.softmax(logits)
            log_probs = tf.nn.log_softmax(logits)
            
            # Q-values from current critics
            q1_all = self.critic1(states, training=False)
            q2_all = self.critic2(states, training=False)
            q_all = tf.minimum(q1_all, q2_all)
            
            # Policy objective: maximize E[Q - alpha * log(pi)]
            # Equivalent to minimizing: sum over actions of: prob * (alpha * log(prob) - Q)
            inside_term = self.alpha * log_probs - q_all
            a_loss = tf.reduce_mean(tf.reduce_sum(probs * inside_term, axis=1))
        
        actor_grads = tape.gradient(a_loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # Soft update target networks
        for var, tgt in zip(self.critic1.variables, self.target_critic1.variables):
            tgt.assign(self.tau * var + (1 - self.tau) * tgt)
        for var, tgt in zip(self.critic2.variables, self.target_critic2.variables):
            tgt.assign(self.tau * var + (1 - self.tau) * tgt)

        # Decay epsilon
        self.decay_epsilon()

        return float(c_loss.numpy())

    def save_model(self, filepath: str) -> None:
        """Save actor model."""
        self.actor.save(filepath)

    def load_model(self, filepath: str) -> None:
        """Load actor model."""
        self.actor = keras.models.load_model(filepath)

    def get_properties(self) -> dict:
        """Return agent hyperparameters for logging."""
        return {
            "algorithm": "SAC",
            "state_dim": int(self.state_dim),
            "action_dim": int(self.action_dim),
            "learning_rate": float(self.actor_opt.learning_rate.numpy()),
            "gamma": float(self.gamma),
            "tau": float(self.tau),
            "alpha": float(self.alpha),
            "batch_size": int(self.batch_size),
            "replay_buffer_size": int(self.replay_buffer.buffer.maxlen),
            "epsilon_start": float(self.epsilon),
            "epsilon_end": float(self.epsilon_end),
            "epsilon_decay": float(self.epsilon_decay)
        }


class DQNAgent:
    """Deep Q-Network agent (clean reimplementation).

    Interface-compatible with the existing agents:
    - `select_action(state, training=True)` -> int (discrete action)
    - `train_step()` -> Optional[float] (returns loss or None if not enough data)
    - `save_model(filepath)`, `load_model(filepath)`

    This implementation uses a target network, an experience replay buffer
    (the `ReplayBuffer` above), epsilon-greedy exploration and a simple
    Double-DQN style target calculation.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 64,
        replay_size: int = 100000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        target_update_frequency: int = 1000,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency

        # exploration
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # networks
        self.q_network = create_q_network(state_dim, action_dim)
        self.target_network = create_q_network(state_dim, action_dim)
        self.update_target_network()

        # optimizer / loss
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.loss_fn = keras.losses.MeanSquaredError()

        # replay buffer
        self.replay_buffer = ReplayBuffer(replay_size)

        self.step_count = 0

    def update_target_network(self) -> None:
        self.target_network.set_weights(self.q_network.get_weights())

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        s = state.reshape(1, -1).astype(np.float32)
        q_vals = self.q_network.predict(s, verbose=0)[0]
        return int(np.argmax(q_vals))

    def decay_epsilon(self) -> None:
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_end:
                self.epsilon = self.epsilon_end

    def train_step(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = states.astype(np.float32)
        next_states = next_states.astype(np.float32)
        actions = actions.astype(np.int32)
        rewards = rewards.astype(np.float32)
        dones = dones.astype(np.float32)

        # Double-DQN target calculation:
        # action selection by online network, evaluation by target network
        next_q_online = self.q_network.predict(next_states, verbose=0)
        next_actions = np.argmax(next_q_online, axis=1)
        next_q_target = self.target_network.predict(next_states, verbose=0)
        next_q_values = next_q_target[np.arange(self.batch_size), next_actions]

        targets = rewards + (1.0 - dones) * (self.gamma * next_q_values)

        # Get current Q-values and replace only taken actions with targets
        with tf.GradientTape() as tape:
            q_preds = self.q_network(states, training=True)
            # gather predicted Q for taken actions
            indices = tf.stack([tf.range(self.batch_size), tf.convert_to_tensor(actions)], axis=1)
            q_taken = tf.gather_nd(q_preds, indices)
            loss = self.loss_fn(targets, q_taken)

        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

        # update target network periodically (hard update)
        self.step_count += 1
        if self.step_count % self.target_update_frequency == 0:
            self.update_target_network()

        # decay epsilon
        self.decay_epsilon()

        return float(loss.numpy())

    def save_model(self, filepath: str) -> None:
        self.q_network.save(filepath)

    def load_model(self, filepath: str) -> None:
        self.q_network = keras.models.load_model(filepath)
        self.update_target_network()

    def get_properties(self) -> dict:
        """Return agent hyperparameters for logging."""
        return {
            "algorithm": "DQN",
            "state_dim": int(self.state_dim),
            "action_dim": int(self.action_dim),
            "learning_rate": float(self.optimizer.learning_rate.numpy()),
            "gamma": float(self.gamma),
            "batch_size": int(self.batch_size),
            "replay_buffer_size": int(self.replay_buffer.buffer.maxlen),
            "epsilon_start": float(self.epsilon),
            "epsilon_end": float(self.epsilon_end),
            "epsilon_decay": float(self.epsilon_decay),
            "target_update_frequency": int(self.target_update_frequency)
        }