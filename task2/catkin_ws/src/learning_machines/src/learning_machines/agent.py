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
    """Soft Actor-Critic (SAC) agent for continuous action spaces.
    
    This implementation uses:
    - Gaussian policy with reparameterization trick
    - Q-networks that output Q-values for continuous actions
    - Continuous entropy calculation
    
    Args:
        state_dim: Dimension of the state space
        action_dim: Dimension of continuous action space
        lr: Learning rate for actor and critic networks
        gamma: Discount factor
        tau: Soft update coefficient for target networks
        alpha: Entropy coefficient (exploration-exploitation balance).
               Higher values encourage more exploration through entropy.
        batch_size: Batch size for training
        replay_size: Size of replay buffer
    """
    def __init__(
            self, state_dim: int, action_dim: int,
            lr=3e-4, gamma=0.99, tau=0.005,alpha=0.0,
            batch_size=64, replay_size=100000,
        ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha  # Entropy coefficient - can be modified during training
        self.batch_size = batch_size

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
        
        # Log std bounds for numerical stability
        self.log_std_min = -20
        self.log_std_max = 2

    def _build_actor(self):
        """Build actor network that outputs mean and log_std for Gaussian policy."""
        inp = layers.Input(shape=(self.state_dim,))
        x = layers.Dense(256, activation='relu')(inp)
        x = layers.Dense(256, activation='relu')(x)
        
        # Output mean and log_std for Gaussian distribution
        mean = layers.Dense(self.action_dim)(x)
        log_std = layers.Dense(self.action_dim)(x)
        
        model = keras.Model(inp, [mean, log_std])
        return model

    def _build_critic(self):
        """Build critic network that takes state and action as input."""
        state_input = layers.Input(shape=(self.state_dim,))
        action_input = layers.Input(shape=(self.action_dim,))
        
        # Concatenate state and action
        x = layers.Concatenate()([state_input, action_input])
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(256, activation='relu')(x)
        q_value = layers.Dense(1)(x)  # Single Q-value output
        
        return keras.Model([state_input, action_input], q_value)

    def select_action(self, state: np.ndarray, training: bool = True):
        """Select action using Gaussian policy with reparameterization."""
        s = state.reshape(1, -1).astype(np.float32)
        mean, log_std = self.actor(s, training=False)
        
        if training:
            # Sample from Gaussian distribution
            std = tf.exp(log_std)
            action = mean + std * tf.random.normal(tf.shape(mean))
        else:
            # Use mean for deterministic evaluation
            action = mean
        
        # Apply tanh squashing and scale to action bounds
        action = tf.tanh(action)
        
        return action.numpy()[0]

    def train_step(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.astype(np.float32)
        next_states = next_states.astype(np.float32)
        actions = actions.astype(np.float32)
        rewards = rewards.reshape(-1, 1).astype(np.float32)
        dones = dones.reshape(-1, 1).astype(np.float32)

        # Compute target Q-values
        next_mean, next_log_std = self.actor(next_states, training=False)
        next_std = tf.exp(tf.clip_by_value(next_log_std, self.log_std_min, self.log_std_max))
        
        # Sample next actions with reparameterization
        eps = tf.random.normal(tf.shape(next_mean))
        next_actions_unsquashed = next_mean + next_std * eps
        next_actions = tf.tanh(next_actions_unsquashed)
        
        # Compute log probability of next actions
        next_log_probs = -0.5 * (tf.square((next_actions_unsquashed - next_mean) / (next_std + 1e-8)) + 
                                  2 * next_log_std + np.log(2 * np.pi))
        next_log_probs = tf.reduce_sum(next_log_probs, axis=1, keepdims=True)
        # Correction for tanh squashing
        next_log_probs -= tf.reduce_sum(tf.math.log(1 - tf.square(next_actions) + 1e-6), axis=1, keepdims=True)
        
        # Target Q-values for next state
        target_q1_next = self.target_critic1([next_states, next_actions], training=False)
        target_q2_next = self.target_critic2([next_states, next_actions], training=False)
        target_q_next = tf.minimum(target_q1_next, target_q2_next)
        
        # Soft Q-value: E[Q] - alpha * H (entropy term)
        next_value = target_q_next - self.alpha * next_log_probs
        
        # TD target
        y = rewards + self.gamma * (1 - dones) * next_value

        # Update critics
        with tf.GradientTape() as tape:
            q1 = self.critic1([states, actions], training=True)
            q2 = self.critic2([states, actions], training=True)
            
            c_loss = tf.reduce_mean((q1 - y)**2) + tf.reduce_mean((q2 - y)**2)
        
        critic_vars = self.critic1.trainable_variables + self.critic2.trainable_variables
        grads = tape.gradient(c_loss, critic_vars)
        self.critic_opt.apply_gradients(zip(grads, critic_vars))

        # Update actor
        with tf.GradientTape() as tape:
            mean, log_std = self.actor(states, training=True)
            std = tf.exp(tf.clip_by_value(log_std, self.log_std_min, self.log_std_max))
            
            # Reparameterization trick
            eps = tf.random.normal(tf.shape(mean))
            actions_unsquashed = mean + std * eps
            sampled_actions = tf.tanh(actions_unsquashed)
            
            # Log probability
            log_probs = -0.5 * (tf.square((actions_unsquashed - mean) / (std + 1e-8)) + 
                                2 * log_std + np.log(2 * np.pi))
            log_probs = tf.reduce_sum(log_probs, axis=1, keepdims=True)
            log_probs -= tf.reduce_sum(tf.math.log(1 - tf.square(sampled_actions) + 1e-6), axis=1, keepdims=True)
            
            # Q-values for sampled actions
            q1_pi = self.critic1([states, sampled_actions], training=False)
            q2_pi = self.critic2([states, sampled_actions], training=False)
            q_pi = tf.minimum(q1_pi, q2_pi)
            
            # Actor loss: maximize Q - alpha * log_prob
            a_loss = tf.reduce_mean(self.alpha * log_probs - q_pi)
        
        actor_grads = tape.gradient(a_loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # Soft update target networks
        for var, tgt in zip(self.critic1.variables, self.target_critic1.variables):
            tgt.assign(self.tau * var + (1 - self.tau) * tgt)
        for var, tgt in zip(self.critic2.variables, self.target_critic2.variables):
            tgt.assign(self.tau * var + (1 - self.tau) * tgt)

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
            "algorithm": "SAC_Continuous",
            "state_dim": int(self.state_dim),
            "action_dim": int(self.action_dim),
            "learning_rate": float(self.actor_opt.learning_rate.numpy()),
            "gamma": float(self.gamma),
            "tau": float(self.tau),
            "alpha": float(self.alpha),
            "batch_size": int(self.batch_size),
            "replay_buffer_size": int(self.replay_buffer.buffer.maxlen),
        }
