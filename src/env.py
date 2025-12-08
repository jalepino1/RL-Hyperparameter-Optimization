import gymnasium as gym
from gymnasium import spaces
import numpy as np
import lightgbm as lgb
from typing import Dict, List
from sklearn.metrics import mean_squared_error, mean_absolute_error

class EnergyForecastingEnv(gym.Env):
    """
    Custom Gymnasium environment for hyperparameter optimization
    in energy demand forecasting using reinforcement learning.

    State Space (12 dimensions):
    - Lagged demand statistics (mean, std)
    - Current hyperparameter indices (lr, depth, num_leaves, min_child_samples)
    - Performance metrics (rmse, mae, prev_rmse, rmse_trend)
    - Step counter

    Action Space (product of choices):
    - N learning rate values × N depth values × N num_leaves values × N min_child_samples values

    Reward: Multi-component shaping with performance, improvement, and stability signals
    """

    def __init__(self, train_data, val_data, features, target='load',
                 max_steps=20, include_stability_penalty=True):
        super(EnergyForecastingEnv, self).__init__()

        self.train_data = train_data
        self.val_data = val_data
        self.features = features
        self.target = target
        self.max_steps = max_steps
        self.include_stability_penalty = include_stability_penalty

        # Hyperparameter ranges (discrete)
        self.lr_values = [0.001, 0.005, 0.01, 0.05, 0.1]
        self.depth_values = [2, 3, 5, 7, 10]
        self.n_leaves_values = [10, 15, 20, 31, 50, 100]
        self.min_child_samples_values = [5, 10, 20]

        # Calculate action space size
        self.N_LR = len(self.lr_values)
        self.N_DEPTH = len(self.depth_values)
        self.N_LEAVES = len(self.n_leaves_values)
        self.N_MIN_CHILD_SAMPLES = len(self.min_child_samples_values)

        action_dim = self.N_LR * self.N_DEPTH * self.N_LEAVES * self.N_MIN_CHILD_SAMPLES
        self.action_space = spaces.Discrete(action_dim) # Updated action space

        # State space with 12 dimensions
        # [mean_load, std_load, lr_idx, depth_idx, leaves_idx, min_child_samples_idx, rmse, mae, prev_rmse, rmse_trend, step_count, best_rmse]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0, -100, 0, 0, 0]), # Added one 0 for min_child_samples_idx
            high=np.array([500, 200, self.N_LR-1, self.N_DEPTH-1, self.N_LEAVES-1, self.N_MIN_CHILD_SAMPLES-1, 200, 200, 200, 100, self.max_steps, 200]), # Adjusted high for indices and max_steps
            dtype=np.float32
        )

        # State variables
        self.current_step = 0
        self.lr_idx = 0 # Start with first value
        self.depth_idx = 0
        self.leaves_idx = 0
        self.child_samples_idx = 0 # New
        self.current_rmse = None
        self.prev_rmse = None
        self.current_mae = None
        self.best_rmse = float('inf')
        self.trial_history = []

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        self.current_step = 0
        self.lr_idx = 0
        self.depth_idx = 0
        self.leaves_idx = 0
        self.child_samples_idx = 0 # New
        self.current_rmse = None
        self.prev_rmse = None
        self.current_mae = None
        self.best_rmse = float('inf')
        self.trial_history = []

        state = self._get_state()
        return state, {}

    def _get_state(self):
        """Construct state representation"""
        # Lagged demand statistics
        recent_loads = self.train_data[self.target].tail(100)
        mean_load = float(recent_loads.mean())
        std_load = float(recent_loads.std())

        # Hyperparameter indices
        lr_state = float(self.lr_idx)
        depth_state = float(self.depth_idx)
        leaves_state = float(self.leaves_idx)
        child_samples_state = float(self.child_samples_idx) # New

        # Performance metrics
        rmse = float(self.current_rmse if self.current_rmse is not None else 0)
        mae = float(self.current_mae if self.current_mae is not None else 0)
        prev_rmse = float(self.prev_rmse if self.prev_rmse is not None else 0)

        # Trend indicator
        if len(self.trial_history) > 1:
            rmse_trend = float(self.trial_history[-1]['rmse'] - self.trial_history[-2]['rmse'])
        else:
            rmse_trend = 0.0

        # Current state - 12 dimensions now
        state = np.array([
            mean_load, std_load, lr_state, depth_state, leaves_state, child_samples_state, # New
            rmse, mae, prev_rmse, rmse_trend, float(self.current_step),
            self.best_rmse
        ], dtype=np.float32)

        return state

    def step(self, action):
        """Execute action and return new state, reward, done, truncated, info"""
        # Decode action to hyperparameter indices
        self.child_samples_idx = action % self.N_MIN_CHILD_SAMPLES
        self.leaves_idx = (action // self.N_MIN_CHILD_SAMPLES) % self.N_LEAVES
        self.depth_idx = (action // (self.N_MIN_CHILD_SAMPLES * self.N_LEAVES)) % self.N_DEPTH
        self.lr_idx = action // (self.N_MIN_CHILD_SAMPLES * self.N_LEAVES * self.N_DEPTH)

        # Clip indices to valid ranges (although modulo should keep them in range, good practice)
        self.lr_idx = np.clip(self.lr_idx, 0, self.N_LR - 1)
        self.depth_idx = np.clip(self.depth_idx, 0, self.N_DEPTH - 1)
        self.leaves_idx = np.clip(self.leaves_idx, 0, self.N_LEAVES - 1)
        self.child_samples_idx = np.clip(self.child_samples_idx, 0, self.N_MIN_CHILD_SAMPLES - 1)

        # Get actual hyperparameter values
        learning_rate = self.lr_values[self.lr_idx]
        max_depth = self.depth_values[self.depth_idx]
        num_leaves = self.n_leaves_values[self.leaves_idx]
        min_child_samples = self.min_child_samples_values[self.child_samples_idx]

        # Train model with current hyperparameters
        model = lgb.LGBMRegressor(
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            n_estimators=100,
            random_state=42,
            verbose=-1
        )

        X_train = self.train_data[self.features]
        y_train = self.train_data[self.target]
        X_val = self.val_data[self.features]
        y_val = self.val_data[self.target]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        # Calculate metrics
        self.prev_rmse = self.current_rmse
        self.current_rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        self.current_mae = float(mean_absolute_error(y_val, y_pred))

        # Track best RMSE
        if self.current_rmse < self.best_rmse:
            self.best_rmse = self.current_rmse

        # Calculate shaped reward
        reward = self._calculate_shaped_reward()

        # Record trial
        self.trial_history.append({
            'lr': learning_rate,
            'max_depth': max_depth,
            'num_leaves': num_leaves,
            'min_child_samples': min_child_samples, # New
            'rmse': self.current_rmse,
            'mae': self.current_mae,
            'step': self.current_step
        })

        self.current_step += 1
        done = self.current_step >= self.max_steps

        state = self._get_state()

        info = {
            'rmse': self.current_rmse,
            'mae': self.current_mae,
            'lr': learning_rate,
            'max_depth': max_depth,
            'num_leaves': num_leaves,
            'min_child_samples': min_child_samples, # New
            'best_rmse': self.best_rmse
        }

        return state, reward, done, False, info

    def _calculate_shaped_reward(self):
        """Simple, stable reward"""
        # Just penalize RMSE directly - scale it reasonably
        # Normalize to [-1, 0] range instead of wild swings
        normalized_rmse = self.current_rmse / 50.0  # 50 is approximate max
        reward = -normalized_rmse  # Range: -1 to 0 (stable!)

        # Small bonus for improvement (not * 50!)
        if self.prev_rmse is not None:
            improvement = self.prev_rmse - self.current_rmse
            if improvement > 0:
                reward += 0.1  # Small fixed bonus

        return reward