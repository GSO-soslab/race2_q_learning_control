import numpy as np

# class OUActionNoise:
#     def __init__(self, mean, std_deviation, theta=0.25, dt=1e-2, x0=None, decay_period=100000):
#         self.theta = theta
#         self.mean = mean
#         # self.std_dev = std_deviation
#         self.dt = dt
#         self.x0 = x0
#         self.reset()
        
#         # Noise annealing parameters
#         self.initial_std = std_deviation.copy()
#         self.min_std = 0.05 * std_deviation  # Minimum noise level (5% of initial)
#         self.decay_period = decay_period  # Steps over which to decay noise
#         self.step_count = 0
        
#     def __call__(self):
#         # Update internal state
#         x = self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt + \
#             self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        
#         # Store x for next call
#         self.x_prev = x
#         return x
        
#     def reset(self):
#         if self.x0 is not None:
#             self.x_prev = self.x0
#         else:
#             self.x_prev = np.zeros_like(self.mean)
            
#     def update_std(self):
#         """Update standard deviation based on annealing schedule"""
#         self.step_count += 1
#         if self.step_count <= self.decay_period:
#             # Linear decay
#             decay_factor = 1.0 # - (self.step_count / self.decay_period) * (1.0 - self.min_std / self.initial_std)
#             self.std_dev = self.initial_std * decay_factor

# class OUActionNoise:
#     def __init__(self, mean, std_deviation, theta=0.5, dt=1e-2, x0=None, decay_period=100000):
#         self.theta = theta
#         self.mean = mean
#         self.dt = dt
#         self.x0 = x0
#         self.reset()
        
#         self.initial_std = std_deviation.copy()
#         self.min_std = 0.05 * std_deviation
#         self.decay_period = decay_period
#         self.step_count = 0
#         self.std_dev = std_deviation.copy()

#     def __call__(self):
#         self.update_std()
#         x = self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt + \
#             self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
#         self.x_prev = x
#         return x
        
#     def reset(self):
#         self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mean)
        
#     def update_std(self):
#         self.step_count += 1
#         if self.step_count <= self.decay_period:
#             decay_ratio = self.step_count / self.decay_period
#             decay_factor = 1.0 - decay_ratio * (1.0 - self.min_std / self.initial_std)
#             self.std_dev = self.initial_std * decay_factor


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.25, dt=1e-2, x0=None, decay_period=500000, 
                 min_std_ratio=0.1, force_exploration=True, exploration_flip_prob=0.1, decay_type="exponential"):
        """
        Ornstein-Uhlenbeck process noise generator with enhanced exploration capabilities.
        
        Args:
            mean: Mean of the noise
            std_deviation: Standard deviation of the noise
            theta: How fast to revert to the mean (higher = faster)
            dt: Time step
            x0: Initial state
            decay_period: Steps over which to decay noise (default: 500000 - much slower decay)
            min_std_ratio: Minimum noise level as ratio of initial std (default: 0.1 - higher floor)
            force_exploration: Whether to force exploration in both directions
            exploration_flip_prob: Probability to flip the sign of noise to force exploration
            decay_type: Type of decay schedule ("exponential", "linear", or "warmup_linear")
        """
        self.theta = theta
        self.mean = np.array(mean)  # Ensure it's a numpy array
        self.std_dev = np.array(std_deviation).copy()  # Fixed missing assignment
        self.dt = dt
        self.x0 = x0
        
        # Noise annealing parameters
        self.initial_std = np.array(std_deviation).copy()
        self.min_std = min_std_ratio * self.initial_std  # Minimum noise level
        self.decay_period = decay_period
        self.step_count = 0
        self.decay_type = decay_type
        
        # Enhanced exploration parameters
        self.force_exploration = force_exploration
        self.exploration_flip_prob = exploration_flip_prob
        self.direction_history = np.zeros_like(self.mean)
        self.consecutive_same_dir = np.zeros_like(self.mean, dtype=int)
        
        self.reset()
        
    def __call__(self):
        # Update internal state using Ornstein-Uhlenbeck process
        x = self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt + \
            self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        
        # Enhanced exploration to prevent getting stuck in one direction
        if self.force_exploration:
            for i in range(len(x)):
                # Track direction history - avoid boolean evaluation of arrays
                current_sign = np.sign(x[i])
                prev_sign = np.sign(self.x_prev[i])
                
                # Check if same direction (avoiding array comparison)
                if current_sign == prev_sign and self.x_prev[i] != 0:  # Same direction as before
                    self.consecutive_same_dir[i] += 1
                else:
                    self.consecutive_same_dir[i] = 0
                
                # Force direction change if stuck in same direction too long
                if self.consecutive_same_dir[i] > 20:  # Threshold for "stuck"
                    # Higher probability to flip direction the longer we're stuck
                    flip_prob = min(0.8, self.exploration_flip_prob * 
                                  (1 + self.consecutive_same_dir[i] / 20))
                    if np.random.random() < flip_prob:
                        x[i] = -x[i]  # Flip the direction
                        self.consecutive_same_dir[i] = 0
                
                # Random direction flips to ensure both directions are explored
                elif np.random.random() < self.exploration_flip_prob:
                    x[i] = -x[i]
        
        # Store x for next call
        self.x_prev = x.copy()  # Make a copy to avoid reference issues
        
        # Update noise level based on annealing schedule
        self.update_std()
        
        return x
    
    def reset(self):
        if self.x0 is not None:
            self.x_prev = np.array(self.x0).copy()
        else:
            self.x_prev = np.zeros_like(self.mean)
        self.consecutive_same_dir = np.zeros_like(self.mean, dtype=int)
    
    def update_std(self):
        """Update standard deviation based on annealing schedule"""
        self.step_count += 1
        if self.step_count <= self.decay_period:
            progress = self.step_count / self.decay_period
            decay_factor = 1.0
            
            if self.decay_type == "exponential":
                # Exponential decay that starts very slow and accelerates later
                # This maintains higher noise levels for much longer during training
                decay_exponent = 3.0  # Lower value = slower decay (was 5)
                decay_factor = np.exp(-decay_exponent * progress * progress)
            
            elif self.decay_type == "linear":
                # Simple linear decay but much slower than original
                decay_factor = 1.0 - progress * (1.0 - self.min_std / self.initial_std)
            
            elif self.decay_type == "warmup_linear":
                # Linear decay with warmup period
                warmup_fraction = 0.3  # First 30% of steps maintain full noise
                if progress < warmup_fraction:
                    decay_factor = 1.0
                else:
                    adjusted_progress = (progress - warmup_fraction) / (1 - warmup_fraction)
                    decay_factor = 1.0 - adjusted_progress * (1.0 - self.min_std / self.initial_std)
            
            # Handle scalar and array min_std cases correctly
            if isinstance(decay_factor, (int, float)):
                decay_factor = max(decay_factor, np.min(self.min_std / self.initial_std))
            else:
                # Element-wise comparison for arrays
                min_ratio = self.min_std / self.initial_std
                decay_factor = np.maximum(decay_factor, min_ratio)
                
            self.std_dev = self.initial_std * decay_factor