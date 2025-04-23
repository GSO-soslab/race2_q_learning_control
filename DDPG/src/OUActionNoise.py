import numpy as np

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.2, dt=1e-2, x0=None, decay_period=100000):
        self.theta = theta
        self.mean = mean
        # self.std_dev = std_deviation
        self.dt = dt
        self.x0 = x0
        self.reset()
        
        # Noise annealing parameters
        self.initial_std = std_deviation.copy()
        self.min_std = 0.05 * std_deviation  # Minimum noise level (5% of initial)
        self.decay_period = decay_period  # Steps over which to decay noise
        self.step_count = 0
        
    def __call__(self):
        # Update internal state
        x = self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt + \
            self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        
        # Store x for next call
        self.x_prev = x
        return x
        
    def reset(self):
        if self.x0 is not None:
            self.x_prev = self.x0
        else:
            self.x_prev = np.zeros_like(self.mean)
            
    def update_std(self):
        """Update standard deviation based on annealing schedule"""
        self.step_count += 1
        if self.step_count <= self.decay_period:
            # Linear decay
            decay_factor = 1.0 # - (self.step_count / self.decay_period) * (1.0 - self.min_std / self.initial_std)
            self.std_dev = self.initial_std * decay_factor