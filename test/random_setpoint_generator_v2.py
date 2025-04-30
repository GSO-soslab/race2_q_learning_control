#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import random
import time
import numpy as np
from std_msgs.msg import Header
from geometry_msgs.msg import Vector3
from mvp_msgs.msg import ControlProcess


class OUNoise:
    """Ornstein-Uhlenbeck process noise generator"""
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.copy(self.mu)
        self.reset()

    def reset(self):
        """Reset the internal state to mean"""
        self.state = np.copy(self.mu)

    def sample(self):
        """Update internal state and return it as noise"""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


class BetterExplorationPublisher(Node):
    def __init__(self):
        super().__init__('better_exploration_publisher')
        
        # Define publisher
        self.set_point_pub = self.create_publisher(
            ControlProcess,
            '/race2_auv/controller/process/set_point',
            10
        )
        
        # Parameters
        self.declare_parameter('rate_hz', 10.0)
        self.declare_parameter('min_depth', 0.5)  # Minimum safe depth
        self.declare_parameter('max_depth', 4.5)  # Maximum safe depth (below your simulator limit)
        self.declare_parameter('exploration_mode', 'structured')  # 'structured', 'random', or 'curriculum'
        
        # Frame IDs
        self.frame_id_value = "race2_auv/world_ned"
        self.child_frame_id = "race2_auv/cg_link"
        self.control_mode_value = "4dof"
        self.rate_hz = self.get_parameter('rate_hz').value
        
        # Depth limits
        self.min_depth = self.get_parameter('min_depth').value
        self.max_depth = self.get_parameter('max_depth').value
        self.exploration_mode = self.get_parameter('exploration_mode').value
        
        # Noise generator for exploration
        self.depth_noise = OUNoise(1, sigma=0.3)  # Ornstein-Uhlenbeck noise for smooth transitions
        self.yaw_noise = OUNoise(1, sigma=0.2)
        
        # Keep track of current setpoints to avoid abrupt changes
        self.current_depth_setpoint = 2.0
        self.current_yaw_setpoint = 0.0
        self.current_surge_setpoint = 0.2
        
        # Episode tracking
        self.episode_count = 0
        self.step_count = 0
        self.steps_per_episode = 700  # Match your RL steps per episode
        
        self.get_logger().info(f'Better Exploration Publisher initialized with depth range: {self.min_depth}m - {self.max_depth}m')
        self.get_logger().info(f'Exploration mode: {self.exploration_mode}')

    def run(self):
        """Main run loop with better exploration"""
        while rclpy.ok():
            if self.exploration_mode == 'structured':
                self.run_structured_exploration()
            elif self.exploration_mode == 'random':
                self.run_random_exploration()
            else:  # curriculum
                self.run_curriculum_based_exploration()

    def run_structured_exploration(self):
        """Use patterns that systematically explore the depth space"""
        self.get_logger().info(f'Starting structured exploration episode {self.episode_count}')
        
        # Reset step counter
        self.step_count = 0
        
        # Different patterns based on episode number
        pattern_type = self.episode_count % 4
        
        if pattern_type == 0:
            # Sawtooth pattern from shallow to deep
            self.run_depth_sawtooth()
        elif pattern_type == 1:
            # Step changes at different depths
            self.run_depth_steps()
        elif pattern_type == 2:
            # Sinusoidal pattern
            self.run_depth_sinusoid()
        else:
            # Fixed depths with yaw changes
            self.run_fixed_depths_with_yaw()
            
        self.episode_count += 1

    def run_random_exploration(self):
        """Random but smooth exploration across the depth range"""
        self.get_logger().info(f'Starting random exploration episode {self.episode_count}')
        
        # Reset step counter
        self.step_count = 0
        
        while self.step_count < self.steps_per_episode and rclpy.ok():
            # Apply noise for smooth transitions
            depth_noise = self.depth_noise.sample()[0]
            yaw_noise = self.yaw_noise.sample()[0]
            
            # Update setpoints with noise, keeping within bounds
            self.current_depth_setpoint = np.clip(
                self.current_depth_setpoint + depth_noise * 0.2,  # Scale noise for smoother changes
                self.min_depth,
                self.max_depth
            )
            
            self.current_yaw_setpoint = self.current_yaw_setpoint + yaw_noise * 0.1
            # Keep yaw in reasonable range
            if abs(self.current_yaw_setpoint) > np.pi:
                self.current_yaw_setpoint = np.sign(self.current_yaw_setpoint) * (abs(self.current_yaw_setpoint) % np.pi)
                
            # Small random variations in surge
            self.current_surge_setpoint = np.clip(
                self.current_surge_setpoint + np.random.uniform(-0.05, 0.05),
                0.0,  # Min forward speed
                0.3   # Max forward speed
            )
            
            # Publish current setpoint
            self.publish_setpoint(
                self.current_depth_setpoint,
                self.current_yaw_setpoint,
                self.current_surge_setpoint
            )
            
            self.step_count += 1
            
        self.episode_count += 1

    def run_curriculum_based_exploration(self):
        """Progressively expand exploration range as training progresses"""
        self.get_logger().info(f'Starting curriculum exploration episode {self.episode_count}')
        
        # Calculate progressive depth range based on episode count
        # Start with small range around 2.0m and gradually expand
        max_range = self.max_depth - self.min_depth
        progress = min(1.0, self.episode_count / 100.0)  # Full range after 100 episodes
        
        current_range = max_range * progress
        mid_depth = (self.min_depth + self.max_depth) / 2
        
        episode_min_depth = max(self.min_depth, mid_depth - current_range/2)
        episode_max_depth = min(self.max_depth, mid_depth + current_range/2)
        
        self.get_logger().info(f'Curriculum depth range for episode {self.episode_count}: {episode_min_depth:.2f}m - {episode_max_depth:.2f}m')
        
        # Reset counters
        self.step_count = 0
        
        # For early episodes, use simpler patterns
        if self.episode_count < 20:
            self.run_simple_depth_changes(episode_min_depth, episode_max_depth)
        else:
            # Use more complex patterns as training progresses
            pattern_type = self.episode_count % 3
            
            if pattern_type == 0:
                self.run_depth_sinusoid(min_depth=episode_min_depth, max_depth=episode_max_depth)
            elif pattern_type == 1:
                self.run_depth_steps(min_depth=episode_min_depth, max_depth=episode_max_depth)
            else:
                self.run_random_bounded_exploration(min_depth=episode_min_depth, max_depth=episode_max_depth)
                
        self.episode_count += 1

    # Specific pattern implementations
    def run_depth_sawtooth(self, min_depth=None, max_depth=None):
        """Sawtooth pattern from shallow to deep"""
        if min_depth is None: min_depth = self.min_depth
        if max_depth is None: max_depth = self.max_depth
        
        self.get_logger().info(f'Running depth sawtooth pattern from {min_depth}m to {max_depth}m')
        
        # Calculate steps for smooth transitions
        steps_per_cycle = min(self.steps_per_episode, 200)  # At most 200 steps per cycle
        depth_increment = (max_depth - min_depth) / steps_per_cycle
        
        while self.step_count < self.steps_per_episode and rclpy.ok():
            # Calculate cycle position
            cycle_position = self.step_count % steps_per_cycle
            
            # Linear increase in depth
            target_depth = min_depth + (depth_increment * cycle_position)
            
            # Small yaw oscillation
            target_yaw = 0.3 * np.sin(cycle_position / steps_per_cycle * 2 * np.pi)
            
            # Publish setpoint
            self.publish_setpoint(target_depth, target_yaw, 0.2)
            
            self.step_count += 1

    def run_depth_steps(self, min_depth=None, max_depth=None):
        """Step changes at different depths with some dwell time at each step"""
        if min_depth is None: min_depth = self.min_depth
        if max_depth is None: max_depth = self.max_depth
        
        self.get_logger().info(f'Running depth step pattern between {min_depth}m and {max_depth}m')
        
        # Create 5 depth levels
        depth_levels = np.linspace(min_depth, max_depth, 5)
        steps_per_level = self.steps_per_episode // len(depth_levels)
        
        while self.step_count < self.steps_per_episode and rclpy.ok():
            # Determine current level
            current_level_index = min(self.step_count // steps_per_level, len(depth_levels) - 1)
            target_depth = depth_levels[current_level_index]
            
            # Small random variations to explore around the level
            depth_variation = np.random.uniform(-0.2, 0.2)
            target_depth = np.clip(target_depth + depth_variation, min_depth, max_depth)
            
            # Random yaw
            target_yaw = np.random.uniform(-np.pi/4, np.pi/4)
            
            # Publish setpoint
            self.publish_setpoint(target_depth, target_yaw, 0.2)
            
            self.step_count += 1

    def run_depth_sinusoid(self, min_depth=None, max_depth=None):
        """Sinusoidal pattern through depth range"""
        if min_depth is None: min_depth = self.min_depth
        if max_depth is None: max_depth = self.max_depth
        
        self.get_logger().info(f'Running sinusoidal depth pattern between {min_depth}m and {max_depth}m')
        
        # We'll complete multiple full cycles during the episode
        num_cycles = 3
        
        while self.step_count < self.steps_per_episode and rclpy.ok():
            # Calculate phase based on step
            phase = (self.step_count / self.steps_per_episode) * num_cycles * 2 * np.pi
            
            # Calculate depth from sinusoid
            depth_range = max_depth - min_depth
            target_depth = min_depth + (depth_range / 2) * (1 + np.sin(phase))
            
            # Yaw that varies less frequently than depth
            target_yaw = np.pi/4 * np.sin(phase / 2)
            
            # Slightly varying surge
            target_surge = 0.2 + 0.05 * np.sin(phase / 3)
            
            # Publish setpoint
            self.publish_setpoint(target_depth, target_yaw, target_surge)
            
            self.step_count += 1

    def run_fixed_depths_with_yaw(self):
        """Hold specific depths while varying yaw"""
        self.get_logger().info('Running fixed depths with varying yaw')
        
        # Choose 3 depths to dwell at
        depths = [1.0, 2.0, 3.0]
        steps_per_depth = self.steps_per_episode // len(depths)
        
        while self.step_count < self.steps_per_episode and rclpy.ok():
            # Determine current depth
            depth_index = min(self.step_count // steps_per_depth, len(depths) - 1)
            target_depth = depths[depth_index]
            
            # Continuous yaw variation
            phase = (self.step_count % steps_per_depth) / steps_per_depth * 4 * np.pi
            target_yaw = np.pi/2 * np.sin(phase)
            
            # Publish setpoint
            self.publish_setpoint(target_depth, target_yaw, 0.2)
            
            self.step_count += 1

    def run_simple_depth_changes(self, min_depth, max_depth):
        """Simple depth changes for early curriculum learning"""
        self.get_logger().info(f'Running simple depth changes between {min_depth}m and {max_depth}m')
        
        # Just alternate between min and max depth with long dwell times
        middle_depth = (min_depth + max_depth) / 2
        depths = [min_depth, middle_depth, max_depth, middle_depth]
        steps_per_depth = self.steps_per_episode // len(depths)
        
        while self.step_count < self.steps_per_episode and rclpy.ok():
            # Determine current depth
            depth_index = min(self.step_count // steps_per_depth, len(depths) - 1)
            target_depth = depths[depth_index]
            
            # Fixed yaw and surge for simplicity in early training
            target_yaw = 0.0
            target_surge = 0.2
            
            # Publish setpoint
            self.publish_setpoint(target_depth, target_yaw, target_surge)
            
            self.step_count += 1

    def run_random_bounded_exploration(self, min_depth, max_depth):
        """Random but bounded exploration within specified depth range"""
        self.get_logger().info(f'Running random bounded exploration between {min_depth}m and {max_depth}m')
        
        # Initialize starting point
        self.current_depth_setpoint = (min_depth + max_depth) / 2
        self.current_yaw_setpoint = 0.0
        self.current_surge_setpoint = 0.2
        
        while self.step_count < self.steps_per_episode and rclpy.ok():
            # Apply noise for smooth transitions
            depth_noise = self.depth_noise.sample()[0]
            yaw_noise = self.yaw_noise.sample()[0]
            
            # Update setpoints with noise, keeping within episode bounds
            self.current_depth_setpoint = np.clip(
                self.current_depth_setpoint + depth_noise * 0.2,
                min_depth,
                max_depth
            )
            
            self.current_yaw_setpoint = self.current_yaw_setpoint + yaw_noise * 0.1
            if abs(self.current_yaw_setpoint) > np.pi/2:
                self.current_yaw_setpoint = np.sign(self.current_yaw_setpoint) * (np.pi/2)
                
            # Small random variations in surge
            self.current_surge_setpoint = np.clip(
                self.current_surge_setpoint + np.random.uniform(-0.02, 0.02),
                0.1,  # Min forward speed
                0.3   # Max forward speed
            )
            
            # Publish current setpoint
            self.publish_setpoint(
                self.current_depth_setpoint,
                self.current_yaw_setpoint,
                self.current_surge_setpoint
            )
            
            self.step_count += 1

    def publish_setpoint(self, depth, yaw, surge):
        """Helper to publish a setpoint with the given parameters"""
        msg = ControlProcess()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id_value
        msg.child_frame_id = self.child_frame_id
        msg.control_mode = self.control_mode_value
        
        msg.position = Vector3(x=0.0, y=0.0, z=depth)
        msg.orientation = Vector3(x=3.14, y=0.0, z=yaw)  # Fixed roll, pitch
        msg.velocity = Vector3(x=surge, y=0.0, z=0.0)    # Only surge velocity
        msg.angular_rate = Vector3(x=0.0, y=0.0, z=0.0)  # No angular rate
        
        self.set_point_pub.publish(msg)
        time.sleep(1.0 / self.rate_hz)

def main(args=None):
    rclpy.init(args=args)
    node = BetterExplorationPublisher()
    
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()