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

class CustomSetPointPublisher(Node):
    def __init__(self):
        super().__init__('custom_set_point_publisher')
        
        # Define publisher for the custom topic
        self.set_point_pub = self.create_publisher(
            ControlProcess,
            '/race2_auv/controller/process/set_point',
            10
        )
        
        # Declare parameters with default values
        self.declare_parameter('random_duration', 20)
        self.declare_parameter('rate_hz', 10.0)
        self.declare_parameter('enable_resets', True)
        self.declare_parameter('reset_interval', 2.0)
        
        # Add exploration parameters
        self.declare_parameter('exploration_factor', 1.0)
        self.declare_parameter('exploration_decay', 0.995)
        self.declare_parameter('range_expansion_rate', 1.01)
        
        # Parameters
        self.frame_id_value = "race2_auv/world_ned"
        self.child_frame_id = "race2_auv/cg_link"
        self.control_mode_value = "4dof"
        self.rate_hz = self.get_parameter('rate_hz').value
        self.reset_interval = self.get_parameter('reset_interval').value
        self.enable_resets = self.get_parameter('enable_resets').value
        
        # Initialize exploration parameters
        self.exploration_factor = self.get_parameter('exploration_factor').value
        self.exploration_decay = self.get_parameter('exploration_decay').value
        self.range_expansion_rate = self.get_parameter('range_expansion_rate').value
        
        # Initialize noise generators
        self.pos_noise = OUNoise(3, sigma=0.2)
        self.ori_noise = OUNoise(3, sigma=0.1)
        self.vel_noise = OUNoise(3, sigma=0.05)
        
        # Stable values to revert to or to use when a field is not being varied
        self.stable_position = Vector3(x=0.0, y=0.0, z=4.0)
        self.stable_orientation = Vector3(x=3.14, y=0.0, z=0.0)
        self.stable_velocity = Vector3(x=0.2, y=0.0, z=0.0)
        self.stable_angular_rate = Vector3(x=0.0, y=0.0, z=0.0)
        
        # Random ranges for fields we want to vary
        self.pos_z_min, self.pos_z_max = 0.0, 3.0
        self.ori_z_min, self.ori_z_max = -1.57, 1.57
        self.vel_x_min, self.vel_x_max = -0.28, 0.28
        
        # Calculate range spans
        self.max_pos_z_range = self.pos_z_max - self.pos_z_min
        self.max_ori_z_range = self.ori_z_max - self.ori_z_min
        self.max_vel_x_range = self.vel_x_max - self.vel_x_min
        
        # Initialize progressive ranges (start smaller)
        self.pos_z_range = 0.5
        self.ori_z_range = 0.5
        self.vel_x_range = 0.1
        
        # Initialize reset tracking
        self.last_reset_time = self.get_clock().now().seconds_nanoseconds()[0]
        self.episode_count = 0
        
        # Initialize previous values
        self.prev_position_z = self.stable_position.z
        self.prev_orientation_z = self.stable_orientation.z
        self.prev_velocity_x = self.stable_velocity.x
        
        self.get_logger().info('CustomSetPointPublisher initialized')

    def biased_random(self, prev_value, min_val, max_val, bias_factor=0.5):
        """Generate a random value biased toward the previous value"""
        new_value = random.uniform(min_val, max_val)
        return prev_value + bias_factor * (new_value - prev_value)
    
    def check_and_perform_reset(self):
        """Check if it's time for a reset and perform one if needed"""
        current_time = self.get_clock().now().seconds_nanoseconds()[0]
        
        if self.enable_resets and current_time - self.last_reset_time >= self.reset_interval:
            # Reset to a new random state
            self.get_logger().info(f'Performing periodic reset #{self.episode_count}')
            
            # Reset noise generators
            self.pos_noise.reset()
            self.ori_noise.reset()
            self.vel_noise.reset()
            
            # Reset to completely new random values
            current_position = Vector3(
                x=random.uniform(-1.0, 1.0),  # Wider range for reset
                y=random.uniform(-1.0, 1.0), 
                z=random.uniform(self.pos_z_min, self.pos_z_max)
            )
            
            current_orientation = Vector3(
                x=self.stable_orientation.x + random.uniform(-0.2, 0.2),
                y=random.uniform(-0.2, 0.2),
                z=random.uniform(self.ori_z_min, self.ori_z_max)
            )
            
            current_velocity = Vector3(
                x=random.uniform(self.vel_x_min, self.vel_x_max),
                y=random.uniform(-0.1, 0.1),
                z=random.uniform(-0.1, 0.1)
            )
            
            current_angular_rate = Vector3(
                x=random.uniform(-0.1, 0.1),
                y=random.uniform(-0.1, 0.1),
                z=random.uniform(-0.1, 0.1)
            )
            
            # Publish the reset values
            msg = ControlProcess()
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id_value
            msg.child_frame_id = self.child_frame_id
            msg.control_mode = self.control_mode_value
            msg.position = current_position
            msg.orientation = current_orientation
            msg.velocity = current_velocity
            msg.angular_rate = current_angular_rate
            
            # Publish the message
            self.set_point_pub.publish(msg)
            
            # Update reset time
            self.last_reset_time = self.get_clock().now().seconds_nanoseconds()[0]
            
            # Record current values as previous for biased random
            self.prev_position_z = current_position.z
            self.prev_orientation_z = current_orientation.z
            self.prev_velocity_x = current_velocity.x
            
            self.episode_count += 1
            
            return True
        
        return False

    def publish_values(self, position, orientation, velocity, angular_rate, duration):
        """Helper function to publish specified values for a given duration."""
        start_time = self.get_clock().now().seconds_nanoseconds()[0]
        end_time = start_time + duration
        
        while self.get_clock().now().seconds_nanoseconds()[0] < end_time:
            # Check if it's time for a reset
            if self.check_and_perform_reset():
                # A reset was performed, so break out of the current publishing loop
                return
            
            # Otherwise, continue with normal publishing
            msg = ControlProcess()
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id_value
            msg.child_frame_id = self.child_frame_id
            msg.control_mode = self.control_mode_value
            msg.position = position
            msg.orientation = orientation
            msg.velocity = velocity
            msg.angular_rate = angular_rate
            
            # Publish the message
            self.set_point_pub.publish(msg)
            time.sleep(1.0 / self.rate_hz)  # Sleep to maintain the desired frequency

    def run(self):
        """Main run loop implementing the publishing patterns."""
        period_index = 0
        random_duration = self.get_parameter('random_duration').value

        while rclpy.ok():
            # Set base values (start from stable)
            current_position = Vector3(x=self.stable_position.x, 
                                     y=self.stable_position.y, 
                                     z=self.stable_position.z)
            current_orientation = Vector3(x=self.stable_orientation.x,
                                        y=self.stable_orientation.y,
                                        z=self.stable_orientation.z)
            current_velocity = Vector3(x=self.stable_velocity.x,
                                     y=self.stable_velocity.y,
                                     z=self.stable_velocity.z)
            current_angular_rate = Vector3(x=self.stable_angular_rate.x,
                                         y=self.stable_angular_rate.y,
                                         z=self.stable_angular_rate.z)
            
            # Check for a reset first
            if self.check_and_perform_reset():
                # A reset was performed, skip to the next iteration
                continue
            
            # Randomize fields based on current period
            if period_index == 0:
                # Vary position.z and orientation.z
                center_z = self.stable_position.z
                limit_z = min(self.pos_z_range/2, self.max_pos_z_range/2)
                current_position.z = self.biased_random(
                    self.prev_position_z,
                    center_z - limit_z,
                    center_z + limit_z
                )
                self.prev_position_z = current_position.z
                
                center_ori_z = self.stable_orientation.z
                limit_ori_z = min(self.ori_z_range/2, self.max_ori_z_range/2)
                current_orientation.z = self.biased_random(
                    self.prev_orientation_z,
                    center_ori_z - limit_ori_z,
                    center_ori_z + limit_ori_z
                )
                self.prev_orientation_z = current_orientation.z
                
            elif period_index == 1:
                # Vary orientation.z and velocity.x
                center_ori_z = self.stable_orientation.z
                limit_ori_z = min(self.ori_z_range/2, self.max_ori_z_range/2)
                current_orientation.z = self.biased_random(
                    self.prev_orientation_z,
                    center_ori_z - limit_ori_z,
                    center_ori_z + limit_ori_z
                )
                self.prev_orientation_z = current_orientation.z
                
                center_vel_x = self.stable_velocity.x
                limit_vel_x = min(self.vel_x_range/2, self.max_vel_x_range/2)
                current_velocity.x = self.biased_random(
                    self.prev_velocity_x,
                    center_vel_x - limit_vel_x,
                    center_vel_x + limit_vel_x
                )
                self.prev_velocity_x = current_velocity.x
                
            else:  # period_index == 2
                # Vary position.z and velocity.x
                center_z = self.stable_position.z
                limit_z = min(self.pos_z_range/2, self.max_pos_z_range/2)
                current_position.z = self.biased_random(
                    self.prev_position_z,
                    center_z - limit_z,
                    center_z + limit_z
                )
                self.prev_position_z = current_position.z
                
                center_vel_x = self.stable_velocity.x
                limit_vel_x = min(self.vel_x_range/2, self.max_vel_x_range/2)
                current_velocity.x = self.biased_random(
                    self.prev_velocity_x,
                    center_vel_x - limit_vel_x,
                    center_vel_x + limit_vel_x
                )
                self.prev_velocity_x = current_velocity.x
            
            # Add exploration noise
            if self.exploration_factor > 0.01:
                pos_noise = self.pos_noise.sample() * self.exploration_factor
                ori_noise = self.ori_noise.sample() * self.exploration_factor
                vel_noise = self.vel_noise.sample() * self.exploration_factor
                
                # Apply noise
                current_position.x += pos_noise[0]
                current_position.y += pos_noise[1]
                current_position.z += pos_noise[2]
                
                current_orientation.x += ori_noise[0]
                current_orientation.y += ori_noise[1]
                current_orientation.z += ori_noise[2]
                
                current_velocity.x += vel_noise[0]
                current_velocity.y += vel_noise[1]
                current_velocity.z += vel_noise[2]

            # Publish these values for the duration
            self.publish_values(
                current_position,
                current_orientation,
                current_velocity,
                current_angular_rate,
                random_duration
            )

            # Move to next pattern
            period_index = (period_index + 1) % 3

            # Expand random ranges progressively
            self.pos_z_range = min(self.pos_z_range * self.range_expansion_rate, self.max_pos_z_range)
            self.ori_z_range = min(self.ori_z_range * self.range_expansion_rate, self.max_ori_z_range)
            self.vel_x_range = min(self.vel_x_range * self.range_expansion_rate, self.max_vel_x_range)
            
            # Decay exploration over time
            self.exploration_factor *= self.exploration_decay

def main(args=None):
    rclpy.init(args=args)
    node = CustomSetPointPublisher()
    
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()