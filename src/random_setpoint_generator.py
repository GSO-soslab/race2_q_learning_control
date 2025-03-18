#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import random
import time
from std_msgs.msg import Header
from geometry_msgs.msg import Vector3
from mvp_msgs.msg import ControlProcess

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
        self.declare_parameter('random_duration', 200)
        self.declare_parameter('rate_hz', 2.0)
        
        # Parameters
        self.frame_id_value = "race2_auv/world_ned"
        self.child_frame_id = "race2_auv/cg_link"  # Added this based on your code
        self.control_mode_value = "4dof"
        self.rate_hz = self.get_parameter('rate_hz').value
        
        # Stable values to revert to or to use when a field is not being varied
        self.stable_position = Vector3(x=0.0, y=0.0, z=4.0)
        self.stable_orientation = Vector3(x=3.14, y=0.0, z=0.0)
        self.stable_velocity = Vector3(x=0.2, y=0.0, z=0.0)
        self.stable_angular_rate = Vector3(x=0.0, y=0.0, z=0.0)
        
        # Random ranges for fields we want to vary
        self.pos_z_min, self.pos_z_max = 5.0, 5.5
        self.ori_z_min, self.ori_z_max = -0.5, 0.5
        self.vel_x_min, self.vel_x_max = -0.28, 0.28

    def biased_random(prev_value, min_val, max_val, bias_factor=0.5):
        new_value = random.uniform(min_val, max_val)
        return new_value + bias_factor * (new_value - prev_value)
    
    def publish_values(self, position, orientation, velocity, angular_rate, duration):
        """Helper function to publish specified values for a given duration."""
        start_time = self.get_clock().now().seconds_nanoseconds()[0]
        end_time = start_time + duration
        
        while self.get_clock().now().seconds_nanoseconds()[0] < end_time:
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
            
            # Randomize fields based on current period
            if period_index == 0:
                # Vary position.z and orientation.z
                current_position.z = random.uniform(self.pos_z_min, self.pos_z_max)
                current_orientation.z = random.uniform(self.ori_z_min, self.ori_z_max)
            elif period_index == 1:
                # Vary orientation.z and velocity.x
                current_orientation.z = random.uniform(self.ori_z_min, self.ori_z_max)
                current_velocity.x = random.uniform(self.vel_x_min, self.vel_x_max)
            else:  # period_index == 2
                # Vary position.z and velocity.x
                current_position.z = random.uniform(self.pos_z_min, self.pos_z_max)
                current_velocity.x = random.uniform(self.vel_x_min, self.vel_x_max)
            

            # # Randomize fields based on current period
            # if period_index == 0:
            #     # Vary position.z and orientation.z
            #     current_position.z = self.biased_random(self.pos_z_min, self.pos_z_max)
            #     current_orientation.z = self.biased_random(self.ori_z_min, self.ori_z_max)
            # elif period_index == 1:
            #     # Vary orientation.z and velocity.x
            #     current_orientation.z = self.biased_random(self.ori_z_min, self.ori_z_max)  # Corrected here
            #     current_velocity.x = self.biased_random(self.vel_x_min, self.vel_x_max)
            # else:  # period_index == 2
            #     # Vary position.z and velocity.x
            #     current_position.z = self.biased_random(self.pos_z_min, self.pos_z_max)
            #     current_velocity.x = self.biased_random(self.vel_x_min, self.vel_x_max)

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