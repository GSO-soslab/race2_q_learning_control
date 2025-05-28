#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import random
import time
from std_msgs.msg import Header
from geometry_msgs.msg import Vector3
from mvp_msgs.msg import ControlProcess

class SimpleSetPointPublisher(Node):
    def __init__(self):
        super().__init__('simple_set_point_publisher')
        # Parameters
        self.declare_parameter('rate_hz', 5.0)
        self.declare_parameter('random_duration', 50.0)
        self.declare_parameter('pos_z_range', [1.0, 8.0])
        self.declare_parameter('ori_z_range', [-3.14, 3.14])
        self.declare_parameter('vel_x_range', [-0.4, 0.35])
        
        self.rate_hz = self.get_parameter('rate_hz').value
        self.random_duration = self.get_parameter('random_duration').value
        self.pos_z_min, self.pos_z_max = self.get_parameter('pos_z_range').value
        self.ori_z_min, self.ori_z_max = self.get_parameter('ori_z_range').value
        self.vel_x_min, self.vel_x_max = self.get_parameter('vel_x_range').value
        
        self.publisher = self.create_publisher(
            ControlProcess,
            '/race2_auv/controller/process/set_point',
            10
        )
        
        self.frame_id = "race2_auv/world_ned"
        self.child_frame_id = "race2_auv/cg_link"
        self.control_mode = "4dof"
        
        # Generate a random setpoint
        self.current_setpoint = self.generate_random_setpoint()
        
        # Create a timer for publishing the same setpoint at the specified rate
        self.publish_timer = self.create_timer(1.0 / self.rate_hz, self.publish_callback)
        
        # Create a timer to generate a new setpoint after random_duration seconds
        self.create_timer(self.random_duration, self.update_setpoint_callback)
        
        self.get_logger().info("SimpleSetPointPublisher initialized.")
    
    def generate_random_setpoint(self):
        msg = ControlProcess()
        msg.header = Header()
        msg.header.frame_id = self.frame_id
        msg.child_frame_id = self.child_frame_id
        msg.control_mode = self.control_mode
        
        msg.position = Vector3(x=0.0, y=0.0, z=random.uniform(self.pos_z_min, self.pos_z_max))
        msg.orientation = Vector3(x=3.14, y=0.0, z=random.uniform(self.ori_z_min, self.ori_z_max))
        msg.velocity = Vector3(x=random.uniform(self.vel_x_min, self.vel_x_max), y=0.0, z=0.0)
        msg.angular_rate = Vector3(x=0.0, y=0.0, z=0.0)
        
        return msg
    
    def update_setpoint_callback(self):
        self.current_setpoint = self.generate_random_setpoint()
        self.get_logger().info(f"Generated new setpoint: pos_z={self.current_setpoint.position.z:.2f}, "
                               f"ori_z={self.current_setpoint.orientation.z:.2f}, "
                               f"vel_x={self.current_setpoint.velocity.x:.2f}")
    
    def publish_callback(self):
        # Update the timestamp to current time
        self.current_setpoint.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(self.current_setpoint)
        self.get_logger().debug("Published current setpoint")

def main(args=None):
    rclpy.init(args=args)
    node = SimpleSetPointPublisher()
    
    try:
        # node.get_logger().info(f"Node will use the same setpoint for {node.random_duration} seconds before generating a new one")
        node.get_logger().info(f"Publishing at {node.rate_hz} Hz")
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node stopped by keyboard interrupt")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()