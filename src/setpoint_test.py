import rclpy,time
from rclpy.node import Node
from std_msgs.msg import Header
from geometry_msgs.msg import Vector3
from mvp_msgs.msg import ControlProcess  # Custom message type

class CustomSetPointPublisher(Node):

    def __init__(self):
        super().__init__('custom_set_point_publisher')

        # Define publisher for the custom topic
        self.set_point_pub = self.create_publisher(ControlProcess, '/race2_auv/controller/process/set_point', 10)

        # Parameters
        self.frame_id_value = "race2_auv/world_ned"
        self.control_mode_value = "4dof"
        self.child_frame_id = "race2_auv/base_link"
        # Reset values
        self.reset_position = Vector3(x=0.0, y=0.0, z=0.0)
        self.reset_orientation = Vector3(x=0.0, y=0.0, z=0.0)
        self.reset_velocity = Vector3(x=0.0, y=0.0, z=0.0)
        self.reset_angular_rate = Vector3(x=0.0, y=0.0, z=0.0)

        # Timing parameters (retrieved from parameters)
        self.reset1_duration = self.declare_parameter("reset1_duration", 2).get_parameter_value().integer_value
        self.set_point_duration = self.declare_parameter("set_point_duration", 300).get_parameter_value().integer_value
        self.reset2_duration = self.declare_parameter("reset2_duration", 2).get_parameter_value().integer_value
        self.rate_hz = self.declare_parameter("rate_hz", 5).get_parameter_value().integer_value

        self.rate = self.create_rate(self.rate_hz)

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
            # self.get_logger().info(f"Published message at {msg.header.stamp}")
            # rclpy.spin_once(self) 
            time.sleep(1.0 / self.rate_hz)  # Sleep to maintain the desired frequency

            
    def run(self):
        # Episode 1: Publish reset values
        self.get_logger().info(f"Publishing reset values for {self.reset1_duration} seconds")
        self.publish_values(self.reset_position, self.reset_orientation, self.reset_velocity, self.reset_angular_rate, self.reset1_duration)

        # Episode 2: Publish desired set points in sequence
        self.get_logger().info("Publishing setpoint 1 for 50 seconds")
        self.publish_values(Vector3(x=0.0, y=0.0, z=5.0), Vector3(x=0.0, y=0.0, z=0.0), Vector3(x=0.25, y=0.0, z=0.0), Vector3(x=0.0, y=0.0, z=0.0), 50)

        self.get_logger().info("Publishing setpoint 2 for 50 seconds")
        self.publish_values(Vector3(x=0.0, y=0.0, z=3.0), Vector3(x=0.0, y=0.0, z=1.57), Vector3(x=0.25, y=0.0, z=0.0), Vector3(x=0.0, y=0.0, z=0.0), 50)

        self.get_logger().info("Publishing setpoint 3 for 50 seconds")
        self.publish_values(Vector3(x=0.0, y=0.0, z=3.0), Vector3(x=0.0, y=0.0, z=3.14), Vector3(x=0.25, y=0.0, z=0.0), Vector3(x=0.0, y=0.0, z=0.0), 50)

        self.get_logger().info("Publishing setpoint 3/1 for 50 seconds")
        self.publish_values(Vector3(x=0.0, y=0.0, z=3.0), Vector3(x=0.0, y=0.0, z=3.14), Vector3(x=-0.25, y=0.0, z=0.0), Vector3(x=0.0, y=0.0, z=0.0), 50)

        self.get_logger().info("Publishing setpoint 3/2 for 100 seconds")
        self.publish_values(Vector3(x=0.0, y=0.0, z=3.0), Vector3(x=0.0, y=0.0, z=3.14), Vector3(x=0.25, y=0.0, z=0.0), Vector3(x=0.0, y=0.0, z=0.0), 100)

        self.get_logger().info("Publishing setpoint 4 for 50 seconds")
        self.publish_values(Vector3(x=0.0, y=0.0, z=5.0), Vector3(x=0.0, y=0.0, z=-1.57), Vector3(x=0.25, y=0.0, z=0.0), Vector3(x=0.0, y=0.0, z=0.0), 50)

        self.get_logger().info("Publishing setpoint 5 for 50 seconds")
        self.publish_values(Vector3(x=0.0, y=0.0, z=5.0), Vector3(x=0.0, y=0.0, z=0.0), Vector3(x=0.25, y=0.0, z=0.0), Vector3(x=0.0, y=0.0, z=0.0), 50)

        # Episode 3: Publish reset values again
        self.get_logger().info(f"Publishing reset values for {self.reset2_duration} seconds")
        self.publish_values(self.reset_position, self.reset_orientation, self.reset_velocity, self.reset_angular_rate, self.reset2_duration)

        self.get_logger().info("All episodes completed. Node will now stop.")


def main(args=None):
    rclpy.init(args=args)

    publisher = CustomSetPointPublisher()
    publisher.run()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
