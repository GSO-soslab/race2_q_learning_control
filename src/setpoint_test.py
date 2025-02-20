import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from geometry_msgs.msg import Vector3
from mvp_msgs.msg import ControlProcess
from dataclasses import dataclass
from typing import List
import time

@dataclass
class SetPoint:
    """Data class to store setpoint configuration"""
    position: Vector3
    orientation: Vector3
    velocity: Vector3
    angular_rate: Vector3
    duration: int
    description: str

class CustomSetPointPublisher(Node):
    def __init__(self):
        super().__init__('custom_set_point_publisher')
        
        # Define publisher
        self.set_point_pub = self.create_publisher(
            ControlProcess, 
            '/race2_auv/controller/process/set_point', 
            10
        )

        # Configuration
        self.config = {
            'frame_id': "race2_auv/world_ned",
            'control_mode': "4dof",
            'child_frame_id': "race2_auv/cg_link"
        }

        # Get parameters with defaults
        self.params = {
            'reset1_duration': self.declare_parameter("reset1_duration", 2).value,
            'set_point_duration': self.declare_parameter("set_point_duration", 300).value,
            'reset2_duration': self.declare_parameter("reset2_duration", 2).value,
            'rate_hz': self.declare_parameter("rate_hz", 5).value
        }

        # Define mission setpoints
        self.reset_point = SetPoint(
            position=Vector3(x=0.0, y=0.0, z=0.0),
            orientation=Vector3(x=3.14, y=0.0, z=0.0),
            velocity=Vector3(x=0.0, y=0.0, z=0.0),
            angular_rate=Vector3(x=0.0, y=0.0, z=0.0),
            duration=0,  # Duration set during execution
            description="Reset Position"
        )

        self.mission_sequence = self._create_mission_sequence()

    def _create_mission_sequence(self) -> List[SetPoint]:
        """Create the sequence of setpoints for the mission"""
        return [
            SetPoint(
                position=Vector3(x=0.0, y=0.0, z=5.0),
                orientation=Vector3(x=3.14, y=0.0, z=0.0),
                velocity=Vector3(x=0.25, y=0.0, z=0.0),
                angular_rate=Vector3(x=0.0, y=0.0, z=0.0),
                duration=50,
                description="Initial ascent and forward movement"
            ),
            SetPoint(
                position=Vector3(x=0.0, y=0.0, z=3.0),
                orientation=Vector3(x=3.14, y=0.0, z=1.57),
                velocity=Vector3(x=0.25, y=0.0, z=0.0),
                angular_rate=Vector3(x=0.0, y=0.0, z=0.0),
                duration=50,
                description="Descend and turn 90 degrees right"
            ),
            SetPoint(
                position=Vector3(x=0.0, y=0.0, z=3.0),
                orientation=Vector3(x=3.14, y=0.0, z=3.14),
                velocity=Vector3(x=0.25, y=0.0, z=0.0),
                angular_rate=Vector3(x=0.0, y=0.0, z=0.0),
                duration=50,
                description="Turn to 180 degrees"
            ),
            SetPoint(
                position=Vector3(x=0.0, y=0.0, z=3.0),
                orientation=Vector3(x=3.14, y=0.0, z=3.14),
                velocity=Vector3(x=-0.25, y=0.0, z=0.0),
                angular_rate=Vector3(x=0.0, y=0.0, z=0.0),
                duration=50,
                description="Reverse direction"
            ),
            SetPoint(
                position=Vector3(x=0.0, y=0.0, z=3.0),
                orientation=Vector3(x=3.14, y=0.0, z=3.14),
                velocity=Vector3(x=0.25, y=0.0, z=0.0),
                angular_rate=Vector3(x=0.0, y=0.0, z=0.0),
                duration=100,
                description="Forward movement at 180 degrees"
            ),
            SetPoint(
                position=Vector3(x=0.0, y=0.0, z=5.0),
                orientation=Vector3(x=3.14, y=0.0, z=-1.57),
                velocity=Vector3(x=0.25, y=0.0, z=0.0),
                angular_rate=Vector3(x=0.0, y=0.0, z=0.0),
                duration=50,
                description="Ascend and turn 90 degrees left"
            ),
            SetPoint(
                position=Vector3(x=0.0, y=0.0, z=5.0),
                orientation=Vector3(x=3.14, y=0.0, z=0.0),
                velocity=Vector3(x=0.25, y=0.0, z=0.0),
                angular_rate=Vector3(x=0.0, y=0.0, z=0.0),
                duration=50,
                description="Return to initial orientation"
            )
        ]

    def _create_control_message(self, setpoint: SetPoint) -> ControlProcess:
        """Create a control message from a setpoint"""
        msg = ControlProcess()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.config['frame_id']
        msg.child_frame_id = self.config['child_frame_id']
        msg.control_mode = self.config['control_mode']
        
        msg.position = setpoint.position
        msg.orientation = setpoint.orientation
        msg.velocity = setpoint.velocity
        msg.angular_rate = setpoint.angular_rate
        
        return msg

    def publish_setpoint(self, setpoint: SetPoint):
        """Publish a setpoint for the specified duration"""
        self.get_logger().info(f"Publishing {setpoint.description} for {setpoint.duration} seconds")
        
        start_time = self.get_clock().now().seconds_nanoseconds()[0]
        end_time = start_time + setpoint.duration
        
        while self.get_clock().now().seconds_nanoseconds()[0] < end_time:
            msg = self._create_control_message(setpoint)
            self.set_point_pub.publish(msg)
            time.sleep(1.0 / self.params['rate_hz'])

    def run(self):
        """Execute the complete mission sequence"""
        # Initial reset
        self.reset_point.duration = self.params['reset1_duration']
        self.publish_setpoint(self.reset_point)

        # Execute mission sequence
        for setpoint in self.mission_sequence:
            self.publish_setpoint(setpoint)

        # Final reset
        self.reset_point.duration = self.params['reset2_duration']
        self.publish_setpoint(self.reset_point)
        
        self.get_logger().info("Mission completed")

def main(args=None):
    rclpy.init(args=args)
    publisher = CustomSetPointPublisher()
    publisher.run()
    rclpy.shutdown()

if __name__ == '__main__':
    main()