#!/usr/bin/env python
import rospy
import random
import time
from std_msgs.msg import Header
from geometry_msgs.msg import Vector3
from mvp_msgs.msg import ControlProcess  # Custom message type


def publisher():
    rospy.init_node('custom_set_point_publisher', anonymous=True)

    # Define publisher for the custom topic
    set_point_pub = rospy.Publisher('/race2/controller/process/set_point', ControlProcess, queue_size=10)

    # Parameters
    frame_id_value = "race2/world_ned"
    control_mode_value = "hold_dof"

    # Stable values to revert to or to use when a field is not being varied
    stable_position = Vector3(0.0, 0.0, 4.0)  
    stable_orientation = Vector3(0.0, 0.0, 0.0) 
    stable_velocity = Vector3(0.2, 0.0, 0.0)   
    stable_angular_rate = Vector3(0.0, 0.0, 0.0)

    # Random ranges for fields we want to vary:
    pos_z_min, pos_z_max = 4.0, 5.5
    ori_z_min, ori_z_max = -1.5, 1.5
    vel_x_min, vel_x_max = -0.28, 0.28

    # Time parameters
    random_duration = rospy.get_param("~random_duration",60)
    rate_hz = rospy.get_param("~rate_hz", 10.0)
    rate = rospy.Rate(rate_hz)

    # Pattern definition:
    # 0: vary position.z & orientation.z, keep velocity stable
    # 1: vary orientation.z & velocity.x, keep position stable
    # 2: vary position.z & velocity.x, keep orientation stable
    period_index = 0

    while not rospy.is_shutdown():
        # Start of a new period
        period_start_time = rospy.Time.now().to_sec()

        # Set base values (start from stable)
        current_position = Vector3(stable_position.x, stable_position.y, stable_position.z)
        current_orientation = Vector3(stable_orientation.x, stable_orientation.y, stable_orientation.z)
        current_velocity = Vector3(stable_velocity.x, stable_velocity.y, stable_velocity.z)
        current_angular_rate = Vector3(stable_angular_rate.x, stable_angular_rate.y, stable_angular_rate.z)

        # Randomize the two fields for this period ONCE
        if period_index == 0:
            # Vary position.z and orientation.z
            current_position.z = random.uniform(pos_z_min, pos_z_max)
            current_orientation.z = random.uniform(ori_z_min, ori_z_max)
            # velocity stays stable
        elif period_index == 1:
            # Vary orientation.z and velocity.x
            current_orientation.z = random.uniform(ori_z_min, ori_z_max)
            current_velocity.x = random.uniform(vel_x_min, vel_x_max)
            # position stays stable
        else:
            # period_index == 2
            # Vary position.z and velocity.x
            current_position.z = random.uniform(pos_z_min, pos_z_max)
            current_velocity.x = random.uniform(vel_x_min, vel_x_max)
            # orientation stays stable

        # Now publish the SAME values for the entire random_duration
        while not rospy.is_shutdown():
            current_time = rospy.Time.now().to_sec()
            elapsed = current_time - period_start_time

            if elapsed > random_duration:
                # Time for this period is up
                break

            # Create the message
            msg = ControlProcess()
            msg.header = Header()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = frame_id_value
            msg.control_mode = control_mode_value

            # Assign the previously chosen random (or stable) values
            msg.position = current_position
            msg.orientation = current_orientation
            msg.velocity = current_velocity
            msg.angular_rate = current_angular_rate

            # Publish the message
            set_point_pub.publish(msg)

            rate.sleep()

        # Move to the next pattern after one period
        period_index = (period_index + 1) % 3


if __name__ == '__main__':
    try:
        publisher()
    except rospy.ROSInterruptException:
        pass
