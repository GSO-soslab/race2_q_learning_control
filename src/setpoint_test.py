#!/usr/bin/env python

import rospy
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

    # # Desired set points
    # position_value = Vector3(0.0, 0.0, 5.0) # x, y, z
    # orientation_value = Vector3(0.0, 0.0, -0.6) # phi, theta, psi
    # velocity_value = Vector3(0.25, 0.0, 0.0) # u, v, w
    # angular_rate_value = Vector3(0.0, 0.0, 0.0) # p, q, r

    # # desired setpoint set1
    # position_value = Vector3(0.0, 0.0, 5.0) # x, y, z
    # orientation_value = Vector3(0.0, 0.0, 0.0) # phi, theta, psi
    # velocity_value = Vector3(0.25, 0.0, 0.0) # u, v, w
    # angular_rate_value = Vector3(0.0, 0.0, 0.0) # p, q, r

    # # desired setpoint set2
    # position_value = Vector3(0.0, 0.0, 3.0) # x, y, z
    # orientation_value = Vector3(0.0, 0.0, -1.57) # phi, theta, psi
    # velocity_value = Vector3(0.25, 0.0, 0.0) # u, v, w
    # angular_rate_value = Vector3(0.0, 0.0, 0.0) # p, q, r

    # # desired setpoint set3
    # position_value = Vector3(0.0, 0.0, 3.0) # x, y, z
    # orientation_value = Vector3(0.0, 0.0, 0.0) # phi, theta, psi
    # velocity_value = Vector3(-0.25, 0.0, 0.0) # u, v, w
    # angular_rate_value = Vector3(0.0, 0.0, 0.0) # p, q, r

    # # desired setpoint set4
    # position_value = Vector3(0.0, 0.0, 5.0) # x, y, z
    # orientation_value = Vector3(0.0, 0.0, 1.57) # phi, theta, psi
    # velocity_value = Vector3(-0.25, 0.0, 0.0) # u, v, w
    # angular_rate_value = Vector3(0.0, 0.0, 0.0) # p, q, r

    # # desired setpoint set5
    # position_value = Vector3(0.0, 0.0, 5.0) # x, y, z
    # orientation_value = Vector3(0.0, 0.0, 0.0) # phi, theta, psi
    # velocity_value = Vector3(0.25, 0.0, 0.0) # u, v, w
    # angular_rate_value = Vector3(0.0, 0.0, 0.0) # p, q, r

    # Reset values
    reset_position = Vector3(0.0, 0.0, 0.0)
    reset_orientation = Vector3(0.0, 0.0, 0.0)
    reset_velocity = Vector3(0.0, 0.0, 0.0)
    reset_angular_rate = Vector3(0.0, 0.0, 0.0)

    # Timing parameters
    reset1_duration = rospy.get_param("~reset1_duration", 10)  # Default 10 seconds
    set_point_duration = rospy.get_param("~set_point_duration", 300)  # Default 300 seconds
    reset2_duration = rospy.get_param("~reset2_duration", 10)  # Default 10 seconds
    rate_hz = rospy.get_param("~rate_hz", 1)  # Default 10 Hz

    rate = rospy.Rate(rate_hz)

    def publish_values(position, orientation, velocity, angular_rate, duration):
        """Helper function to publish specified values for a given duration."""
        for _ in range(int(duration * rate_hz)):
            msg = ControlProcess()
            msg.header = Header()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = frame_id_value

            msg.control_mode = control_mode_value
            msg.position = position
            msg.orientation = orientation
            msg.velocity = velocity
            msg.angular_rate = angular_rate

            set_point_pub.publish(msg)
            rate.sleep()

    # Episode 1: Publish reset values
    rospy.loginfo("Publishing reset values for %d seconds", reset1_duration)
    publish_values(reset_position, reset_orientation, reset_velocity, reset_angular_rate, reset1_duration)

    # # Episode 2: Publish desired set points
    # rospy.loginfo("Publishing set point values for %d seconds", set_point_duration)
    # publish_values(position_value, orientation_value, velocity_value, angular_rate_value, set_point_duration)

    # Episode 2: Publish desired set points in sequence
    rospy.loginfo("Publishing setpoint 1 for 100 seconds")
    ##############################
    #     _  _
    #    |    | 
    ##############################
    publish_values(Vector3(0.0, 0.0, 5.0), Vector3(0.0, 0.0, 0.0), Vector3(0.25, 0.0, 0.0), Vector3(0.0, 0.0, 0.0), 30)




    rospy.loginfo("Publishing setpoint 2 for 30 seconds")
    ##############################
    # __
    #   |
    # __|  
    ##############################
    publish_values(Vector3(0.0, 0.0, 3.0), Vector3(0.0, 0.0, 1.57), Vector3(0.25, 0.0, 0.0), Vector3(0.0, 0.0, 0.0), 30)




    rospy.loginfo("Publishing setpoint 3 for 30 seconds")
    ##############################
    # |    |
    #  _  _
    ##############################
    publish_values(Vector3(0.0, 0.0, 3.0), Vector3(0.0, 0.0, 3.14), Vector3(0.25, 0.0, 0.0), Vector3(0.0, 0.0, 0.0), 30)




    rospy.loginfo("Publishing setpoint 3/1 for 30 seconds")
    
    publish_values(Vector3(0.0, 0.0, 3.0), Vector3(0.0, 0.0, 3.14), Vector3(-0.25, 0.0, 0.0), Vector3(0.0, 0.0, 0.0), 30)


    rospy.loginfo("Publishing setpoint 3/2 for 30 seconds")
    
    publish_values(Vector3(0.0, 0.0, 3.0), Vector3(0.0, 0.0, 3.14), Vector3(0.25, 0.0, 0.0), Vector3(0.0, 0.0, 0.0), 30)


    rospy.loginfo("Publishing setpoint 4 for 30 seconds")
    ##############################
    #   __
    # |  
    # | __
    ##############################
    publish_values(Vector3(0.0, 0.0, 5.0), Vector3(0.0, 0.0, -1.57), Vector3(0.25, 0.0, 0.0), Vector3(0.0, 0.0, 0.0), 30)



    rospy.loginfo("Publishing setpoint 5 for 50 seconds")
    ##############################
    #  _  _
    # |    | 
    ##############################
    publish_values(Vector3(0.0, 0.0, 5.0), Vector3(0.0, 0.0, 0.0), Vector3(0.25, 0.0, 0.0), Vector3(0.0, 0.0, 0.0), 50)



    # Episode 3: Publish reset values again
    rospy.loginfo("Publishing reset values for %d seconds", reset2_duration)
    publish_values(reset_position, reset_orientation, reset_velocity, reset_angular_rate, reset2_duration)

    rospy.loginfo("All episodes completed. Node will now stop.")


if __name__ == '__main__':
    try:
        publisher()
    except rospy.ROSInterruptException:
        pass
