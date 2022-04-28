#!/usr/bin/env python

import rospy
from time import time
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelState
import numpy as np
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from math import radians, degrees

# Q-learning speed parameters
CONST_LINEAR_SPEED_FORWARD = 0.1
CONST_ANGULAR_SPEED_FORWARD = 0.0
CONST_LINEAR_SPEED_TURN = 0.1
CONST_ANGULAR_SPEED_TURN = 0.5

# Feedback control parameters
K_RO = 2
K_ALPHA = 15
K_BETA = -3
V_CONST = 0.1 # [m/s]

# Goal reaching threshold
GOAL_DIST_THRESHOLD = 0.3 # [m]
GOAL_ANGLE_THRESHOLD = 30 # [degrees]

k_h_gain = 0.3
k_v_gain = 0.9

# Get theta in [radians]
def getRotation(odomMsg):
    orientation_q = odomMsg.pose.pose.orientation
    orientation_list = [ orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
    (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
    return yaw

# Get (x,y) coordinates in [m]
def getPosition(odomMsg):
    x = odomMsg.pose.pose.position.x
    y = odomMsg.pose.pose.position.y
    return ( x , y)

# Get linear speed in [m/s]
def getLinVel(odomMsg):
    return odomMsg.twist.twist.linear.x

# Get angular speed in [rad/s] - z axis
def getAngVel(odomMsg):
    return odomMsg.twist.twist.angular.z

# Create rosmsg Twist()
def createVelMsg(v,w):
    velMsg = Twist()
    velMsg.linear.x = v
    velMsg.linear.y = 0
    velMsg.linear.z = 0
    velMsg.angular.x = 0
    velMsg.angular.y = 0
    velMsg.angular.z = w
    return velMsg

# Go forward command
def robotGoForward(velPub):
    velMsg = createVelMsg(CONST_LINEAR_SPEED_FORWARD,CONST_ANGULAR_SPEED_FORWARD)
    velPub.publish(velMsg)

# Turn left command
def robotTurnLeft(velPub):
    velMsg = createVelMsg(CONST_LINEAR_SPEED_TURN,+CONST_ANGULAR_SPEED_TURN)
    velPub.publish(velMsg)

# Turn right command
def robotTurnRight(velPub):
    velMsg = createVelMsg(CONST_LINEAR_SPEED_TURN,-CONST_ANGULAR_SPEED_TURN)
    velPub.publish(velMsg)

# Stop command
def robotStop(velPub):
    velMsg = createVelMsg(0.0,0.0)
    velPub.publish(velMsg)

# Set robot position and orientation
def robotSetPos(setPosPub, x, y, theta):
    checkpoint = ModelState()

    checkpoint.model_name = 'turtlebot3_waffle'

    checkpoint.pose.position.x = x
    checkpoint.pose.position.y = y
    checkpoint.pose.position.z = 0.0

    [x_q,y_q,z_q,w_q] = quaternion_from_euler(0.0,0.0,radians(theta))

    checkpoint.pose.orientation.x = x_q
    checkpoint.pose.orientation.y = y_q
    checkpoint.pose.orientation.z = z_q
    checkpoint.pose.orientation.w = w_q

    checkpoint.twist.linear.x = 0.0
    checkpoint.twist.linear.y = 0.0
    checkpoint.twist.linear.z = 0.0

    checkpoint.twist.angular.x = 0.0
    checkpoint.twist.angular.y = 0.0
    checkpoint.twist.angular.z = 0.0

    setPosPub.publish(checkpoint)
    return ( x , y , theta )

# Perform an action
def robotDoAction(velPub, action):
    if action == 0:
        robotGoForward(velPub)
    elif action == 1:
        robotTurnLeft(velPub)
    elif action == 2:
        robotTurnRight(velPub)
    else:
        robotGoForward(velPub)


# Feedback Control Algorithm
# def robotFeedbackControl(velPub, x, y, theta, x_goal, y_goal, theta_goal):
#     # theta goal normalization
#     if theta_goal >= pi:
#         theta_goal_norm = theta_goal - 2 * pi
#     else:
#         theta_goal_norm = theta_goal

#     ro = sqrt( pow( ( x_goal - x ) , 2 ) + pow( ( y_goal - y ) , 2) )
#     lamda = atan2( y_goal - y , x_goal - x )

#     alpha = (lamda -  theta + pi) % (2 * pi) - pi
#     beta = (theta_goal - lamda + pi) % (2 * pi) - pi

#     if ro < GOAL_DIST_THRESHOLD and degrees(abs(theta-theta_goal_norm)) < GOAL_ANGLE_THRESHOLD:
#         status = 'Goal position reached!'
#         v = 0
#         w = 0
#         v_scal = 0
#         w_scal = 0
#     else:
#         status = 'Goal position not reached!'
#         v = K_RO * ro
#         w = K_ALPHA * alpha + K_BETA * beta
#         v_scal = v / abs(v) * V_CONST
#         w_scal = w / abs(v) * V_CONST

#     velMsg = createVelMsg(v_scal, w_scal)
#     velPub.publish(velMsg)

#     return status

# def P_controller(velPub, x_start, y_start, goal_x, goal_y, last_rotation):
#     angle_to_goal = atan2(goal_y - y_start, goal_x - x_start)

#     # the domain of arctan(x) is (-inf, inf)
#     # we would like to restrict the domain to (0, 2pi)
#     if angle_to_goal < -pi / 4 or angle_to_goal > pi / 4:
#         if 0 > goal_y > y_start:
#             angle_to_goal = -2 * pi + angle_to_goal
#         elif 0 <= goal_y < y_start:
#             angle_to_goal = 2 * pi + angle_to_goal
#     if last_rotation > pi - 0.1 and rotation <= 0:
#         rotation = 2 * pi + rotation
#     elif last_rotation < -pi + 0.1 and rotation > 0:
#         rotation = -2 * pi + rotation

#     # proportional control for rotating the robot
#     w = k_v_gain * angle_to_goal - rotation
#     distance_to_goal = sqrt( pow( ( goal_x - x_start) , 2 ) + pow( ( goal_y - y_start) , 2) )
#     if distance_to_goal <= GOAL_DIST_THRESHOLD:
#         status = 'Goal position reached!'
#         v = 0
#         w = 0
#     else:
#         status = 'Goal position not reached!'
#         # proportional control to move the robot forward
#         # We will drive the robot at a maximum speed of 0.1
#         v = min(k_h_gain * distance_to_goal, CONST_LINEAR_SPEED_FORWARD)

#         # set the z angular velocity for positive and negative rotations
#         if w > 0:
#             w = min(w, 1.5)
#         else:
#             w = max(w, -1.5)

#     # update the new rotation for the next loop
#     last_rotation = rotation
#     velMsg = createVelMsg(v, w)
#     velPub.publish(velMsg)
#     return status, last_rotation

# Stability Condition
def check_stability(k_rho, k_alpha, k_beta):
    return k_rho > 0 and k_beta < 0 and k_alpha > k_rho

# Strong Stability Condition
def check_strong_stability(k_rho, k_alpha, k_beta):
    return k_rho > 0 and k_beta < 0 and k_alpha + 5 * k_beta / 3 - 2 * k_rho / np.pi > 0