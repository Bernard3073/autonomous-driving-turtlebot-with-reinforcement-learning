#! /usr/bin/env python

import rospy
from time import time
from time import sleep
from datetime import datetime
# import matplotlib.pyplot as plt

import sys
DATA_PATH = '/home/bo/690_ws/src/final_proj/Data'
MODULES_PATH = '/home/bo/690_ws/src/final_proj/scripts'
sys.path.insert(0, MODULES_PATH)

# from Qlearning import *
from qlearn import *
from Lidar import *
from controller import *
# from Control import *


# Action parameter
MIN_TIME_BETWEEN_ACTIONS = 0.0

# Initial & Goal position
X_INIT = -2.0
Y_INIT = 0.0
THETA_INIT = 0.0
X_GOAL = 2.0
Y_GOAL = 0.0
THETA_GOAL = 0

setPosPub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size = 10)
velPub = rospy.Publisher('/cmd_vel', Twist, queue_size = 10)

def main():
    rospy.init_node('main', anonymous = False)

    actions = createActions()
    state_space = createStateSpace()
    Q_table = readQTable(DATA_PATH + '/Q_table.csv')
    print('Initial Q-table:')
    print(Q_table)

    # Init time
    t_0 = rospy.Time.now()
    t_start = rospy.Time.now()

    # init timer
    while not (t_start > t_0):
        t_start = rospy.Time.now()

    t_step = t_start
    count = 0

    # robot in initial position
    robot_in_pos = False

    # because of the video recording
    sleep(1)

    # main loop
    while not rospy.is_shutdown():
        msgScan = rospy.wait_for_message('/scan', LaserScan)
        odomMsg = rospy.wait_for_message('/odom', Odometry)

        # Secure the minimum time interval between 2 actions
        step_time = (rospy.Time.now() - t_step).to_sec()

        if step_time > MIN_TIME_BETWEEN_ACTIONS:
            t_step = rospy.Time.now()

            if not robot_in_pos:
                robotStop(velPub)
                ( x_init , y_init , theta_init ) = robotSetPos(setPosPub, X_INIT, Y_INIT, THETA_INIT)
                # check init pos
                odomMsg = rospy.wait_for_message('/odom', Odometry)
                ( x , y ) = getPosition(odomMsg)
                theta = degrees(getRotation(odomMsg))
                print(theta, theta_init)
                if abs(x-x_init) < 0.05 and abs(y-y_init) < 0.05 and abs(theta-theta_init) < 2:
                    robot_in_pos = True
                    print('\r\nInitial position:')
                    print('x = %.2f [m]' % x)
                    print('y = %.2f [m]' % y)
                    print('theta = %.2f [degrees]' % theta)
                    print('')
                    sleep(1)
                else:
                    robot_in_pos = False
            else:
                count += 1
                text = '\r\nStep %d' % (count)

                # Get robot position and orientation
                ( x , y ) = getPosition(odomMsg)
                theta = getRotation(odomMsg)

                # check if the robot reaches the goal
                if np.sqrt((X_GOAL - x)**2  + (Y_GOAL - y)**2) <= GOAL_DIST_THRESHOLD:
                    robotStop(velPub)
                    rospy.signal_shutdown('End of testing!')
                    text = text + '\r\n\r\nGoal position reached! End of simulation!'
                
                # Get lidar scan
                lidar = lidarScan(msgScan)
                state_ind = scanDiscretization_twostate(state_space, lidar)

                # Check for objects nearby
                crash = checkCrash(lidar)


                # Stop the simulation
                if crash:
                    robotStop(velPub)
                    rospy.signal_shutdown('End of testing!')
                    text = text + ' ==> Crash! End of simulation!'
                # Q-learning algorithm
                else:
                    # Epsilon-greedy policy
                    action = getBestAction(Q_table, state_ind, actions)

                    robotDoAction(velPub, action)

                print(text)

if __name__ == '__main__':

    try:
        main()

    except rospy.ROSInterruptException:
        robotStop(velPub)
        print('Simulation terminated!')
        pass