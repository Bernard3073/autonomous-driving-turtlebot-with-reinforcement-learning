#! /usr/bin/env python

import numpy as np
from math import *
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan

MAX_LIDAR_DISTANCE = 1.0
COLLISION_DISTANCE = 0.12 # LaserScan.range_min = 0.1199999
NEARBY_DISTANCE = 0.45

ZONE_0_LENGTH = 0.35
ZONE_1_LENGTH = 0.7

ANGLE_MAX = 360 - 1
ANGLE_MIN = 1 - 1
HORIZON_WIDTH = 75

GOAL_DIST_THRESHOLD = 0.5

# Convert LaserScan msg to array
def lidarScan(msgScan):
    distances = np.array([])
    angles = np.array([])

    for i in range(len(msgScan.ranges)):
        angle = degrees(i * msgScan.angle_increment)
        if ( msgScan.ranges[i] > MAX_LIDAR_DISTANCE ):
            distance = MAX_LIDAR_DISTANCE
        elif ( msgScan.ranges[i] < msgScan.range_min ):
            distance = msgScan.range_min
            # For real robot - protection
            if msgScan.ranges[i] < 0.01:
                distance = MAX_LIDAR_DISTANCE
        else:
            distance = msgScan.ranges[i]

        distances = np.append(distances, distance)
        angles = np.append(angles, angle)

    # distances in [m], angles in [degrees]
    return ( distances, angles )

# Discretization using 2 state space variables of lidar scan
def scanDiscretization_twostate(state_space, lidar):
    x1 = 2 # Left zone (no obstacle detected)
    x2 = 2 # Right zone (no obstacle detected)

    # Find the left side lidar values of the vehicle
    lidar_left = min(lidar[(ANGLE_MIN):(ANGLE_MIN + HORIZON_WIDTH)])
    if ZONE_1_LENGTH > lidar_left > ZONE_0_LENGTH:
        x1 = 1 # zone 1
    elif lidar_left <= ZONE_0_LENGTH:
        x1 = 0 # zone 0

    # Find the right side lidar values of the vehicle
    lidar_right = min(lidar[(ANGLE_MAX - HORIZON_WIDTH):(ANGLE_MAX)])
    if ZONE_1_LENGTH > lidar_right > ZONE_0_LENGTH:
        x2 = 1 # zone 1
    elif lidar_right <= ZONE_0_LENGTH:
        x2 = 0 # zone 0


    # Find the state space index of (x1,x2) in Q table
    print(state_space)
    ss = np.where(np.all(state_space == np.array([x1,x2]), axis = 1))
    state_ind = int(ss[0])

    return ( state_ind, x1, x2)

# Check - crash
def checkCrash(lidar):
    lidar_horizon = np.concatenate((lidar[(ANGLE_MIN + HORIZON_WIDTH):(ANGLE_MIN):-1],lidar[(ANGLE_MAX):(ANGLE_MAX - HORIZON_WIDTH):-1]))
    W = np.linspace(1.2, 1, len(lidar_horizon) // 2)
    W = np.append(W, np.linspace(1, 1.2, len(lidar_horizon) // 2))
    if np.min( W * lidar_horizon ) < COLLISION_DISTANCE:
        return True
    else:
        return False

# Check - object nearby
def checkObjectNearby(lidar):
    lidar_horizon = np.concatenate((lidar[(ANGLE_MIN + HORIZON_WIDTH):(ANGLE_MIN):-1],lidar[(ANGLE_MAX):(ANGLE_MAX - HORIZON_WIDTH):-1]))
    W = np.linspace(1.4, 1, len(lidar_horizon) // 2)
    W = np.append(W, np.linspace(1, 1.4, len(lidar_horizon) // 2))
    if np.min( W * lidar_horizon ) < NEARBY_DISTANCE:
        return True
    else:
        return False

# Check - goal near
def checkGoalNear(x, y, x_goal, y_goal):
    ro = sqrt( pow( ( x_goal - x ) , 2 ) + pow( ( y_goal - y ) , 2) )
    if ro < GOAL_DIST_THRESHOLD:
        return True
    else:
        return False
