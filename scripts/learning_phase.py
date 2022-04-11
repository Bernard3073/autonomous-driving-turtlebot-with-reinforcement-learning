#! /usr/bin/env python

import controller
from Lidar import *
from qlearn import *
import rospy
from time import time
from time import sleep
from datetime import datetime
import matplotlib.pyplot as plt
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState
import sys
import os.path
DATA_PATH = '/home/bo/690_ws/src/final_proj/Data'
MODULES_PATH = '/home/bo/690_ws/src/final_proj/scripts'
sys.path.insert(0, MODULES_PATH)


# Episode parameters
MAX_EPISODES = 200
MAX_STEPS_PER_EPISODE = 500
MIN_TIME_BETWEEN_ACTIONS = 0.0

# Learning parameters
ALPHA = 0.2
GAMMA = 0.8

EPSILON_INIT = 0.9
EPSILON_GRAD = 0.96
EPSILON_MIN = 0.05

T_INIT = 25
T_GRAD = 0.95
T_MIN = 0.001


# Initial position
X_START = -2
Y_START = 0
THETA_INIT = 0


# Q table source file
Q_SOURCE_DIR = ''


def initLearning():
    global actions, state_space, Q_table
    actions = createActions()
    state_space = createStateSpace()
    if os.path.exists(DATA_PATH + '/Q_table.csv'):
        Q_table = readQTable(Q_SOURCE_DIR+'/Q_table.csv')
    else:
        Q_table = createQTable(len(state_space), len(actions))


def initParams():
    global T, EPSILON, alpha, gamma
    global ep_steps, ep_reward, episode, steps_per_episode, reward_per_episode, T_per_episode, EPSILON_per_episode
    global ep_reward_arr, reward_max_per_episode, reward_min_per_episode, reward_avg_per_episode
    global crash, t_ep, t_per_episode, t_sim_start, t_step
    global log_sim_info, log_sim_params
    global now_start, now_stop
    global robot_in_pos, first_action_taken


    # Learning parameters
    T = T_INIT
    EPSILON = EPSILON_INIT
    alpha = ALPHA
    gamma = GAMMA

    # Episodes, steps, rewards
    ep_steps = 0
    ep_reward = 0
    episode = 1
    crash = 0
    reward_max_per_episode = np.array([])
    reward_min_per_episode = np.array([])
    reward_avg_per_episode = np.array([])
    ep_reward_arr = np.array([])
    steps_per_episode = np.array([])
    reward_per_episode = np.array([])

    # initial position
    robot_in_pos = False
    first_action_taken = False

    # init time
    t_0 = rospy.Time.now()
    t_start = rospy.Time.now()

    # init timer
    while not (t_start > t_0):
        t_start = rospy.Time.now()

    t_ep = t_start
    t_sim_start = t_start
    t_step = t_start

    T_per_episode = np.array([])
    EPSILON_per_episode = np.array([])
    t_per_episode = np.array([])

def main():
    try:
        global actions, state_space, Q_table
        global T, EPSILON, alpha, gamma
        global ep_steps, ep_reward, episode, steps_per_episode, reward_per_episode, T_per_episode, EPSILON_per_episode
        global ep_reward_arr, reward_max_per_episode, reward_min_per_episode, reward_avg_per_episode
        global crash, t_ep, t_per_episode, t_sim_start, t_step
        global log_sim_info, log_sim_params
        global now_start, now_stop
        global robot_in_pos, first_action_taken

        rospy.init_node('Q_learning_node', anonymous=False)
        rate = rospy.Rate(10)

        setPosPub = rospy.Publisher(
            '/gazebo/set_model_state', ModelState, queue_size=10)
        velPub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        initLearning()
        initParams()
        # sleep(5)

        # main loop
        while not rospy.is_shutdown():
            msgScan = rospy.wait_for_message('/scan', LaserScan)

            # # Secure the minimum time interval between 2 actions
            # step_time = (rospy.Time.now() - t_step).to_sec()
            # if step_time > MIN_TIME_BETWEEN_ACTIONS:
            #     t_step = rospy.Time.now()
            #     if step_time > 2:
            #         text = '\r\nTOO BIG STEP TIME: %.2f s' % step_time
            #         print(text)
            #         log_sim_info.write(text+'\r\n')
                
                
            # End of Learning
            if episode > MAX_EPISODES:

                # Log data to file
                saveQTable(DATA_PATH+'/Q_table.csv', Q_table)
                # np.savetxt(LOG_FILE_DIR+'/StateSpace.csv',
                #            state_space, '%d')

                
                rospy.signal_shutdown('End of learning')
            else:
                
                ep_time = (rospy.Time.now() - t_ep).to_sec()
                # End of en Episode
                if crash or ep_steps >= MAX_STEPS_PER_EPISODE:
                    controller.robotStop(velPub)
                    if crash:
                        # get crash position
                        odomMsg = rospy.wait_for_message('/odom', Odometry)
                        (x_crash, y_crash) = controller.getPosition(odomMsg)
                        theta_crash = degrees(controller.getRotation(odomMsg))

                    t_ep = rospy.Time.now()
                    reward_min = np.min(ep_reward_arr)
                    reward_max = np.max(ep_reward_arr)
                    reward_avg = np.mean(ep_reward_arr)

                    steps_per_episode = np.append(
                        steps_per_episode, ep_steps)
                    reward_per_episode = np.append(
                        reward_per_episode, ep_reward)
                    T_per_episode = np.append(T_per_episode, T)
                    EPSILON_per_episode = np.append(
                        EPSILON_per_episode, EPSILON)
                    t_per_episode = np.append(t_per_episode, ep_time)
                    reward_min_per_episode = np.append(
                        reward_min_per_episode, reward_min)
                    reward_max_per_episode = np.append(
                        reward_max_per_episode, reward_max)
                    reward_avg_per_episode = np.append(
                        reward_avg_per_episode, reward_avg)
                    ep_reward_arr = np.array([])
                    ep_steps = 0
                    ep_reward = 0
                    crash = 0
                    robot_in_pos = False
                    first_action_taken = False
                    if T > T_MIN:
                        T = T_GRAD * T
                    if EPSILON > EPSILON_MIN:
                        EPSILON = EPSILON_GRAD * EPSILON
                    episode = episode + 1
                else:
                    ep_steps = ep_steps + 1
                    # Initial position
                    if not robot_in_pos:
                        controller.robotStop(velPub)
                        ep_steps = ep_steps - 1
                        first_action_taken = False
                        # init pos
                        (x_START, y_START, theta_init) = controller.robotSetPos(
                                setPosPub, X_START, Y_START, THETA_INIT)

                        odomMsg = rospy.wait_for_message('/odom', Odometry)
                        (x, y) = controller.getPosition(odomMsg)
                        theta = degrees(controller.getRotation(odomMsg))
                        # check init pos
                        if abs(x-x_START) < 0.01 and abs(y-y_START) < 0.01 and abs(theta-theta_init) < 1:
                            robot_in_pos = True
                            # sleep(2)
                        else:
                            robot_in_pos = False
                    # First action
                    elif not first_action_taken:
                        (lidar, angles) = lidarScan(msgScan)
                        # 4 state space
                        # (state_ind, x1, x2, x3, x4) = scanDiscretization(state_space, lidar)
                        # # 2 state space
                        (state_ind, x1, x2) = scanDiscretization_twostate(state_space, lidar)
                        
                        crash = checkCrash(lidar)
                        # make sure it's convergence
                        (action, status_strat) = epsilonGreedyExploration(
                                Q_table, state_ind, actions, T)

                        status_rda = controller.robotDoAction(velPub, action)

                        prev_lidar = lidar
                        prev_action = action
                        prev_state_ind = state_ind

                        first_action_taken = True

                        if not (status_strat == 'softMaxSelection => OK' or status_strat == 'epsiloGreedyExploration => OK'):
                            print('\r\n', status_strat, '\r\n')
                            log_sim_info.write('\r\n'+status_strat+'\r\n')

                        if not status_rda == 'robotDoAction => OK':
                            print('\r\n', status_rda, '\r\n')
                            log_sim_info.write('\r\n'+status_rda+'\r\n')

                    # Rest of the algorithm
                    else:
                        (lidar, angles) = lidarScan(msgScan)
                        # 4 state space
                        # (state_ind, x1, x2, x3, x4) = scanDiscretization(state_space, lidar)
                        
                        # 2 state space
                        (state_ind, x1, x2) = scanDiscretization_twostate(state_space, lidar)
                        crash = checkCrash(lidar)

                        (reward, terminal_state) = getReward(
                            action, prev_action, lidar, prev_lidar, crash)

                        (Q_table, status_uqt) = updateQTable(
                            Q_table, prev_state_ind, action, reward, state_ind, alpha, gamma)

                        # apply epsilon greedy exploration
                        (action, status_strat) = epsilonGreedyExploration(
                                Q_table, state_ind, actions, T)

                        status_rda = controller.robotDoAction(velPub, action)

                        if not status_uqt == 'updateQTable => OK':
                            print('\r\n', status_uqt, '\r\n')
                            log_sim_info.write('\r\n'+status_uqt+'\r\n')
                        if not (status_strat == 'softMaxSelection => OK' or status_strat == 'epsiloGreedyExploration => OK'):
                            print('\r\n', status_strat, '\r\n')
                            log_sim_info.write('\r\n'+status_strat+'\r\n')
                        if not status_rda == 'robotDoAction => OK':
                            print('\r\n', status_rda, '\r\n')
                            log_sim_info.write('\r\n'+status_rda+'\r\n')

                        ep_reward = ep_reward + reward
                        ep_reward_arr = np.append(ep_reward_arr, reward)
                        prev_lidar = lidar
                        prev_action = action
                        prev_state_ind = state_ind

    except rospy.ROSInterruptException:
        controller.robotStop(velPub)
        print('End Simulation !!!!!')
        pass

if __name__ == '__main__':
    main()