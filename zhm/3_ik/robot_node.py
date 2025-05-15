# code by LinCC111 Boxjod 2025.1.13 Box2AI-Robotics copyright 盒桥智能 版权所有

import os
import mujoco
import mujoco.viewer
import numpy as np
import time
from lerobot_kinematics import lerobot_IK, lerobot_FK, get_robot
from pynput import keyboard
import threading

from lerobot.common.robot_devices.motors.feetech import *
from lerobot.common.robot_devices.robots.configs import So100RobotConfig
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot

import rospy
from std_msgs.msg import Float64MultiArray

np.set_printoptions(linewidth=200)

# create robot
robot = get_robot('so100')

# Define joint limits
control_qlimit = [[-2.1, -3.1, -0.0, -1.375,  -1.5708, -0.1508], 
                  [ 2.1,  0.0,  3.1,  1.475,   3.1,  1.5]]
control_glimit = [[0.125, -0.4,  0.046, -3.1, -0.75, -1.5], 
                  [0.340,  0.4,  0.23, 2.0,  1.57,  1.5]]

# Initialize target joint positions
init_qpos = np.array([0.0, -3.1415926, 3.1415926, 0.0, -1.57, -0.157])
target_qpos = init_qpos.copy()  # Copy the initial joint positions
init_gpos = lerobot_FK(init_qpos[1:5], robot=robot)
target_gpos = init_gpos.copy()

# Backup for target_gpos in case of invalid IK
target_gpos_last = init_gpos.copy()
target_qpos_last = init_qpos.copy()

#---------------- Callback function----------------------
latest_qpos = init_qpos.copy()
def callback_qpos(msg):
    global latest_qpos
    robot_arm.follower_arms["main"].write("Goal_Position", np.array(msg.data))
#----------------------------------------------------------------------
#        setup so100 arm
#----------------------------------------------------------------------
so_arm100_configs=So100RobotConfig()
robot_arm = ManipulatorRobot(so_arm100_configs)
robot_arm.connect()
#-----------------------------------------------------------------------
#    setup ROS node 
#-----------------------------------------------------------------------
rospy.init_node('robot_node')
rospy.Subscriber('/target_qpos_deg', Float64MultiArray, callback_qpos)
rate = rospy.Rate(60)
try:
    while not rospy.is_shutdown():
        rate.sleep()
        
          

except KeyboardInterrupt:
    print("User interrupted the simulation.")
finally:
    robot_arm.follower_arms["main"].write("Torque_Enable", TorqueMode.DISABLED.value)  #关闭
    viewer.close()
