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

# Set up the MuJoCo render backend
os.environ["MUJOCO_GL"] = "egl"

# Define joint names
JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]

# Absolute path of the XML model
xml_path = "/home/zhm/lerobot/zhm/3_ik/scene.xml"
mjmodel = mujoco.MjModel.from_xml_path(xml_path)
qpos_indices = np.array([mjmodel.jnt_qposadr[mjmodel.joint(name).id] for name in JOINT_NAMES])
mjdata = mujoco.MjData(mjmodel)

# Define joint control increment (in radians)
JOINT_INCREMENT = 0.005  # Can be adjusted as needed
POSITION_INSERMENT = 0.0008

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

# Thread-safe lock
lock = threading.Lock()

# Define key mappings
key_to_joint_increase = {
    'w': 0,  # Move forward
    'a': 1,  # Move right
    'r': 2,  # Move up
    'q': 3,  # Roll +
    'g': 4,  # Pitch +
    'z': 5,  # Gripper +
}

key_to_joint_decrease = {
    's': 0,  # Move backward
    'd': 1,  # Move left
    'f': 2,  # Move down
    'e': 3,  # Roll -
    't': 4,  # Pitch -
    'c': 5,  # Gripper -
}

# Dictionary to track the currently pressed keys and their direction
keys_pressed = {}

# Handle key press events
def on_press(key):
    try:
        k = key.char.lower()  # Convert to lowercase to handle both upper and lower case inputs
        if k in key_to_joint_increase:
            with lock:
                keys_pressed[k] = 0.5  # Increase direction
        elif k in key_to_joint_decrease:
            with lock:
                keys_pressed[k] = -0.5  # Decrease direction
        elif k == "0":
            with lock:
                global target_qpos, target_gpos
                target_qpos = init_qpos.copy()  # Reset to initial position
                target_gpos = init_gpos.copy()  # Reset to initial gripper position
        print(f'{key}')

    except AttributeError:
        pass  # Handle special keys if necessary

# Handle key release events
def on_release(key):
    try:
        k = key.char.lower()
        if k in keys_pressed:
            with lock:
                del keys_pressed[k]
    except AttributeError:
        pass  # Handle special keys if necessary

# Start the keyboard listener in a separate thread
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# Backup for target_gpos in case of invalid IK
target_gpos_last = init_gpos.copy()
target_qpos_last = init_qpos.copy()

#-----------------------------------------------------------------------
#    setup ROS node 
#-----------------------------------------------------------------------
pub_gpos = rospy.Publisher('/target_gpos', Float64MultiArray, queue_size=10)
pub_qpos = rospy.Publisher('/target_qpos_deg', Float64MultiArray, queue_size=10)
rospy.init_node('keyboard_control_node')
rate = rospy.Rate(30)
try:
    # Launch the MuJoCo viewer
    with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
        
        start = time.time()
        while not rospy.is_shutdown() and viewer.is_running() and time.time() - start < 1000:
            step_start = time.time()

            with lock:
                for k, direction in keys_pressed.items():
                    if k in key_to_joint_increase:
                        position_idx = key_to_joint_increase[k]
                        if position_idx == 1 or position_idx == 5:  # Special handling for joint 1 and 5
                            position_idx = 0 if position_idx == 1 else 5
                            if (target_qpos[position_idx]) < control_qlimit[1][position_idx] - JOINT_INCREMENT * direction:
                                target_qpos[position_idx] += JOINT_INCREMENT * direction
                        elif position_idx == 4 or position_idx == 3:
                            if target_gpos[position_idx] <= control_glimit[1][position_idx]:
                                target_gpos[position_idx] += POSITION_INSERMENT * direction * 4
                        else:
                            if target_gpos[position_idx] <= control_glimit[1][position_idx]:
                                target_gpos[position_idx] += POSITION_INSERMENT * direction
                        
                    elif k in key_to_joint_decrease:
                        position_idx = key_to_joint_decrease[k]
                        if position_idx == 1 or position_idx == 5:
                            position_idx = 0 if position_idx == 1 else 5
                            if (target_qpos[position_idx]) > control_qlimit[0][position_idx] - JOINT_INCREMENT * direction:
                                target_qpos[position_idx] += JOINT_INCREMENT * direction
                        elif position_idx == 4 or position_idx == 3:
                            if target_gpos[position_idx] >= control_glimit[0][position_idx]:
                                target_gpos[position_idx] += POSITION_INSERMENT * direction * 4
                        else:
                            if target_gpos[position_idx] >= control_glimit[0][position_idx]:
                                target_gpos[position_idx] += POSITION_INSERMENT * direction
                                
            print("target_gpos:", [f"{x:.3f}" for x in target_gpos])
            fd_qpos = mjdata.qpos[qpos_indices][1:5]
            qpos_inv, ik_success = lerobot_IK(fd_qpos, target_gpos, robot=robot)
            
            if ik_success:  # Check if IK solution is valid
                # target_qpos = np.concatenate((target_qpos[0:1], qpos_inv[1:5], target_qpos[5:]))
                target_qpos = np.concatenate((target_qpos[0:1], qpos_inv[:4], target_qpos[5:]))
                # mjdata.ctrl[qpos_indices] = target_qpos
                mjdata.qpos[qpos_indices] = target_qpos
                print("target qpos:",target_qpos*57.3)
                target_qpos_degree=target_qpos*57.3
                target_qpos_degree+=np.array([0, 180, -180, 0, 90, 0])
                pattern = np.array([-1, 1, -1, 1, -1, 1]) 
                target_qpos_degree=target_qpos_degree*pattern
                mujoco.mj_step(mjmodel, mjdata)
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(mjdata.time % 2)
                viewer.sync()
                
                # backup
                target_gpos_last = target_gpos.copy()  # Save backup of target_gpos
                target_qpos_last = target_qpos.copy()  # Save backup of target_gpos
                # ROS publish
                pub_gpos.publish(Float64MultiArray(data=target_gpos.tolist()))
                pub_qpos.publish(Float64MultiArray(data=target_qpos_degree.tolist()))
  
            else:
                target_gpos = target_gpos_last.copy()  # Restore the last valid target_gpos
            
            # Time management to maintain simulation timestep
            time_until_next_step = mjmodel.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

except KeyboardInterrupt:
    print("User interrupted the simulation.")
finally:
    listener.stop()  # Stop the keyboard listener
    viewer.close()
