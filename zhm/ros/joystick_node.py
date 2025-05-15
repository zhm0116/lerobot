#!/usr/bin/env python3
# code by LinCC111 Boxjod 2025.1.13 Box2AI-Robotics copyright 盒桥智能 版权所有

import os
import mujoco
import mujoco.viewer
import numpy as np
import time
import pygame
import threading
import rospy
from std_msgs.msg import Float64MultiArray
from lerobot_kinematics import lerobot_IK, lerobot_FK, get_robot

np.set_printoptions(linewidth=200)
os.environ["MUJOCO_GL"] = "egl"

JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
xml_path = "/home/zhm/lerobot/zhm/3_ik/scene.xml"

mjmodel = mujoco.MjModel.from_xml_path(xml_path)
qpos_indices = np.array([mjmodel.jnt_qposadr[mjmodel.joint(name).id] for name in JOINT_NAMES])
mjdata = mujoco.MjData(mjmodel)

# Joystick setup
pygame.init()
pygame.joystick.init()
if pygame.joystick.get_count() == 0:
    raise RuntimeError("No joystick detected")
joystick = pygame.joystick.Joystick(0)
joystick.init()
print(f"Using joystick: {joystick.get_name()}")

# Constants
JOINT_INCREMENT = 0.005
POSITION_INCREMENT = 0.0008
DEADZONE = 0.1

# Robot setup
robot = get_robot('so100')
control_qlimit = [[-1.375, -3.1, -0.0, -1.375, -1.5708, -0.1508],
                  [ 1.375,  0.0,  3.1,  1.475,  3.1,     1.5]]
control_glimit = [[0.125, -0.4,  0.046, -3.1, -0.75, -1.5],
                  [0.340,  0.4,  0.23,   2.0,  1.57,  1.5]]

init_qpos = np.array([0.0, -3.1415926, 3.1415926, 0.0, -1.57, -0.157])
target_qpos = init_qpos.copy()
init_gpos = lerobot_FK(init_qpos[1:5], robot=robot)
target_gpos = init_gpos.copy()

target_gpos_last = init_gpos.copy()
target_qpos_last = init_qpos.copy()

lock = threading.Lock()

# ROS
pub_gpos = rospy.Publisher('/target_gpos', Float64MultiArray, queue_size=10)
pub_qpos = rospy.Publisher('/target_qpos_deg', Float64MultiArray, queue_size=10)
rospy.init_node('joystick_control_node')
rate = rospy.Rate(30)

try:
    with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
        start = time.time()
        while not rospy.is_shutdown() and viewer.is_running() and time.time() - start < 1000:
            step_start = time.time()
            pygame.event.pump()

            axis_0 = joystick.get_axis(0)  # left stick X: left/right → qpos[0]
            axis_1 = joystick.get_axis(1)  # left stick Y: fwd/back  → gpos[0]
            axis_4 = joystick.get_axis(4)  # right stick Y: up/down  → gpos[2]
            axis_5 = joystick.get_axis(5)  # right stick Y: up/down  → gpos[2]
            button_0 = joystick.get_button(0)
            button_3 = joystick.get_button(3)
            button_4 = joystick.get_button(4)
            button_5 = joystick.get_button(5)

            axis_0 = 0 if abs(axis_0) < DEADZONE else axis_0
            axis_1 = 0 if abs(axis_1) < DEADZONE else axis_1
            axis_4 = 0 if abs(axis_4) < DEADZONE else axis_4
            axis_5 = 0 if abs(axis_5) < DEADZONE else axis_5

            button_0 = 0 if abs(button_0) < DEADZONE else button_0
            button_3 = 0 if abs(button_3) < DEADZONE else button_3
            button_4 = 0 if abs(button_4) < DEADZONE else button_4
            button_5 = 0 if abs(button_5) < DEADZONE else button_5

            with lock:
                # axis 0 → base rotation
                if axis_0 > 0 and target_qpos[0] > control_qlimit[0][0]:
                    target_qpos[0] += JOINT_INCREMENT * (-axis_0)
                elif axis_0 < 0 and target_qpos[0] < control_qlimit[1][0]:
                    target_qpos[0] += JOINT_INCREMENT * (-axis_0)

                # axis 1 → gpos x
                if axis_1 < 0 and target_gpos[0] < control_glimit[1][0]:
                    target_gpos[0] += POSITION_INCREMENT * (-axis_1)
                elif axis_1 > 0 and target_gpos[0] > control_glimit[0][0]:
                    target_gpos[0] -= POSITION_INCREMENT * axis_1

                # axis 4 → gpos z
                if axis_4 < 0 and target_gpos[2] < control_glimit[1][2]:
                    target_gpos[2] += POSITION_INCREMENT * (-axis_4)
                elif axis_4 > 0 and target_gpos[2] > control_glimit[0][2]:
                    target_gpos[2] -= POSITION_INCREMENT * axis_4
                
                # button 4 → claw rotation
                if button_4 > 0 and target_gpos[3] < control_qlimit[1][3]:
                    target_gpos[3] += POSITION_INCREMENT * button_4
                
                if button_5 > 0 and target_gpos[3] > control_qlimit[0][3]:
                    target_gpos[3] -= POSITION_INCREMENT * button_5
                
                # button 4 → claw rotation
                if button_3 > 0 and target_gpos[4] < control_qlimit[1][4]:
                    target_gpos[4] -= POSITION_INCREMENT * button_3
                
                if button_0 > 0 and target_gpos[4] > control_qlimit[0][4]:
                    target_gpos[4] += POSITION_INCREMENT * button_0

                # axis  → claw
                if axis_5 < 0:
                    target_qpos[5] = 0.0
                else:
                    target_qpos[5] = 0.7
                
            print("target_gpos:", [f"{x:.3f}" for x in target_gpos])
            fd_qpos = mjdata.qpos[qpos_indices][1:5]
            qpos_inv, ik_success = lerobot_IK(fd_qpos, target_gpos, robot=robot)

            if ik_success:
                target_qpos = np.concatenate((target_qpos[0:1], qpos_inv[:4], target_qpos[5:]))
                mjdata.qpos[qpos_indices] = target_qpos
                print("target qpos:", target_qpos * 57.3)

                target_qpos_degree = target_qpos * 57.3 + np.array([0, 180, -180, 0, 90, 0])
                target_qpos_degree *= np.array([-1, 1, -1, 1, -1, 1])

                mujoco.mj_step(mjmodel, mjdata)
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(mjdata.time % 2)
                viewer.sync()

                target_gpos_last = target_gpos.copy()
                target_qpos_last = target_qpos.copy()

                pub_gpos.publish(Float64MultiArray(data=target_gpos.tolist()))
                pub_qpos.publish(Float64MultiArray(data=target_qpos_degree.tolist()))

            else:
                target_gpos = target_gpos_last.copy()

            time_until_next_step = mjmodel.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

except KeyboardInterrupt:
    print("User interrupted.")
finally:
    pygame.quit()
    viewer.close()
