from lerobot.common.robot_devices.motors.feetech import *
from lerobot.common.robot_devices.robots.configs import So100RobotConfig
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
import numpy as np
import time

so_arm100_configs=So100RobotConfig()
"""
config = FeetechMotorsBusConfig(
    port=so_arm100_configs.follower_arms["main"].port,
    motors=so_arm100_configs.follower_arms["main"].motors
 )
"""
robot = ManipulatorRobot(so_arm100_configs)
robot.connect()
#---------------------current pos-------------------------------
follower_pos = robot.follower_arms["main"].read("Present_Position")
print("follower Position:", np.array2string(follower_pos, precision=3, suppress_small=True))

#---------------------give goal pos------------------------------
time.sleep(2.0)
goal_pos =[0,0,0,0,0,50]
print("goal Position:", goal_pos)
robot.follower_arms["main"].write("Goal_Position", goal_pos)
#-----------------------torque off------------------------------------
time.sleep(2)
robot.follower_arms["main"].write("Torque_Enable", TorqueMode.DISABLED.value)  #关闭

robot.disconnect()
