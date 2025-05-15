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

while True:
    follower_pos = robot.leader_arms["main"].read("Present_Position")
    print(type(follower_pos))
    print("Leader Position: ", np.array2string(follower_pos, precision=3, suppress_small=True))
    time.sleep(0.1)
robot.disconnect()
