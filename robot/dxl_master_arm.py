#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ##
# @brief    [record_episode] 
# @author   Daegyu Lim (dglim@robros.co.kr)   

from __future__ import print_function

import os
# from os.path import dirname, join, abspath
import time
import rospy
import struct
from dynamixel_sdk import *  # Dynamixel SDK를 임포트합니다.

from std_msgs.msg import Float32MultiArray, Float32, Bool
from geometry_msgs.msg import PoseStamped, Twist 

import numpy as np
from numpy.linalg import norm, solve
import ctypes
from scipy.spatial.transform import Rotation
from termcolor import colored

import tkinter
import threading

import pinocchio as pin
from pinocchio.utils import *

# CONSTANTS
UNITVEL2DEG = 0.229*360/60 
DEG2RAD = np.pi/180
RAD2DEG = 180/np.pi

# 기본 설정
MY_DXL = 'XL330-M288'  # 모터 모델명
PROTOCOL_VERSION = 2.0  # 사용하는 Dynamixel 프로토콜 버전
BAUDRATE = 2000000  # Dynamixel의 통신 속도
GELLO2ROBOT_RATIO = 2.0 # ROBOT/GELLO kinematics ratio

# DATA
POSITION_CONTROL_MODE = 3
CURRENT_CONTROL_MODE = 0
CURRENT_BASED_POSITION_CONTROL_MODE = 5

TORQUE_ON = 1
TORQUE_OFF = 0

# DXL Communication Address
DXL_OPERATING_MODE_ADDR = 11  # operation mode addresse of model XL330-M288 
DXL_TORQUE_ENABLE_ADDR = 64  # torque enable addresse of model XL330-M288 
DXL_HARDWARE_ERROR_ADDR = 70
DXL_CURRENT_POSITION_ADDR = 132  # current position addresse of model XL330-M288 
DXL_CURRENT_VELOCITY_ADDR = 128  # current velocity addresse of model XL330-M288 
DXL_CURRENT_TORQUE_ADDR = 126  # current Current addresse of model XL330-M288 
DXL_GOAL_TORQUE_ADDR = 102  # goal Current addresse of model XL330-M288 
DXL_GOAL_POSITION_ADDR = 116  # goal Current addresse of model XL330-M288 

# Data Length
LEN_OPERATING_MODE = 1
LEN_HARDWARE_ERROR = 1
LEN_CURRENT_POSITION = 4
LEN_CURRENT_VELOCITY = 4
LEN_CURRENT_TORQUE = 2

LEN_GOAL_TORQUE = 2
LEN_GOAL_POSITION = 4
NUM_JOINTS = 7 # including gripper
NUM_ARM_JOINTS = 6 # only the number of arm joints
        
class dsrMasterArmCore:
    def __init__(self, robot_id='dsr_l', port='/dev/ttyUSB0', hz = 50):
        rospy.on_shutdown(self.shutdown)

        self.tick = 0
        self.hz = hz
        
        ## pinocchio ##
        urdf_model_path = "/home/robros-ai/dg/robros_imitation_learning/robot/asset/urdf/a0509_white.urdf"
        self.model, _, _ = pin.buildModelsFromUrdf(urdf_model_path)
        print(self.model.name)
        self.data  = self.model.createData()
        self.tcpframeId = self.model.getFrameId("end_effector")

        # class input variable
        self.port = port 
        self.robot_id = robot_id  

        # Dynamixel 포트 초기화
        self.portHandler = PortHandler(self.port)
        self.packetHandler = PacketHandler(PROTOCOL_VERSION)

        self.groupSyncHwErrorRead = GroupSyncRead(self.portHandler, self.packetHandler, DXL_HARDWARE_ERROR_ADDR, LEN_HARDWARE_ERROR)

        self.groupSyncPositionRead = GroupSyncRead(self.portHandler, self.packetHandler, DXL_CURRENT_POSITION_ADDR, LEN_CURRENT_POSITION)
        self.groupSyncVelocityRead = GroupSyncRead(self.portHandler, self.packetHandler, DXL_CURRENT_VELOCITY_ADDR, LEN_CURRENT_VELOCITY)
        self.groupSyncCurrentRead = GroupSyncRead(self.portHandler, self.packetHandler, DXL_CURRENT_TORQUE_ADDR, LEN_CURRENT_TORQUE)

        self.groupSyncPositionWrite = GroupSyncWrite(self.portHandler, self.packetHandler, DXL_GOAL_POSITION_ADDR, LEN_GOAL_POSITION)
        self.groupSyncTorqueWrite = GroupSyncWrite(self.portHandler, self.packetHandler, DXL_GOAL_TORQUE_ADDR, LEN_GOAL_TORQUE)

        if self.portHandler.openPort():
            print(f"[{self.robot_id}] Succeeded to open the port")
        else:
            print(f"[{self.robot_id}] Failed to open the port")
            quit()

        if self.portHandler.setBaudRate(BAUDRATE):
            print(f"[{self.robot_id}] Succeeded to change the baudrate")
        else:
            print(f"[{self.robot_id}] Failed to change the baudrate")
            quit()

        for dxl_id in range(1, NUM_JOINTS+1):
            if not self.groupSyncPositionRead.addParam(dxl_id):
                raise RuntimeError(
                    f"Failed to add parameter for Dynamixel with ID {dxl_id}"
                )
            if not self.groupSyncVelocityRead.addParam(dxl_id):
                raise RuntimeError(
                    f"Failed to add parameter for Dynamixel with ID {dxl_id}"
                )
            if not self.groupSyncCurrentRead.addParam(dxl_id):
                raise RuntimeError(
                    f"Failed to add parameter for Dynamixel with ID {dxl_id}"
                )
            if not self.groupSyncHwErrorRead.addParam(dxl_id):
                raise RuntimeError(
                    f"Failed to add parameter for Dynamixel with ID {dxl_id}"
                )

        for dxl_id in range(1, NUM_JOINTS+1):
            # send control mode packet    
            self.packetHandler.write1ByteTxOnly(self.portHandler, dxl_id, DXL_OPERATING_MODE_ADDR, CURRENT_CONTROL_MODE)

        for dxl_id in range(1, NUM_JOINTS+1):
            # TORQUE ON MOTOR
            self.packetHandler.write1ByteTxOnly(self.portHandler, dxl_id, DXL_TORQUE_ENABLE_ADDR, TORQUE_ON)
        

        ## gripper power
        gripper_power = 100
        gripper_power_bytes = gripper_power.to_bytes(2, 'little', signed = True)
        self.groupSyncTorqueWrite.addParam(7, gripper_power_bytes)
        self.groupSyncTorqueWrite.txPacket()


        self.q = np.zeros(NUM_ARM_JOINTS)
        self.q_dot = np.zeros(NUM_ARM_JOINTS)
        self.tau_c = np.zeros(NUM_ARM_JOINTS)

        self.hand_trigger_close = 0

        self.hand_trigger_release_pos = 15
        self.hand_trigger_pull_pos = -24
        
    

        self.rate = rospy.Rate(self.hz)

        ### subscriber ###
        self.dsr_pose_sub = rospy.Subscriber(f'/{self.robot_id}/state/pose', PoseStamped, self.dsr_pose_callback)
        self.dsr_joint_state_sub = rospy.Subscriber(f'/{self.robot_id}/state/joint', Float32MultiArray, self.dsr_joint_state_callback)

        self.dsr_hand_pose_raw = PoseStamped()
        self.dsr_joint_state_raw = Float32MultiArray()
        self.dsr_joint_state_raw.data = [90*DEG2RAD, 0, 90*DEG2RAD, 0, 90*DEG2RAD, 90*DEG2RAD, 0]

        self.dsr_hand_pose = np.zeros(7)
        self.dsr_pose_rcv_tick = -1

        ### publisher ###
        self.pub_hand_pose = rospy.Publisher(f'/{self.robot_id}/teleop/gello_hand_pose', PoseStamped, queue_size=1)
        self.pub_arm_joint = rospy.Publisher(f'/{self.robot_id}/teleop/gello_arm_joints', Float32MultiArray, queue_size=1)
        self.pub_hand_trigger = rospy.Publisher(f'/{self.robot_id}/teleop/gello_hand_trigger', Float32, queue_size=1)

        self.pub_connection_signal = rospy.Publisher(f'/{self.robot_id}/teleop/gello_connection', Bool, queue_size=1)
        self.pub_homepose_signal = rospy.Publisher(f'/{self.robot_id}/teleop/gello_home_pose', Bool, queue_size=1)

        self.gello_hand = PoseStamped()
        self.gello_arm_joints = Float32MultiArray()
        self.gello_hand_tripper = Float32()

        self.connection_signal = Bool()
        self.home_pose_signal = Bool()

        self.connection_signal.data = False
        self.home_pose_signal.data = False

        self.connect_start_tick = -1e6
        self.connect_start_delay = 3.0 #[seconds]
        self.connect_spline_duration = 3.0 #[seconds]
        self.connect_init_pose = np.zeros(7) # xyzquat
        self.connect_spline_high = False
        self.desired_pose = np.zeros(7)

        self.sync_signal = False
        self.sync_start_tick = -1
        self.sync_duration = 3.0
        self.sync_init_q = self.q
        self.desired_q = np.zeros( (NUM_ARM_JOINTS))

        # 현재 각도 읽기
        self.initial_position = [180.0,180.0,180.0,180.0,180.0,0.0,180.0]  # 초기 위치 저장
        self.joint_axis = [1,1,-1,1,-1,1,1]  # 초기 위치 저장

        self.stop_control_loop = False

    def dsr_pose_callback(self, data):
        self.dsr_hand_pose_raw = data
        self.dsr_pose_rcv_tick = self.tick
    
    def dsr_joint_state_callback(self, data):
        self.dsr_joint_state_raw = data

        ## GUI
    def dsrConnect(self, connect_delay = 0.0, connect_spline_duration = 3.0):

        if self.tick < self.dsr_pose_rcv_tick + 5:
            print(f"Connect Button Pressed, connect_delay={connect_delay}, connect_spline_duration={connect_spline_duration}")
            self.connection_signal.data = True

            self.connect_start_tick = self.tick
            self.connect_start_delay = min(connect_delay, 100)
            self.connect_spline_duration = min(connect_spline_duration, 100)

            self.connect_init_pose[0] = self.dsr_hand_pose_raw.pose.position.x
            self.connect_init_pose[1] = self.dsr_hand_pose_raw.pose.position.y
            self.connect_init_pose[2] = self.dsr_hand_pose_raw.pose.position.z

            self.connect_init_pose[3] = self.dsr_hand_pose_raw.pose.orientation.x
            self.connect_init_pose[4] = self.dsr_hand_pose_raw.pose.orientation.y
            self.connect_init_pose[5] = self.dsr_hand_pose_raw.pose.orientation.z
            self.connect_init_pose[6] = self.dsr_hand_pose_raw.pose.orientation.w

            self.desired_pose = self.connect_init_pose.copy()
        else:
             print(colored('Can not connect. DSR pose is not recieved recently!', 'red') )


    def dsrDisconnect(self):
        print("Disconnect")
        self.connection_signal.data = False

    def dsrSync(self):
        print("Syncronize")
        self.sync_signal = True
        self.sync_start_tick = self.tick
        self.sync_init_q = self.q

    def dsrHomePose(self):
        print("DSR HOME POSE")
        self.home_pose_signal.data = True

    def convert_to_angle(self, position):
        MAX_POSITION_VALUE = 4095  # 모터의 최대 위치 값
        angle = (position / MAX_POSITION_VALUE) * 360  # 0도에서 360도 사이의 값으로 변환
        return angle
    
    def degree_to_dxl_command(self, position):
        MAX_POSITION_VALUE = 4095  # 모터의 최대 위치 값
        return int(position/360 * MAX_POSITION_VALUE)
    
    def convert_to_velocity(self, velocity):
        UNITVEL2DEG = 0.229*360/60  # 모터의 최대 위치 값
        velocity_new = velocity * UNITVEL2DEG  # 0도에서 360도 사이의 값으로 변환
        return velocity_new

    def convert_to_current(self, current):
        current_new = current / 1000  # mA to A
        return current_new

    def set_init_q(self, init_q):
        assert len(init_q) == NUM_JOINTS
        self.initial_position = init_q

    def set_joint_axis(self, joint_axis):
        assert len(joint_axis) == NUM_JOINTS
        self.joint_axis = joint_axis


    def cubic(self, time, time_0, time_f, x_0, x_f, x_dot_0, x_dot_f):
        if (time < time_0):
            x_t = x_0
        elif (time > time_f):
            x_t = x_f
        else:
            elapsed_time = time - time_0
            total_time = time_f - time_0
            total_time2 = total_time * total_time
            total_time3 = total_time2 * total_time
            total_x = x_f - x_0

            x_t = x_0 + x_dot_0 * elapsed_time + (3 * total_x / total_time2 - 2 * x_dot_0 / total_time - x_dot_f / total_time) * elapsed_time * elapsed_time + (-2 * total_x / total_time3 + (x_dot_0 + x_dot_f) / total_time2)*elapsed_time * elapsed_time * elapsed_time

        return x_t
            
    def read_data(self):
        position_read_try = 0
        dxl_comm_result = self.groupSyncPositionRead.txRxPacket()

        while dxl_comm_result != COMM_SUCCESS:
            dxl_comm_result = self.groupSyncPositionRead.txRxPacket()
            position_read_try += 1
            if position_read_try>10:
                return -1
            
        # if dxl_comm_result != COMM_SUCCESS:
        #     for DXL_ID in range(1, NUM_JOINTS+1): 
        #         hardware_error_msg = self.packetHandler.read1ByteTxRx(self.portHandler, DXL_ID, DXL_HARDWARE_ERROR_ADDR)
        #         print(f"warning, self.groupSyncPositionRead comm failed: {dxl_comm_result} with HW error {hardware_error_msg} at DXL_ID={DXL_ID}")
        #         return -1
        
        # dxl_comm_result = self.groupSyncVelocityRead.txRxPacket()
        # if dxl_comm_result != COMM_SUCCESS:
        #     print(f"warning, self.groupSyncVelocityRead comm failed: {dxl_comm_result}")
        #     return -1

        # dxl_comm_result = self.groupSyncCurrentRead.txRxPacket()
        # if dxl_comm_result != COMM_SUCCESS:
        #     print(f"warning, self.groupSyncCurrentRead comm failed: {dxl_comm_result}")
        #     return -1

        dxl_comm_result = self.groupSyncHwErrorRead.txRxPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print(f"warning, self.groupSyncHwErrorRead comm failed: {dxl_comm_result}")
            return -1
        return 1


    def main(self):
        while not rospy.is_shutdown():  # 사용자가 종료할 때까지 무한 루프
            # print("\n")
            
            if self.read_data() < 0:
                break

            for DXL_ID in range(1, NUM_JOINTS+1): 
                # Read hardware error message
                # if self.groupSyncHwErrorRead.isAvailable(DXL_ID, DXL_HARDWARE_ERROR_ADDR, LEN_HARDWARE_ERROR):
                #     hardware_error_msg = self.groupSyncHwErrorRead.getData(DXL_ID, DXL_HARDWARE_ERROR_ADDR, LEN_HARDWARE_ERROR)
                #     if hardware_error_msg != 0:
                #         print(f'HARDWARE ERROR at {DXL_ID} with message [{hardware_error_msg}]')
                # else:
                #     raise RuntimeError(
                #         f"Failed to get joint HW error message for Dynamixel with ID {DXL_ID}"
                #     )
                
                # Read Current Joint Position
                if self.groupSyncPositionRead.isAvailable(DXL_ID, DXL_CURRENT_POSITION_ADDR, LEN_CURRENT_POSITION):
                    dxl_present_position = self.groupSyncPositionRead.getData(DXL_ID, DXL_CURRENT_POSITION_ADDR, LEN_CURRENT_POSITION)
                    angle = self.joint_axis[DXL_ID-1]*(self.convert_to_angle(dxl_present_position) - self.initial_position[DXL_ID-1])

                    # if self.tick%self.hz ==0:
                    #     print("Joint Position [ID:%03d] : %f Deg" % (DXL_ID, angle))
                    
                    if DXL_ID != NUM_JOINTS: # change degree to radian except for gripper joint
                        self.q[DXL_ID-1] = angle*DEG2RAD
                    else:
                        self.hand_trigger_close = (angle - self.hand_trigger_release_pos)/(self.hand_trigger_pull_pos - self.hand_trigger_release_pos)
                        self.hand_trigger_close = np.clip(self.hand_trigger_close, 0.0, 1.0)
                else:
                    raise RuntimeError(
                        f"Failed to get joint angles for Dynamixel with ID {DXL_ID}"
                    )

                # if self.groupSyncVelocityRead.isAvailable(DXL_ID, DXL_CURRENT_VELOCITY_ADDR, LEN_CURRENT_VELOCITY):
                #     dxl_present_velocity = self.groupSyncVelocityRead.getData(DXL_ID, DXL_CURRENT_VELOCITY_ADDR, LEN_CURRENT_VELOCITY)
                #     velocity = self.joint_axis[DXL_ID-1]*convert_to_velocity(dxl_present_velocity)
                #     print("Joint Velocity [ID:%03d] : %f Deg/s" % (DXL_ID, velocity))
                #     if DXL_ID != NUM_JOINTS: # change degree to radian
                #         q_dot[DXL_ID-1] = dxl_present_velocity*DEG2RAD
                # else:
                #     raise RuntimeError(
                #         f"Failed to get joint velocities for Dynamixel with ID {DXL_ID}"
                #     )
                
                # if self.groupSyncCurrentRead.isAvailable(DXL_ID, DXL_CURRENT_TORQUE_ADDR, LEN_CURRENT_TORQUE):
                #     dxl_present_current = self.groupSyncCurrentRead.getData(DXL_ID, DXL_CURRENT_TORQUE_ADDR, LEN_CURRENT_TORQUE)
                #     current = self.joint_axis[DXL_ID-1]*convert_to_current(dxl_present_current)
                #     print("Joint Current [ID:%03d] : %f A" % (DXL_ID, current))
                #     if DXL_ID != NUM_JOINTS: # change degree to radian
                #         tau_c[DXL_ID-1] = current
                # else:
                #     raise RuntimeError(
                #         f"Failed to get joint velocities for Dynamixel with ID {DXL_ID}"
                #     )



            pin.forwardKinematics(self.model,self.data,self.q)
            pin.updateFramePlacements(self.model,self.data)
            
            self.gello_arm_joints.data = self.q
            self.gello_hand_tripper.data = self.hand_trigger_close
            # Print out the placement of each joint of the kinematic tree
            # for name, oMi in zip(model.names, self.data.oMi):
            #     print(("{:<24} : {: .2f} {: .2f} {: .2f}"
            #         .format( name, *oMi.translation.T.flat)))
            #     # print('rotation: ', oMi.rotation)
            #     xyzquat = se3ToXYZQUAT(oMi)
            #     print('xyzquat: ', xyzquat)

            se3_endeffector = self.data.oMf[self.tcpframeId]
            xyzquatee = pin.SE3ToXYZQUAT(se3_endeffector)
            

            if self.tick <= self.connect_start_tick + (self.connect_start_delay)*self.hz:
                delay_tick = self.tick - self.connect_start_tick
                if ((self.connect_start_delay)*self.hz - delay_tick)%self.hz == 0:
                    remain_time = self.connect_start_delay - delay_tick/self.hz
                    if remain_time == 0.0:
                        print('Connect!')
                    else:
                        print(f'{self.robot_id} master arm will be connected in {remain_time} seconds')
                self.desired_pose = self.connect_init_pose

            elif self.tick > self.connect_start_tick + (self.connect_start_delay)*self.hz and self.tick <= self.connect_start_tick + (self.connect_start_delay + self.connect_spline_duration)*self.hz:
                ### position ###
                for i in range(3):
                    self.desired_pose[i] = self.cubic(self.tick, self.connect_start_tick + (self.connect_start_delay)*self.hz, self.connect_start_tick + (self.connect_start_delay + self.connect_spline_duration)*self.hz, self.connect_init_pose[i], xyzquatee[i], 0.0, 0.0)

                ### orientation ###
                connect_init_rotation = Rotation.from_quat(self.connect_init_pose[3:7])
                current_rotation = Rotation.from_quat(xyzquatee[3:7])

                connect_init_rotm = connect_init_rotation.as_matrix()
                current_rotm = current_rotation.as_matrix()

                del_rotation = Rotation.from_matrix( np.dot(current_rotm, np.transpose(connect_init_rotm)) )
                del_rotvec = del_rotation.as_rotvec()
                tau = self.cubic(self.tick, self.connect_start_tick + (self.connect_start_delay)*self.hz, self.connect_start_tick + (self.connect_start_delay + self.connect_spline_duration)*self.hz, 0.0, 1.0, 0.0, 0.0)
                del_spline_rotvec = del_rotvec*tau
                del_spline_rotation = Rotation.from_rotvec(del_spline_rotvec)
                desired_rotm = np.dot(del_spline_rotation.as_matrix(), connect_init_rotm)

                desired_rotation = Rotation.from_matrix(desired_rotm)

                self.desired_pose[3:7] = desired_rotation.as_quat()
            else:
                self.desired_pose[:] = xyzquatee[:]
            
            # xyzquat8 = se3ToXYZQUAT(self.data.oMi[7])
            # print('xyzquat8: ', xyzquat8)
            # print('xyzquatee: ', xyzquatee)

            ### Publish ROS message
            self.gello_hand.header.stamp = rospy.Time.now()
            self.gello_hand.header.frame_id = "world"

            self.gello_hand.pose.position.x = self.desired_pose[0]
            self.gello_hand.pose.position.y = self.desired_pose[1]
            self.gello_hand.pose.position.z = self.desired_pose[2]

            self.gello_hand.pose.orientation.x = self.desired_pose[3]
            self.gello_hand.pose.orientation.y = self.desired_pose[4]
            self.gello_hand.pose.orientation.z = self.desired_pose[5]
            self.gello_hand.pose.orientation.w = self.desired_pose[6]

            self.pub_hand_pose.publish(self.gello_hand)
            self.pub_arm_joint.publish(self.gello_arm_joints)
            self.pub_hand_trigger.publish(self.gello_hand_tripper)

            self.pub_connection_signal.publish(self.connection_signal)
            self.pub_homepose_signal.publish(self.home_pose_signal)

            ## command torque to motors
            # if self.sync_signal == True and (self.sync_start_tick + self.sync_duration*self.hz >= self.tick):
            #     for i in range(NUM_ARM_JOINTS): # exept gripper joint
            #         # send control mode packet    
            #         if self.tick == self.sync_start_tick:
            #             print('change DXL OPERATION MODE TO POSITION CONTROL')
            #             self.packetHandler.write1ByteTxOnly(self.portHandler, i+1, DXL_OPERATING_MODE_ADDR, POSITION_CONTROL_MODE)
            #         self.desired_q[i] = self.cubic(self.tick, self.sync_start_tick, self.sync_start_tick + self.sync_duration*self.hz, self.sync_init_q[i], self.dsr_joint_state.data[i], 0, 0)
            #         self.desired_q[i] *= RAD2DEG
            #         print(f'self.desired_q[{i}]: ', self.desired_q[i])

            #         self.desired_q[i] = self.joint_axis[i]* self.desired_q[i]
            #         self.desired_q[i] = self.desired_q[i] + self.initial_position[i]
            #         self.desired_q[i] = self.degree_to_dxl_command(self.desired_q[i])
            #         desired_q_temp_bytes = struct.pack('i', int(self.desired_q[i]))
            #         self.groupSyncPositionWrite.addParam(i+1, desired_q_temp_bytes)                       
            #     self.groupSyncPositionWrite.txPacket()
            #     self.groupSyncPositionWrite.clearParam()

            # elif self.sync_signal == True:
            #     for dxl_id in range(1, NUM_JOINTS):
            #         # send control mode packet    
            #         self.packetHandler.write1ByteTxOnly(self.portHandler, dxl_id, DXL_OPERATING_MODE_ADDR, CURRENT_CONTROL_MODE)
            #         self.sync_signal == False


            # gravity_torque = pin.computeGeneralizedGravity(model,self.data,q)
            # print("gravity_torque: ", gravity_torque)
            # target_pos = 0

            # a_int = 300
            # b_int = -100
            # a_bytes = a_int. (2, 'little', signed = True)
            # b_bytes = b_int.to_bytes(2, 'little', signed = True)
            # # self.groupSyncTorqueWrite.addParam(5, a_bytes)
            # self.groupSyncTorqueWrite.addParam(2, a_bytes)
            # self.groupSyncTorqueWrite.addParam(3, b_bytes)
            # self.groupSyncTorqueWrite.txPacket()

            # self.groupSyncPositionWrite.addParam(1, b'0')
            # self.groupSyncPositionWrite.addParam(2, b'0')
            # self.groupSyncPositionWrite.addParam(3, b'0')
            
            # desired_q_temp_bytes = struct.pack('f', 0.0)
            # self.groupSyncPositionWrite.addParam(4, desired_q_temp_bytes)
            # self.groupSyncPositionWrite.addParam(5, b'0')
            # self.groupSyncPositionWrite.addParam(6, b'0')
            # self.groupSyncPositionWrite.txPacket()
            # print('t2-t1: ', t2-t1)
            # print('t3-t2: ', t3-t2)
            # print('t4-t3: ', t4-t3)
            # print('t5-t4: ', t5-t4)
            # print('t6-t5: ', t6-t5)
            # print('t7-t6: ', t7-t6)
            # print('t8-t7: ', t8-t7)
            
            self.tick += 1

            if self.home_pose_signal.data == True:
                self.home_pose_signal.data = False

            if self.stop_control_loop:
                break
            self.rate.sleep()
    
    def shutdown(self):
        # TORQUE OFF MOTOR
        print("DXL MOTORS TORQUE OFF")
        self.stop_control_loop = True
        for DXL_ID in range(1, NUM_JOINTS+1): 
            self.packetHandler.write1ByteTxOnly(self.portHandler, DXL_ID, DXL_TORQUE_ENABLE_ADDR, TORQUE_OFF)
        # 포트 닫기
        print("DXL PORT CLOSE")
        self.portHandler.closePort()

class dsrMasterArm:
    def __init__(self, robot_id_list=['dsr_l'], hz = 50, init_node = True):      
        if init_node:
            rospy.init_node('dsr_master_arm')

        self.hz = hz
        self.robot_id_list = robot_id_list
        self.num_robots = len(self.robot_id_list)

        rospy.on_shutdown(self.shutdown)
        self.window=tkinter.Tk()

        self.master_list = []
        self.master_thread_list = []
        for idx, robot_id in enumerate(self.robot_id_list):
            print('idx: ', idx, robot_id)
            if robot_id == 'dsr_l':
                master_static_port = '/dev/dxl_dsr_l'
            elif robot_id == 'dsr_r':
                master_static_port = '/dev/dxl_dsr_r'
            else:
                raise NotImplementedError
            
            self.master_list.append(dsrMasterArmCore(robot_id = robot_id, port=master_static_port, hz = self.hz))
            self.master_thread_list.append(threading.Thread(target=self.master_list[idx].main))
            self.master_thread_list[idx].daemon = True

    def dsrConnect(self, connect_delay = 3.0, connect_spline_duration = 3.0):
        for idx in range(self.num_robots):
            self.master_list[idx].dsrConnect(connect_delay = connect_delay, connect_spline_duration = connect_spline_duration)
    
    def dsrDisconnect(self):
        for idx in range(self.num_robots):
            self.master_list[idx].dsrDisconnect()

    def dsrHomePose(self):
        for idx in range(self.num_robots):
            self.master_list[idx].dsrHomePose()

    def thread_start(self):
        for idx in range(self.num_robots):
            self.master_thread_list[idx].start()    

    def thread_stop(self):
        for idx in range(self.num_robots):
            self.master_list[idx].stop_control_loop =True

    def set_init_q(self, robot_id, init_q):
        for idx, robot_id_in_list in enumerate(self.robot_id_list):
            if robot_id_in_list == robot_id:
                self.master_list[idx].set_init_q(init_q)

    def set_joint_axis(self, robot_id, joint_axis):
        for idx, robot_id_in_list in enumerate(self.robot_id_list):
            if robot_id_in_list == robot_id:
                self.master_list[idx].set_joint_axis(joint_axis)

    def run_gui(self):
        

        self.window.title("DSR MASTER ARM GUI")
        self.window.geometry("650x100+100+100")
        self.window.resizable(True, True)

        # label=tkinter.Label(window, text="TEST", padx=10, pady=5, width=10, height=5, fg="black", relief="solid")
        # label.grid(row=0, column=1)

        connect_button = tkinter.Button(self.window, overrelief="solid", text="Connect to DSR", font=("Arial", 12), foreground = 'white', activeforeground='white', width=20, height=4, command=self.dsrConnect, repeatdelay=0, repeatinterval=0, background='green', activebackground='grey')
        connect_button.grid(row=0, column=0, padx= 5, pady= 5)

        disconnect_button = tkinter.Button(self.window, overrelief="solid", text="Disconnect from DSR", font=("Arial", 12), foreground = 'white', activeforeground='white', width=20, height=4, command=self.dsrDisconnect, repeatdelay=0, repeatinterval=0, background='red', activebackground='grey')
        disconnect_button.grid(row=0, column=1, padx= 5, pady= 5)

        home_button = tkinter.Button(self.window, overrelief="solid", text="Home Pose", font=("Arial", 12), foreground = 'black', activeforeground='black', width=20, height=4, command=self.dsrHomePose, repeatdelay=0, repeatinterval=0, background='white', activebackground='grey')
        home_button.grid(row=0, column=2, padx= 5, pady= 5)

        self.window.mainloop()

    def shutdown(self):
        print('DXL-DSR Master Arm is shutdown!')
        self.thread_stop()
        self.window.quit()

if __name__ == "__main__":
    master_arms = dsrMasterArm(robot_id_list = ['dsr_l', 'dsr_r'], hz=40, init_node=True)
    # master_arms = dsrMasterArm(robot_id_list = ['dsr_l'], hz=40, init_node=True)
    master_arms.thread_start()

    master_arms.set_init_q('dsr_l', [90.0,180.0,180.0,180.0,180.0,360.0,180.0])
    master_arms.set_joint_axis('dsr_l', [1,1,-1,1,-1,1,1])

    master_arms.set_init_q('dsr_r', [360.0,180.0,180.0,180.0,180.0,180.0,180.0])
    master_arms.set_joint_axis('dsr_r', [1,1,-1,1,-1,1,1])

    master_arms.run_gui()






