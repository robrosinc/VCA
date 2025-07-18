#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ##
# @brief    [dsr teleoperation] control dsr according to the input from meta quest controller and send command to dsl script
# @author   Daegyu Lim (dglim@robros.co.kr)   

import rospy
import os
import threading, time
import sys
# from sensor_msgs.msg import Joy
from std_msgs.msg import Float32MultiArray, Float32, Bool
from quest2ros.msg import OVR2ROSInputs, OVR2ROSHapticFeedback
from geometry_msgs.msg import PoseStamped, Twist 
import numpy as np
from scipy.spatial.transform import Rotation
import socket
from crc import Calculator, Crc16
import struct

sys.dont_write_bytecode = True
sys.path.append( os.path.abspath(os.path.join(os.path.dirname(__file__),"../../../../common/imp")) ) # get import path : DSR_ROBOT.py 


ROBOT_MODEL = "a0509"

# import DR_init
# DR_init.__dsr__id = ROBOT_ID
# DR_init.__dsr__model = ROBOT_MODEL
# from DSR_ROBOT import *

class dsrSingleArmControl:
    def __init__(self, robot_id = 'dsr_l', hz = 30, teleop=True, master='gello', thru_cpp=False):
        self.robot_id = robot_id
        self.hz = hz
        self.dt = 1/self.hz
        self.teleop = teleop
        self.master = 'gello' # 'gello' or 'quest'

        rospy.on_shutdown(self.shutdown)
        ### publisher ###
        if not thru_cpp:
            self.pose_state_pub = rospy.Publisher('/'+robot_id +'/state/pose', PoseStamped, tcp_nodelay=True, queue_size=10)
            self.joint_state_pub = rospy.Publisher('/'+robot_id +'/state/joint', Float32MultiArray, tcp_nodelay=True, queue_size=10)
            self.pose_action_pub = rospy.Publisher('/'+robot_id +'/action/pose', PoseStamped, tcp_nodelay=True, queue_size=10)
        else:
            self.pose_state_pub = rospy.Publisher('/'+robot_id +'/state/posefrompython', PoseStamped, tcp_nodelay=True, queue_size=10)
            self.pose_action_pub = rospy.Publisher('/'+robot_id +'/action/posefrompython', PoseStamped, tcp_nodelay=True, queue_size=10)
            
        
        ### subscriber ###
        if self.master == 'quest':
            self.ovr2ros_right_hand_pose_sub = rospy.Subscriber("/q2r_right_hand_pose",PoseStamped,self.ovr2ros_right_hand_pose_callback)
            self.ovr2ros_right_hand_twist_sub = rospy.Subscriber("/q2r_right_hand_twist",Twist,self.ovr2ros_right_hand_twist_callback)
            self.ovr2ros_right_hand_inputs_sub = rospy.Subscriber("/q2r_right_hand_inputs",OVR2ROSInputs,self.ovr2ros_right_hand_inputs_callback)
            self.ovr2ros_left_hand_pose_sub = rospy.Subscriber("/q2r_left_hand_pose",PoseStamped,self.ovr2ros_left_hand_pose_callback)
            self.ovr2ros_left_hand_twist_sub = rospy.Subscriber("/q2r_left_hand_twist",Twist,self.ovr2ros_left_hand_twist_callback)
            self.ovr2ros_left_hand_inputs_sub = rospy.Subscriber("/q2r_left_hand_inputs",OVR2ROSInputs,self.ovr2ros_left_hand_inputs_callback)
        elif self.master == 'gello':
            self.gello_hand_pose_sub = rospy.Subscriber(f'/{robot_id}/teleop/gello_hand_pose',PoseStamped,self.gello_hand_pose_callback)
            self.gello_connection_sub = rospy.Subscriber(f'/{robot_id}/teleop/gello_connection',Bool,self.gello_connection_callback)
            self.gello_homepose_sub = rospy.Subscriber(f'/{robot_id}/teleop/gello_home_pose',Bool,self.gello_homepose_callback)
        else:
            raise NotImplementedError

        ### initialize variables ###
        # self.r = CDsrRobot(ROBOT_ID, ROBOT_MODEL)

        # self.msgRobotState_cb_count = 0
        self.right_hand_pose_raw = PoseStamped()
        self.right_hand_pose_raw.pose.orientation.w = 1
        self.right_hand_twist_raw = Twist()
        self.right_hand_inputs_raw = OVR2ROSInputs()

        self.left_hand_pose_raw = PoseStamped()
        self.left_hand_pose_raw.pose.orientation.w = 1
        self.left_hand_twist_raw = Twist()
        self.left_hand_inputs_raw = OVR2ROSInputs()

        self.right_hand_pose = PoseStamped()
        self.right_hand_pose.pose.orientation.w = 1
        self.right_hand_twist = Twist()
        self.right_hand_inputs = OVR2ROSInputs()

        self.left_hand_pose = PoseStamped()
        self.left_hand_pose.pose.orientation.w = 1
        self.left_hand_twist = Twist()
        self.left_hand_inputs = OVR2ROSInputs()

        self.right_arm_pose_state_msg = PoseStamped()
        self.right_arm_pose_state_msg.header.frame_id = "world"
        self.right_arm_pose_state_msg.pose.orientation.w = 1
        self.right_gripper_state_msg = Float32()

        self.right_arm_pose_action_msg = PoseStamped()
        self.right_arm_pose_action_msg.header.frame_id = "world"
        self.right_arm_pose_action_msg.pose.orientation.w = 1
        self.right_gripper_action_msg = Float32()

        self.current_dsr_pos_raw = np.zeros((3))
        self.current_dsr_pos = np.zeros((3))
        self.start_dsr_pos = np.zeros((3))
        self.current_controller_pos = np.zeros((3))
        self.start_controller_pos = np.zeros((3))

        self.current_dsr_rotm_raw = np.eye(3)
        self.current_dsr_rotm = np.eye(3)
        self.start_dsr_rotm = np.eye(3)
        self.currnet_controller_pose = np.eye(3)
        self.start_controller_rotm = np.eye(3)

        self.current_dsr_euler_raw = np.zeros((3))
        self.current_dsr_euler = np.zeros((3))
        self.start_dsr_euler = np.zeros((3))

        self.current_dsr_rotvec_raw = np.zeros((3))
        self.current_dsr_rotvec = np.zeros((3))

        self.controller_delta_rotvec = np.zeros((3))
        self.controller_delta_rotvec_pre = np.zeros((3))
        self.controller_delta_rotvec_lpf = np.zeros((3))

        self.right_button_lower_pre = False
        self.right_botton_lower_clicked = False
        self.right_botton_lower_released = False

        self.right_button_upper_pre = False
        self.right_botton_upper_clicked = False
        self.right_botton_upper_released = False
        

        self.left_button_lower_pre = False
        self.left_botton_lower_clicked = False

        self.left_button_upper_pre = False
        self.left_botton_upper_clicked = False
        self.left_botton_upper_released = False

        self.dsr_desired_x_pose = [0, 0, 0, 0, 0, 0]
        self.dsr_desired_x_pose_pre = [0, 0, 0, 0, 0, 0]
        self.dsr_desired_x_pose_lpf = [0, 0, 0, 0, 0, 0]

        self.dsr_desired_rotm = np.eye(3)
        self.quat_des = np.array([0,0,0,1])
        # self.dsr_desired_x_home_pose = [365,  0, 290, 0, -180, 0]

        self.dsr_desired_x_vel = [0, 0, 0, 0, 0, 0]
        self.dsr_desired_x_vel_pre = [0, 0, 0, 0, 0, 0]
        self.dsr_desired_x_vel_lpf = [0, 0, 0, 0, 0, 0]

        self.dsr_desired_zero_pose_j = [0, 0, 0, 0, 0, 0]

        self.control_mode = 0 # 1: teleoperation, 2: ready position, 3: zero position, 0: stop robot
        self.tick = 0

        if self.robot_id == 'dsr_l':
            self.drl_tcp_client = TcpClient(tcp_ip="192.168.0.77", tcp_port=777)
        elif self.robot_id == 'dsr_r':
            self.drl_tcp_client = TcpClient(tcp_ip="192.168.0.78", tcp_port=778)
        else:
            raise NotImplementedError

        rospy.set_param('/use_sim_time', False)
        time.sleep(1)
        # t1 = threading.Thread(target=teleop.thread_subscriber)
        self.thread1 = threading.Thread(target=self.tcp_trhead)
        self.thread1.daemon = True 
        self.thread1.start()

        while not rospy.is_shutdown() and self.drl_tcp_client.first_data_get == False:
            time.sleep(1)
            print('waiting robot data read from DSL tcp server...')
        
        self.dsr_desired_x_pose_lpf[:] = self.drl_tcp_client.current_posx[:]

        self.stop_control_loop = False
        

    def shutdown(self):
        print(f"\n DSR({self.robot_id}) CONTROL SHUTDOWN! \n")
        self.stop()
        self.stop_control_loop = True
        self.drl_tcp_client.shutdown()
        return 0

    # def msgRobotState_cb(self, msg):
    #     self.msgRobotState_cb_count += 1

    #     if (0==(self.msgRobotState_cb_count % 100)): 
    #         rospy.loginfo("________ ROBOT STATUS ________")
    #         print("  robot_state       : %d" % (msg.robot_state))
    #         print("  robot_state_str   : %s" % (msg.robot_state_str))
    #         print("  current_posj      : %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f" % (msg.current_posj[0],msg.current_posj[1],msg.current_posj[2],msg.current_posj[3],msg.current_posj[4],msg.current_posj[5]))
    #         print("  current_posx      : %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f" % (msg.current_posx[0],msg.current_posx[1],msg.current_posx[2],msg.current_posx[3],msg.current_posx[4],msg.current_posx[5]))

    #         #print("  io_control_box    : %d" % (msg.io_control_box))
    #         ##print("  io_modbus         : %d" % (msg.io_modbus))
    #         ##print("  error             : %d" % (msg.error))
    #         #print("  access_control    : %d" % (msg.access_control))
    #         #print("  homming_completed : %d" % (msg.homming_completed))
    #         #print("  tp_initialized    : %d" % (msg.tp_initialized))
    #         #print("  speed             : %d" % (msg.speed))
    #         #print("  mastering_need    : %d" % (msg.mastering_need))
    #         #print("  drl_stopped       : %d" % (msg.drl_stopped))
    #         #print("  disconnected      : %d" % (msg.disconnected))

    #     self.current_dsr_pos_raw[0] = msg.current_posx[0]
    #     self.current_dsr_pos_raw[1] = msg.current_posx[1]
    #     self.current_dsr_pos_raw[2] = msg.current_posx[2]
        
    #     self.current_dsr_euler_raw[0] = msg.current_posx[3]
    #     self.current_dsr_euler_raw[1] = msg.current_posx[4]
    #     self.current_dsr_euler_raw[2] = msg.current_posx[5]

    #     r_current_ori = Rotation.from_euler("ZYZ", msg.current_posx[3:6], degrees=True)
    #     self.current_dsr_rotm_raw = r_current_ori.as_matrix()
    #     self.current_dsr_rotvec_raw = r_current_ori.as_rotvec()
    

    # def thread_subscriber(self):
    #     rospy.Subscriber('/'+ROBOT_ID +ROBOT_MODEL+'/state', RobotState, self.msgRobotState_cb)
    #     rospy.spin()
    #     #rospy.spinner(2)    


    def ovr2ros_right_hand_pose_callback(self, data): 
        self.right_hand_pose_raw = data

    def ovr2ros_right_hand_twist_callback(self, data):  
        self.right_hand_twist_raw = data

    def ovr2ros_right_hand_inputs_callback(self, data):    
        self.right_hand_inputs_raw = data

    def ovr2ros_left_hand_pose_callback(self, data): 
        self.left_hand_pose_raw = data

    def ovr2ros_left_hand_twist_callback(self, data):  
        self.left_hand_twist_raw = data

    def ovr2ros_left_hand_inputs_callback(self, data):    
        self.left_hand_inputs_raw = data

    def gello_hand_pose_callback(self, data): 
        self.right_hand_pose_raw = data
    
    def gello_connection_callback(self, data): 
        self.right_hand_inputs_raw.button_lower = data.data

    def gello_homepose_callback(self, data): 
        self.right_hand_inputs_raw.button_upper = data.data

    def readControllerInputs(self):
        self.right_hand_pose = self.right_hand_pose_raw
        self.right_hand_twist = self.right_hand_twist_raw
        self.right_hand_inputs = self.right_hand_inputs_raw
        self.left_hand_pose = self.left_hand_pose_raw
        self.left_hand_twist = self.left_hand_twist_raw
        self.left_hand_inputs = self.left_hand_inputs_raw

        self.current_controller_pos[0] = self.right_hand_pose.pose.position.x
        self.current_controller_pos[1] = self.right_hand_pose.pose.position.y
        self.current_controller_pos[2] = self.right_hand_pose.pose.position.z

        # Rotation.from_quaternion
        # rotation_right_controller = Rotation.from_quat([self.right_hand_pose.pose.orientation.w, self.right_hand_pose.pose.orientation.x, self.right_hand_pose.pose.orientation.y, self.right_hand_pose.pose.orientation.z])
        # rotation_right_controller = Rotation.from_quat([-self.right_hand_pose.pose.orientation.z, self.right_hand_pose.pose.orientation.x, -self.right_hand_pose.pose.orientation.y, self.right_hand_pose.pose.orientation.w])
        rotation_right_controller = Rotation.from_quat([self.right_hand_pose.pose.orientation.x, self.right_hand_pose.pose.orientation.y, self.right_hand_pose.pose.orientation.z, self.right_hand_pose.pose.orientation.w])
        self.current_controller_rotm = rotation_right_controller.as_matrix()
        # self.current_controller_rotm = quaternion.as_rotation_matrix(self.right_hand_pose.pose.orientation)
        # if teleop.tick % 5 == 0:
        #     print(self.current_controller_rotm)

        self.right_gripper_action_msg.data = self.right_hand_inputs.press_index

        if self.right_button_lower_pre == False and self.right_hand_inputs.button_lower == True:
            self.right_botton_lower_clicked = True
            print("right button A is clicked!")

        if self.right_button_lower_pre == True and self.right_hand_inputs.button_lower == False:
            self.right_botton_lower_released = True
            print("right button A is released!")

        if self.right_button_upper_pre == False and self.right_hand_inputs.button_upper == True:
            self.right_botton_upper_clicked = True
            print("right button B is clicked!")


        if self.left_button_lower_pre == False and self.left_hand_inputs.button_lower == True:
            self.left_botton_lower_clicked = True
            # print("left button A is clicked!")

        if self.left_button_upper_pre == False and self.left_hand_inputs.button_upper == True:
            self.left_botton_upper_clicked = True
            # print("left button B is clicked!")

        if self.left_button_upper_pre == True and self.left_hand_inputs.button_upper == False:
            self.left_botton_upper_released = True
            # print("left button B is released!")

    def readRobotData(self):
        # self.current_dsr_pos = self.current_dsr_pos_raw.copy()
        # self.current_dsr_rotm = self.current_dsr_rotm_raw.copy()
        # self.current_dsr_euler = self.current_dsr_euler_raw.copy()
        # self.current_dsr_rotvec = self.current_dsr_rotvec_raw.copy()

        current_dsr_rotation = Rotation.from_euler("ZYZ", self.current_dsr_euler, degrees=True)
        current_dsr_quat = current_dsr_rotation.as_quat() # [x, y, z, w]

        self.right_arm_pose_state_msg.header.stamp = rospy.Time.now()
        self.right_arm_pose_state_msg.pose.position.x = self.current_dsr_pos[0]/1000 # [mm] to [m]
        self.right_arm_pose_state_msg.pose.position.y = self.current_dsr_pos[1]/1000
        self.right_arm_pose_state_msg.pose.position.z = self.current_dsr_pos[2]/1000
        self.right_arm_pose_state_msg.pose.orientation.x = current_dsr_quat[0]
        self.right_arm_pose_state_msg.pose.orientation.y = current_dsr_quat[1]
        self.right_arm_pose_state_msg.pose.orientation.z = current_dsr_quat[2]
        self.right_arm_pose_state_msg.pose.orientation.w = current_dsr_quat[3]


    def calculateTragetPose(self):
        # self.control_mode = 0

        if self.right_botton_lower_clicked:
            self.right_botton_lower_clicked = False
            # save the start pose
            self.start_dsr_pos = self.current_dsr_pos.copy()
            self.start_dsr_rotm = self.current_dsr_rotm.copy()
            self.start_controller_pos = self.current_controller_pos.copy()
            self.start_controller_rotm = self.current_controller_rotm.copy()
            self.start_dsr_euler = self.current_dsr_euler.copy()

            self.dsr_desired_x_pose_pre[0:3] = self.start_dsr_pos[0:3]
            self.dsr_desired_x_pose_pre[3:6] = self.start_dsr_euler[0:3]

            self.controller_delta_rotvec_pre = np.zeros(3)
            self.controller_delta_rotvec_lpf = np.zeros(3)

            self.dsr_desired_x_pose_lpf[0:3] = self.start_dsr_pos[0:3]
            self.dsr_desired_x_pose_lpf[3:6] = self.start_dsr_euler[0:3]
            # self.dsr_desired_x_pose_lpf = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            
            self.position_motion_gain = 1.0
            self.lpf_cutoff_hz = 3
            
            # self.max_lin_vel = 200
            # self.max_ang_vel = 100

            # print("Robot is connected with controller", self.start_controller_pos)

        if self.right_botton_lower_released:
            self.right_botton_lower_released = False
            # self.dsr_desired_x_pose_pre[0:3] = self.current_dsr_pos.copy()
            # self.dsr_desired_x_pose_pre[3:6] = self.current_dsr_euler.copy()

            # self.dsr_desired_x_pose_lpf[0:3] = self.current_dsr_pos.copy()
            # self.dsr_desired_x_pose_lpf[3:6] = self.current_dsr_euler.copy()


        if self.right_hand_inputs.button_lower == True:
            self.control_mode = 1

            ############################## linear velocity ##################################
            for i in range(3):
                if self.master == 'quest':
                    #### META QUEST ####
                    self.dsr_desired_x_pose[i] = self.start_dsr_pos[i] + self.position_motion_gain*(self.current_controller_pos[i] - self.start_controller_pos[i])*1000 # multiply 1000 for [m] to [mm]
                elif self.master == 'gello':
                    #### GELLO ####
                    self.dsr_desired_x_pose[i] = self.position_motion_gain*(self.current_controller_pos[i])*1000 # multiply 1000 for [m] to [mm]
                
            ## linear velocity clipping
            # d_x_pose = np.array(self.dsr_desired_x_pose[0:3]) - np.array(self.dsr_desired_x_pose_pre[0:3])
            # d_x_pose_norm = np.linalg.norm(d_x_pose)
            # if d_x_pose_norm/self.dt > self.max_lin_vel:
            #     print("[WARNING] over maximum lin vel", d_x_pose_norm/self.dt)
            #     for i in range(3):
            #         self.dsr_desired_x_pose[i] = self.dsr_desired_x_pose_pre[i] + (d_x_pose[i])/d_x_pose_norm*self.max_lin_vel

            for i in range(3):
                self.dsr_desired_x_pose_lpf[i] = self.lpf(self.dsr_desired_x_pose[i], self.dsr_desired_x_pose_lpf[i], self.lpf_cutoff_hz)

            #################################################################################

            ############################## agnular velocity ##################################
            if self.master == 'quest':
                #### META QUEST ####
                controller_del_rot = Rotation.from_matrix( np.dot(self.current_controller_rotm, np.transpose(self.start_controller_rotm)) )
                self.controller_delta_rotvec = controller_del_rot.as_rotvec(degrees=True)
                
                for i in range(3):
                    self.controller_delta_rotvec_lpf[i] = self.lpf(self.controller_delta_rotvec[i], self.controller_delta_rotvec_lpf[i], self.lpf_cutoff_hz) 
                        
                # R_des = r_delta_controller_lpf.as_matrix() # orientation lpf version
                r_delta_controller_lpf = Rotation.from_rotvec(self.controller_delta_rotvec_lpf, degrees = True)
                self.dsr_desired_rotm = np.dot( r_delta_controller_lpf.as_matrix(), self.start_dsr_rotm) # orientation lpf version
            elif self.master == 'gello':
                #### GELLO ####
                self.dsr_desired_rotm = self.current_controller_rotm
                r_des = Rotation.from_matrix(self.dsr_desired_rotm)
                self.dsr_desired_x_pose_lpf[3:6] = r_des.as_euler("ZYZ", degrees=True)

            self.quat_des = r_des.as_quat()
            #################################################################################

        elif self.right_botton_upper_clicked:
            self.control_mode = 2
            print("move to READY-POSE")
            self.right_botton_upper_clicked = False
        elif self.left_botton_upper_clicked:
            self.control_mode = 0
            print("STOP ROBOT!")
            self.left_botton_upper_clicked = False


        # movej_result = self.r.movej(self.dsr_desired_ready_pose_j, time=3)
        # x1 = self.dsr_desired_x_home_pose
        # x1[2] -= 0.1
        # movel_result = self.r.amovel(x1, v=300, a=300)
        # if movel_result < 0:
        #         print("movel is not working correctly, ", movel_result)
        # movel_result = self.r.movel(x1, time=10, ref=DR_BASE, mod=DR_MV_MOD_ABS)
        # if movel_result < 0:
        #         print("movel is not working correctly, ", movel_result)
        # movej_result = self.r.movej(self.dsr_desired_ready_pose_j, time=3)
        # if movej_result < 0:
        #         print("movej is not working correctly, ", movej_result)

    def calculateTragetVelocity(self, target_pose):
        # position
        k_v = 2.3 # 2.3
        target_vel = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for i in range(3):
            target_vel[i] = np.clip(k_v*(target_pose[i] - self.current_dsr_pos[i]), -300, 300) # dim: [mm/s]

        # rotation
        target_rotation = Rotation.from_euler("ZYZ", target_pose[3:6], degrees=True)
        self.dsr_desired_rotm = target_rotation.as_matrix()
        delta_rotation = Rotation.from_matrix( np.dot(self.dsr_desired_rotm, np.transpose(self.current_dsr_rotm)) )
        k_w = 2.5
        for i in range(3):
            target_vel[i+3] = np.clip(k_w*delta_rotation.as_rotvec(degrees = True)[i], -90.0, 90.0) # dim:[deg/s]

        return target_vel

    def rosMsgPublish(self):
        self.right_arm_pose_action_msg.pose.position.x = self.dsr_desired_x_pose_lpf[0]/1000
        self.right_arm_pose_action_msg.pose.position.y = self.dsr_desired_x_pose_lpf[1]/1000
        self.right_arm_pose_action_msg.pose.position.z = self.dsr_desired_x_pose_lpf[2]/1000

        action_rotation = Rotation.from_euler("ZYZ", self.dsr_desired_x_pose_lpf[3:6], degrees=True)
        self.quat_des = action_rotation.as_quat()
        
        self.right_arm_pose_action_msg.pose.orientation.x = self.quat_des[0]
        self.right_arm_pose_action_msg.pose.orientation.y = self.quat_des[1]
        self.right_arm_pose_action_msg.pose.orientation.z = self.quat_des[2]
        self.right_arm_pose_action_msg.pose.orientation.w = self.quat_des[3]
        self.right_arm_pose_action_msg.header.stamp = rospy.Time.now()
            
        self.pose_state_pub.publish(self.right_arm_pose_state_msg)
        self.pose_action_pub.publish(self.right_arm_pose_action_msg)


    def savePrevData(self):
        self.right_button_lower_pre = self.right_hand_inputs.button_lower
        self.right_button_upper_pre = self.right_hand_inputs.button_upper
        self.left_button_lower_pre = self.left_hand_inputs.button_lower
        self.left_button_upper_pre = self.left_hand_inputs.button_upper
        
        self.dsr_desired_x_pose_pre = self.dsr_desired_x_pose
        self.controller_delta_rotvec_pre = self.controller_delta_rotvec.copy()

    def lpf(self, data, prev_data, cutoff_frequency):
        self.tau = 1 / (2 * np.pi * cutoff_frequency)
        val = (self.dt * data + self.tau * prev_data) / (self.tau + self.dt)
        return val
    
    def tcp_trhead(self):
        
        while not rospy.is_shutdown():
            self.drl_tcp_client.read_data()
            for i in range(3):
                self.current_dsr_pos[i] = self.drl_tcp_client.current_posx[i]
                self.current_dsr_euler[i] = self.drl_tcp_client.current_posx[i+3]
            self.current_dsr_rotm = self.drl_tcp_client.current_dsr_rotm.copy()
            self.current_dsr_rotvec = self.drl_tcp_client.current_dsr_rotvec.copy()

            self.dsr_desired_x_vel = self.calculateTragetVelocity(self.dsr_desired_x_pose_lpf)
            # print('self.dsr_desired_x_vel: ', self.dsr_desired_x_vel)
            self.drl_tcp_client.send_command(self.control_mode, self.dsr_desired_x_vel)

            if self.control_mode >= 2: # send preset motion mode only once
                self.control_mode = 0

    def step(self):
        # print(1e9*rospy.Time.now().secs + rospy.Time.now().nsecs)
        # read controller inputs
        t1 = rospy.Time.now()
        # drl_tcp_client.read_data()
        
        # read robot data
        self.readRobotData()

        # if self.tick % self.hz == 0:
        #     print("teleop.current_dsr_pos: ", self.current_dsr_pos[0:3])
        #     print("teleop.dsr_desired_x_pos_lpf: ", self.dsr_desired_x_pose_lpf[0:3])
        # make motion of DSR
        if self.teleop:
            self.readControllerInputs()
            self.calculateTragetPose()
        t2 = rospy.Time.now()
        # self.drl_tcp_client.send_command(self.control_mode, self.dsr_desired_x_vel) # inverse trigger command
        # print(f'get_xpos: {self.get_xpos()}')
        # print(f'get_euler: {self.get_euler()}')
        # print(f'rotvec: {self.current_dsr_rotvec}')
        # print(f'get_action: {self.get_action()}')
        
        self.rosMsgPublish()
        # save current values to the previous values
        self.savePrevData()
        self.tick += 1
        t3 = rospy.Time.now()
        # print("t2-t1: ", (t2-t1))
        # print("t3-t2: ", (t3-t2))
        # print("t3-t1: ", (t3-t1))
        # time.sleep(0.1)

    def control_loop(self):
        print(f'DSR ({self.robot_id}) CONTROL START')
        rate = rospy.Rate(self.hz)
        # rate = rospy.Rate(20)
        
        while not rospy.is_shutdown():
            if self.stop_control_loop:
                break
            self.step()
            rate.sleep()
        print('good bye!')

    def stop(self):
        print(f"stop dsr({self.robot_id}) robot")
        self.control_mode = 0
        self.dsr_desired_x_pose_lpf[0:3] = self.current_dsr_pos[:]
        self.dsr_desired_x_pose_lpf[3:6] = self.current_dsr_euler[:]

        self.dsr_desired_x_vel = [0, 0, 0, 0, 0, 0]
        self.drl_tcp_client.send_command(self.control_mode, self.dsr_desired_x_vel)

    def get_xpos(self):
        return self.current_dsr_pos
    
    def get_euler(self):
        return self.current_dsr_euler
    
    def get_action(self):
        return np.array(self.dsr_desired_x_pose_lpf)

    def set_action(self, target_pose):
        self.control_mode = 1
        for i in range(6):
            self.dsr_desired_x_pose_lpf[i] = target_pose[i]
    
class TcpClient:
    def __init__(self, tcp_ip="192.168.0.77", tcp_port=777):        
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = 1
        while result and (not rospy.is_shutdown()):
            result = self.client_socket.connect_ex((tcp_ip, tcp_port))
            print('Waiting for DSL server...(', result,')')
            if result == 0:
                print(f'TCP Client is connected! ({tcp_ip}, {tcp_port})')
            rospy.sleep(1.0)

        
        self.calculator = Calculator(Crc16.MODBUS, optimized=True)
        self.current_posx = [0,0,0,0,0,0]
        self.current_dsr_rotm = np.eye(3)
        self.current_dsr_rotvec = np.zeros(3)
        self.first_data_get = False

    def send_command(self, control_mode, desired_hand_vel):    
        control_mode_byte = struct.pack('1I', control_mode)
        desired_hand_vel_bytes = struct.pack('6f', desired_hand_vel[0],desired_hand_vel[1],desired_hand_vel[2],desired_hand_vel[3],desired_hand_vel[4],desired_hand_vel[5] )
        command_bytes = control_mode_byte + desired_hand_vel_bytes

        # print("command_bytes: ",command_bytes)
        self.client_socket.send(command_bytes)

    def read_data(self):
    
        # t1 = rospy.Time.now()
        try:
            self.current_posx_bytes = self.client_socket.recv(24)
        except:
            print("Exit DSR state recieve thread!")
            return
        # print("self.current_posx_bytes: ",self.current_posx_bytes)
        if len(self.current_posx_bytes) == 24:
            currnet_posx_tuple = struct.unpack('6f', self.current_posx_bytes)
        else:
            print('[WARNING] the size of current_posx_bytes is not 24!!')
            # break
        # print("currnet_posx_tuple: ", currnet_posx_tuple)
        for i in range(6):
            self.current_posx[i] = currnet_posx_tuple[i]
        # self.right_gripper_state_msg.data = currnet_posx_tuple[6]

        r_current_ori = Rotation.from_euler("ZYZ", self.current_posx[3:6], degrees=True)
        self.current_dsr_rotm = r_current_ori.as_matrix()
        self.current_dsr_rotvec = r_current_ori.as_rotvec()
        # t2 = rospy.Time.now()
        # print("t2-t1: ", (t2-t1)) # 90~150 ms
        if self.first_data_get == False:
            self.first_data_get = True
            print('get first robot data: ', self.current_posx)
    
    def shutdown(self):
        print('TcpClient is shutdown')
    #     self.client_socket.close()
        

class drlControl:
    def __init__(self, robot_id_list = ['dsr_l'], hz = 30, init_node=True, teleop=True, thru_cpp=False):
        
        if init_node:
            rospy.init_node('dsr_teleop_control_py')

        self.hz = hz
        self.teleop = teleop
        self.robot_id_list = robot_id_list
        self.num_robots = len(self.robot_id_list)

        self.dsr_list = []
        self.dsr_thread_list = []
        for idx, robot_id in enumerate(self.robot_id_list):
            print('idx: ', idx, robot_id)
            self.dsr_list.append(dsrSingleArmControl(robot_id = robot_id, hz = self.hz, teleop=self.teleop))
            self.dsr_thread_list.append(threading.Thread(target=self.dsr_list[idx].control_loop))

    def control_thread_start(self):
        for idx in range(self.num_robots):
            self.dsr_thread_list[idx].start()
    
    def control_thread_stop(self):
        for idx in range(self.num_robots):
            self.dsr_list[idx].stop_control_loop = True

    def stop(self):
        for idx in range(self.num_robots):
            self.dsr_list[idx].stop()
            
            # self.dsr_list[idx].control_mode = 0

            # self.dsr_list[idx].dsr_desired_x_pose_lpf[0:3] = self.dsr_list[idx].current_dsr_pos[:]
            # self.dsr_list[idx].dsr_desired_x_pose_lpf[3:6] = self.dsr_list[idx].current_dsr_euler[:]

            # self.dsr_list[idx].dsr_desired_x_vel = [0, 0, 0, 0, 0, 0]
            # self.dsr_list[idx].drl_tcp_client.send_command(self.dsr_list[idx].control_mode, self.dsr_list[idx].dsr_desired_x_vel)

    def get_xpos(self):
        current_dsr_pos = np.zeros(0)
        for idx in range(len(self.robot_id_list)):
            current_dsr_pos = np.concatenate( (current_dsr_pos, self.dsr_list[idx].current_dsr_pos) )
        return current_dsr_pos
    
    def get_euler(self):
        current_dsr_euler = np.zeros(0)
        for idx in range(len(self.robot_id_list)):
            current_dsr_euler = np.concatenate( (current_dsr_euler, self.dsr_list[idx].current_dsr_euler) )
        return current_dsr_euler

    def get_action(self):
        dsr_actions = np.zeros(0)
        for idx in range(len(self.robot_id_list)):
            dsr_desired_x_pose_temp = np.array(self.dsr_list[idx].dsr_desired_x_pose_lpf)
            dsr_actions = np.concatenate( (dsr_actions, dsr_desired_x_pose_temp) )
        return dsr_actions

    def set_action(self, target_pose):
        for idx in range(len(self.robot_id_list)):
            self.dsr_list[idx].control_mode = 1
            for i in range(6):
                self.dsr_list[idx].dsr_desired_x_pose_lpf[i] = target_pose[6*idx+i]

if __name__ == "__main__":
    # rospy.init_node('metaquest_teleop_py')
    # dsr_l = dsrSingleArmControl(robot_id = 'dsr_l', hz = 30, teleop=True)

    # dsr_l_thread = threading.Thread(target=dsr_l.control_loop)
    # dsr_l_thread.daemon = True
    # dsr_l_thread.start()

    # dsr_l.control_loop()

    drlcontrol = drlControl(robot_id_list=['dsr_l', 'dsr_r'], hz = 20, init_node=True, teleop= True)
    # drlcontrol = drlControl(robot_id_list=['dsr_r'], hz = 20, init_node=True, teleop= True)
    drlcontrol.control_thread_start()
    