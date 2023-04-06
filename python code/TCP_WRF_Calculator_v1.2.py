# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 14:17:39 2023

@author: g. pinto, j_dejko
"""
import numpy as np
from numpy import linalg as LA
import tkinter as tk
from PIL import ImageTk, Image

##################### TCS / WRF Calculation Funcions ##########################
class Mat():
    '''
    A Matrix class to handle the pose transformation and rotation matrices
    used for calculating WRF and TCP
    '''
    
    def __init__(self,pose):

        self.x = pose[0]
        self.y = pose[1]
        self.z = pose[2]
        
        alpha = np.deg2rad(pose[3])
        beta = np.deg2rad(pose[4])
        gamma = np.deg2rad(pose[5]) 
        sin_alpha = np.sin(alpha)
        cos_alpha = np.cos(alpha)

        sin_beta = np.sin(beta)
        cos_beta = np.cos(beta)

        sin_gamma = np.sin(gamma)
        cos_gamma = np.cos(gamma)

        self.r11 = cos_beta*cos_gamma
        self.r12 = -(cos_beta)*(sin_gamma)
        self.r13 = sin_beta

        self.r21 = cos_alpha*sin_gamma + sin_alpha*sin_beta*cos_gamma
        self.r22 = cos_alpha*cos_gamma - sin_alpha*sin_beta*sin_gamma
        self.r23 = -(sin_alpha*cos_beta)

        self.r31 = sin_alpha*sin_gamma - cos_alpha*sin_beta*cos_gamma
        self.r32 = sin_alpha*cos_gamma + cos_alpha*sin_beta*sin_gamma
        self.r33 = cos_alpha*cos_beta


    def H(self):
        return np.array([[self.r11, self.r12, self.r13, self.x],[self.r21, self.r22, self.r23, self.y],[self.r31, self.r32, self.r33, self.z],[0, 0, 0, 1]])
       
    def R33(self):
        return np.array([[self.r11, self.r12, self.r13],[self.r21, self.r22, self.r23],[self.r31, self.r32, self.r33]])


def xyzRxyz_to_H(pose):
    '''
    Function to calculate the Pose Transformation Matrix for a given (x,y,z,rx,ry,rz)
    ***Uses Euler Angle notation***
    '''
    return Mat(pose).H()

def H_to_xyzRxyz(Pose_Mat):
    '''
    Function to calculate the (x,y,z,rx,ry,rz) from the Pose Transormation Matrix.
    ***Uses Euler Angle notation***
    '''
    if abs(Pose_Mat[0,2]) == 1:
        ry = Pose_Mat[0,2]*90

        rz_rad = np.arctan2(Pose_Mat[1,0],Pose_Mat[1,1])
        rz = np.rad2deg(rz_rad)

        rx = 0
    else:
        ry_rad = np.arcsin(Pose_Mat[0,2])
        ry = np.rad2deg(ry_rad)

        rz_rad = np.arctan2(-Pose_Mat[0,1],Pose_Mat[0,0])
        rz = np.rad2deg(rz_rad)

        rx_rad = np.arctan2(-Pose_Mat[1,2],Pose_Mat[2,2])
        rx = np.rad2deg(rx_rad)

    return (Pose_Mat[0,3],Pose_Mat[1,3],Pose_Mat[2,3],rx,ry,rz)
        

def CalcWRFChild(WRF_Parent, WRF_Child):
    '''
    Function to calculate the WRF of a child reference frame with respect to BRF of Meca500
    Takes the parent WRF and child WRF values and returns the WRF of child with respect to BRF of Meca500
    Parent Frame: Reference frame defined with respect to BRF of Meca500
    Child Frame: Reference frame defined with respect to Parent Frame
    '''
    Pose_Parent = xyzRxyz_to_H(WRF_Parent)
    Pose_Child = xyzRxyz_to_H(WRF_Child)

    Pose_Child_Parent = Pose_Parent.dot(Pose_Child)

    return str(H_to_xyzRxyz(Pose_Child_Parent))


def CalcWRF(p1,p2,p3):
    '''
    Function to calculate the WRF with respect to BRF of Meca500 using 3 points
    P1 is at Origin of the reference frame
    P2 is a point on the +X axis of the reference frame
    P3 is a point on the +XY plane of the reference frame
    '''
    p1p2 = True
    p1p3 = True
    
    ux = [p2[0]-p1[0],p2[1]-p1[1],p2[2]-p1[2]] 
    norm_ux = LA.norm(ux)
    if norm_ux < 10:
        print(" P1 and P2 are too close to each other")
        label_result3.config(text="WRF calculation error:")
        label_result4.config(text="P1 and P2 are too close to each other")
        label_result5.config(text="")
        label_result6.config(text="")
        label_result7.config(text="")
        label_result8.config(text="")
        p1p2 = False
        
    else:
        uvx = ux / norm_ux # Unit Vector along X

    if p1p2:
        v13 = [p3[0]-p1[0],p3[1]-p1[1],p3[2]-p1[2]] 
        norm_v13 = LA.norm(v13)
        if norm_v13 < 10:
            print(" P3 and P1 are too close to each other")
            label_result3.config(text="WRF calculation error:")
            label_result4.config(text="P3 and P1 are too close to each other")
            label_result5.config(text="")
            label_result6.config(text="")
            label_result7.config(text="")
            label_result8.config(text="")
            p1p3 = False
            
        else:
            uv13 = v13 / norm_v13
   
    if p1p2 and p1p3:
        uz = np.cross(uvx,uv13)
        norm_uz = LA.norm(uz)
        uvz = uz / norm_uz # Unit Vector along Z

        uvy = np.cross(uvz,uvx) # Unit Vector along Y
        
        if abs(uvz[0]) == 1:
            beta_deg = uvz[0]*90

            gamma = np.arctan2(uvy[0],uvx[0])
            gamma_deg = np.rad2deg(gamma)

            alpha_deg = 0
    
        else:
            beta = np.arcsin(uvz[0])
            beta_deg = np.rad2deg(beta)

            gamma = np.arctan2(-uvy[0],uvx[0])
            gamma_deg = np.rad2deg(gamma)

            alpha = np.arctan2(-uvz[1],uvz[2])
            alpha_deg = np.rad2deg(alpha)

    
        result = [p1[0],p1[1],p1[2],alpha_deg,beta_deg,gamma_deg]
        #result_str = 'setWRF'+'(' + (','.join(format(vi, ".3f") for vi in result)+')')
        
    else:
        result= []
        #result_str = " Invalid Data"
    return result
        
def CalcTCP(nPoses):
    '''
    Function to calculate the TCP (X,Y,Z) from 4 or more poses
    Takes a list of Poses as input and returns the TCP 
    
        TCP(x,y,z) = [(R_Transpose.R)^-1].R_Transpose.p
    '''
    
    #nRows = 3*len(nPoses)
    #nCols = 3

    R = np.empty((0,3))
    p = np.empty((0,1))

    nPoseMat = []
    for pose in nPoses:
        nPoseMat.append(Mat(pose))

    for i in range(len(nPoses)):
        if i < len(nPoses)-1:
            R = np.append(R, np.array([[nPoseMat[i].r11 - nPoseMat[i+1].r11, nPoseMat[i].r12 - nPoseMat[i+1].r12, nPoseMat[i].r13 - nPoseMat[i+1].r13],[nPoseMat[i].r21 - nPoseMat[i+1].r21, nPoseMat[i].r22 - nPoseMat[i+1].r22, nPoseMat[i].r23 - nPoseMat[i+1].r23],[nPoseMat[i].r31 - nPoseMat[i+1].r31, nPoseMat[i].r32 - nPoseMat[i+1].r32, nPoseMat[i].r33 - nPoseMat[i+1].r33]]),axis = 0)
            p = np.append(p,np.array([[-nPoseMat[i].x + nPoseMat[i+1].x],[-nPoseMat[i].y + nPoseMat[i+1].y],[-nPoseMat[i].z + nPoseMat[i+1].z]]),axis = 0)
            
            
        elif i == len(nPoses)-1:
           R = np.append(R, np.array([[nPoseMat[i].r11 - nPoseMat[0].r11, nPoseMat[i].r12 - nPoseMat[0].r12, nPoseMat[i].r13 - nPoseMat[0].r13],[nPoseMat[i].r21 - nPoseMat[0].r21, nPoseMat[i].r22 - nPoseMat[0].r22, nPoseMat[i].r23 - nPoseMat[0].r23],[nPoseMat[i].r31 - nPoseMat[0].r31, nPoseMat[i].r32 - nPoseMat[0].r32, nPoseMat[i].r33 - nPoseMat[0].r33]]),axis = 0)
           p = np.append(p,np.array([[-nPoseMat[i].x + nPoseMat[0].x],[-nPoseMat[i].y + nPoseMat[0].y],[-nPoseMat[i].z + nPoseMat[0].z]]),axis = 0)
            
    
    TCP = (LA.inv(R.transpose().dot(R))).dot(R.transpose()).dot(p)
    return TCP  
###############################################################################
###############################################################################

######################### TCP CALCULATION REQUEST #############################
def tcp_calc_req():
    try:
         Pose1 = [float(pose1_x_entry.get()), float(pose1_y_entry.get()), float(pose1_z_entry.get()), float(pose1_rx_entry.get()), float(pose1_ry_entry.get()), float(pose1_rz_entry.get())]
         Pose2 = [float(pose2_x_entry.get()), float(pose2_y_entry.get()), float(pose2_z_entry.get()), float(pose2_rx_entry.get()), float(pose2_ry_entry.get()), float(pose2_rz_entry.get())]
         Pose3 = [float(pose3_x_entry.get()), float(pose3_y_entry.get()), float(pose3_z_entry.get()), float(pose3_rx_entry.get()), float(pose3_ry_entry.get()), float(pose3_rz_entry.get())]
         Pose4 = [float(pose4_x_entry.get()), float(pose4_y_entry.get()), float(pose4_z_entry.get()), float(pose4_rx_entry.get()), float(pose4_ry_entry.get()), float(pose4_rz_entry.get())]
         Pose5 = [float(pose5_x_entry.get()), float(pose5_y_entry.get()), float(pose5_z_entry.get()), float(pose5_rx_entry.get()), float(pose5_ry_entry.get()), float(pose5_rz_entry.get())]
         Pose6 = [float(pose6_x_entry.get()), float(pose6_y_entry.get()), float(pose6_z_entry.get()), float(pose6_rx_entry.get()), float(pose6_ry_entry.get()), float(pose6_rz_entry.get())]
         Pose7 = [float(pose7_x_entry.get()), float(pose7_y_entry.get()), float(pose7_z_entry.get()), float(pose7_rx_entry.get()), float(pose7_ry_entry.get()), float(pose7_rz_entry.get())]
         Pose8 = [float(pose8_x_entry.get()), float(pose8_y_entry.get()), float(pose8_z_entry.get()), float(pose8_rx_entry.get()), float(pose8_ry_entry.get()), float(pose8_rz_entry.get())]
         Pose9 = [float(pose9_x_entry.get()), float(pose9_y_entry.get()), float(pose9_z_entry.get()), float(pose9_rx_entry.get()), float(pose9_ry_entry.get()), float(pose9_rz_entry.get())]
         Pose10 = [float(pose10_x_entry.get()), float(pose10_y_entry.get()), float(pose10_z_entry.get()), float(pose10_rx_entry.get()), float(pose10_ry_entry.get()), float(pose10_rz_entry.get())]
         
         print ("--------------------------------------------------")
         print ("List of entered positions from 1st to 10th")
         print (Pose1)
         print (Pose2)
         print (Pose3)
         print (Pose4)
         print (Pose5)
         print (Pose6)
         print (Pose7)
         print (Pose8)
         print (Pose9)
         print (Pose10)
         print ("--------------------------------------------------")
         print ("Calculated New TCP. Array displayed as [[x], [y], [z]]")
         poses = [Pose1, Pose2, Pose3, Pose4, Pose5, Pose6, Pose7, Pose8, Pose9, Pose10]
         #print (poses)
         #print ("-------------------------")
         
         mytcp = CalcTCP(poses)
         print (mytcp)
         result0 = mytcp[0]
         result1 = mytcp[1]
         result2 = mytcp[2]
         label_result0.config(text="X: " + str(result0))
         label_result1.config(text="Y: " + str(result1))
         label_result2.config(text="Z: " + str(result2))
    except ValueError:
        label_result0.config(text="No value calculated for X. Please enter valid numbers!")
        label_result1.config(text="No value calculated for Y. Please enter valid numbers!")
        label_result2.config(text="No value calculated for Z. Please enter valid numbers!")

######################### WRF CALCULATION REQUEST #############################
def wrf_calc_req():
    try:
         PoseA = [float(poseA_x_entry.get()), float(poseA_y_entry.get()), float(poseA_z_entry.get()), float(poseA_rx_entry.get()), float(poseA_ry_entry.get()), float(poseA_rz_entry.get())]
         PoseB = [float(poseB_x_entry.get()), float(poseB_y_entry.get()), float(poseB_z_entry.get()), float(poseB_rx_entry.get()), float(poseB_ry_entry.get()), float(poseB_rz_entry.get())]
         PoseC = [float(poseC_x_entry.get()), float(poseC_y_entry.get()), float(poseC_z_entry.get()), float(poseC_rx_entry.get()), float(poseC_ry_entry.get()), float(poseC_rz_entry.get())]      
         print ("--------------------------------------------------")
         print ("List of entered positions from 1st to 3rd")
         print (PoseA)
         print (PoseB)
         print (PoseC)
         print ("--------------------------------------------------")
         print ("Calculated New WRF. Array displayed as [[x], [y], [z], [rX], [rY], [rZ]]")
         #poses = [Pose1, Pose2, Pose3]
         #print (poses)
         #print ("-------------------------")
         
         myWRF = CalcWRF(PoseA, PoseB, PoseC)
         print (myWRF)
         result3 = myWRF[0]
         result4 = myWRF[1]
         result5 = myWRF[2]
         result6 = myWRF[3]
         result7 = myWRF[4]
         result8 = myWRF[5]
         label_result3.config(text="X: " + str(result3))
         label_result4.config(text="Y: " + str(result4))
         label_result5.config(text="Z: " + str(result5))
         label_result6.config(text="rX: " + str(result6))
         label_result7.config(text="rY: " + str(result7))
         label_result8.config(text="rZ: " + str(result8))
    except ValueError:
        label_result3.config(text="No value calculated for X. Please enter valid numbers!")
        label_result4.config(text="No value calculated for Y. Please enter valid numbers!")
        label_result5.config(text="No value calculated for Z. Please enter valid numbers!")
        label_result6.config(text="No value calculated for rX. Please enter valid numbers!")
        label_result7.config(text="No value calculated for rY. Please enter valid numbers!")
        label_result8.config(text="No value calculated for rZ. Please enter valid numbers!")

root = tk.Tk()
root.title("TCP/WRF Calculator - Modular Automation 2023")

###############################################################################
############################  GUI  ############################################

image = Image.open("logo_medium.jpg")
photo = ImageTk.PhotoImage(image)
labellogo = tk.Label(image=photo)
labellogo.grid(row=0, column=0,padx=10, pady=10)

text_label = tk.Label(root, text="TCP / WRF Calculator - Designed and made by Modular Automation 2023")
text_label.grid(row=0, column=1, columnspan=15)

### Pose 1 Inputs
pose1_label = tk.Label(root, text="Pose 1")
pose1_label.grid(row=1, column=3)

pose1_x_label = tk.Label(root, text="X:")
pose1_x_label.grid(row=1, column=4)

pose1_x_entry = tk.Entry(root, width=10)
pose1_x_entry.grid(row=1, column=5)

pose1_y_label = tk.Label(root, text="Y:")
pose1_y_label.grid(row=1, column=6)

pose1_y_entry = tk.Entry(root, width=10)
pose1_y_entry.grid(row=1, column=7)

pose1_z_label = tk.Label(root, text="Z:")
pose1_z_label.grid(row=1, column=8)

pose1_z_entry = tk.Entry(root, width=10)
pose1_z_entry.grid(row=1, column=9)

pose1_rx_label = tk.Label(root, text="RX:")
pose1_rx_label.grid(row=1, column=10)

pose1_rx_entry = tk.Entry(root, width=10)
pose1_rx_entry.grid(row=1, column=11)

pose1_ry_label = tk.Label(root, text="RY:")
pose1_ry_label.grid(row=1, column=12)

pose1_ry_entry = tk.Entry(root, width=10)
pose1_ry_entry.grid(row=1, column=13)

pose1_rz_label = tk.Label(root, text="RZ:")
pose1_rz_label.grid(row=1, column=14)

pose1_rz_entry = tk.Entry(root, width=10)
pose1_rz_entry.grid(row=1, column=15)

### Pose 2 Inputs
pose2_label = tk.Label(root, text="Pose 2")
pose2_label.grid(row=2, column=3)

pose2_x_label = tk.Label(root, text="X:")
pose2_x_label.grid(row=2, column=4)

pose2_x_entry = tk.Entry(root, width=10)
pose2_x_entry.grid(row=2, column=5)

pose2_y_label = tk.Label(root, text="Y:")
pose2_y_label.grid(row=2, column=6)

pose2_y_entry = tk.Entry(root, width=10)
pose2_y_entry.grid(row=2, column=7)

pose2_z_label = tk.Label(root, text="Z:")
pose2_z_label.grid(row=2, column=8)

pose2_z_entry = tk.Entry(root, width=10)
pose2_z_entry.grid(row=2, column=9)

pose2_rx_label = tk.Label(root, text="RX:")
pose2_rx_label.grid(row=2, column=10)

pose2_rx_entry = tk.Entry(root, width=10)
pose2_rx_entry.grid(row=2, column=11)

pose2_ry_label = tk.Label(root, text="RY:")
pose2_ry_label.grid(row=2, column=12)

pose2_ry_entry = tk.Entry(root, width=10)
pose2_ry_entry.grid(row=2, column=13)

pose2_rz_label = tk.Label(root, text="RZ:")
pose2_rz_label.grid(row=2, column=14)

pose2_rz_entry = tk.Entry(root, width=10)
pose2_rz_entry.grid(row=2, column=15)

### Pose 3 Inputs
pose3_label = tk.Label(root, text="Pose 3")
pose3_label.grid(row=3, column=3)

pose3_x_label = tk.Label(root, text="X:")
pose3_x_label.grid(row=3, column=4)

pose3_x_entry = tk.Entry(root, width=10)
pose3_x_entry.grid(row=3, column=5)

pose3_y_label = tk.Label(root, text="Y:")
pose3_y_label.grid(row=3, column=6)

pose3_y_entry = tk.Entry(root, width=10)
pose3_y_entry.grid(row=3, column=7)

pose3_z_label = tk.Label(root, text="Z:")
pose3_z_label.grid(row=3, column=8)

pose3_z_entry = tk.Entry(root, width=10)
pose3_z_entry.grid(row=3, column=9)

pose3_rx_label = tk.Label(root, text="RX:")
pose3_rx_label.grid(row=3, column=10)

pose3_rx_entry = tk.Entry(root, width=10)
pose3_rx_entry.grid(row=3, column=11)

pose3_ry_label = tk.Label(root, text="RY:")
pose3_ry_label.grid(row=3, column=12)

pose3_ry_entry = tk.Entry(root, width=10)
pose3_ry_entry.grid(row=3, column=13)

pose3_rz_label = tk.Label(root, text="RZ:")
pose3_rz_label.grid(row=3, column=14)

pose3_rz_entry = tk.Entry(root, width=10)
pose3_rz_entry.grid(row=3, column=15)

### Pose 4 Inputs
pose4_label = tk.Label(root, text="Pose 4")
pose4_label.grid(row=4, column=3)

pose4_x_label = tk.Label(root, text="X:")
pose4_x_label.grid(row=4, column=4)

pose4_x_entry = tk.Entry(root, width=10)
pose4_x_entry.grid(row=4, column=5)

pose4_y_label = tk.Label(root, text="Y:")
pose4_y_label.grid(row=4, column=6)

pose4_y_entry = tk.Entry(root, width=10)
pose4_y_entry.grid(row=4, column=7)

pose4_z_label = tk.Label(root, text="Z:")
pose4_z_label.grid(row=4, column=8)

pose4_z_entry = tk.Entry(root, width=10)
pose4_z_entry.grid(row=4, column=9)

pose4_rx_label = tk.Label(root, text="RX:")
pose4_rx_label.grid(row=4, column=10)

pose4_rx_entry = tk.Entry(root, width=10)
pose4_rx_entry.grid(row=4, column=11)

pose4_ry_label = tk.Label(root, text="RY:")
pose4_ry_label.grid(row=4, column=12)

pose4_ry_entry = tk.Entry(root, width=10)
pose4_ry_entry.grid(row=4, column=13)

pose4_rz_label = tk.Label(root, text="RZ:")
pose4_rz_label.grid(row=4, column=14)

pose4_rz_entry = tk.Entry(root, width=10)
pose4_rz_entry.grid(row=4, column=15)

### Pose 5 Inputs
pose5_label = tk.Label(root, text="Pose 5")
pose5_label.grid(row=5, column=3)

pose5_x_label = tk.Label(root, text="X:")
pose5_x_label.grid(row=5, column=4)

pose5_x_entry = tk.Entry(root, width=10)
pose5_x_entry.grid(row=5, column=5)

pose5_y_label = tk.Label(root, text="Y:")
pose5_y_label.grid(row=5, column=6)

pose5_y_entry = tk.Entry(root, width=10)
pose5_y_entry.grid(row=5, column=7)

pose5_z_label = tk.Label(root, text="Z:")
pose5_z_label.grid(row=5, column=8)

pose5_z_entry = tk.Entry(root, width=10)
pose5_z_entry.grid(row=5, column=9)

pose5_rx_label = tk.Label(root, text="RX:")
pose5_rx_label.grid(row=5, column=10)

pose5_rx_entry = tk.Entry(root, width=10)
pose5_rx_entry.grid(row=5, column=11)

pose5_ry_label = tk.Label(root, text="RY:")
pose5_ry_label.grid(row=5, column=12)

pose5_ry_entry = tk.Entry(root, width=10)
pose5_ry_entry.grid(row=5, column=13)

pose5_rz_label = tk.Label(root, text="RZ:")
pose5_rz_label.grid(row=5, column=14)

pose5_rz_entry = tk.Entry(root, width=10)
pose5_rz_entry.grid(row=5, column=15)

### Pose 6 Inputs
pose6_label = tk.Label(root, text="Pose 6")
pose6_label.grid(row=6, column=3)

pose6_x_label = tk.Label(root, text="X:")
pose6_x_label.grid(row=6, column=4)

pose6_x_entry = tk.Entry(root, width=10)
pose6_x_entry.grid(row=6, column=5)

pose6_y_label = tk.Label(root, text="Y:")
pose6_y_label.grid(row=6, column=6)

pose6_y_entry = tk.Entry(root, width=10)
pose6_y_entry.grid(row=6, column=7)

pose6_z_label = tk.Label(root, text="Z:")
pose6_z_label.grid(row=6, column=8)

pose6_z_entry = tk.Entry(root, width=10)
pose6_z_entry.grid(row=6, column=9)

pose6_rx_label = tk.Label(root, text="RX:")
pose6_rx_label.grid(row=6, column=10)

pose6_rx_entry = tk.Entry(root, width=10)
pose6_rx_entry.grid(row=6, column=11)

pose6_ry_label = tk.Label(root, text="RY:")
pose6_ry_label.grid(row=6, column=12)

pose6_ry_entry = tk.Entry(root, width=10)
pose6_ry_entry.grid(row=6, column=13)

pose6_rz_label = tk.Label(root, text="RZ:")
pose6_rz_label.grid(row=6, column=14)

pose6_rz_entry = tk.Entry(root, width=10)
pose6_rz_entry.grid(row=6, column=15)

### Pose 7 Inputs
pose7_label = tk.Label(root, text="Pose 7")
pose7_label.grid(row=7, column=3)

pose7_x_label = tk.Label(root, text="X:")
pose7_x_label.grid(row=7, column=4)

pose7_x_entry = tk.Entry(root, width=10)
pose7_x_entry.grid(row=7, column=5)

pose7_y_label = tk.Label(root, text="Y:")
pose7_y_label.grid(row=7, column=6)

pose7_y_entry = tk.Entry(root, width=10)
pose7_y_entry.grid(row=7, column=7)

pose7_z_label = tk.Label(root, text="Z:")
pose7_z_label.grid(row=7, column=8)

pose7_z_entry = tk.Entry(root, width=10)
pose7_z_entry.grid(row=7, column=9)

pose7_rx_label = tk.Label(root, text="RX:")
pose7_rx_label.grid(row=7, column=10)

pose7_rx_entry = tk.Entry(root, width=10)
pose7_rx_entry.grid(row=7, column=11)

pose7_ry_label = tk.Label(root, text="RY:")
pose7_ry_label.grid(row=7, column=12)

pose7_ry_entry = tk.Entry(root, width=10)
pose7_ry_entry.grid(row=7, column=13)

pose7_rz_label = tk.Label(root, text="RZ:")
pose7_rz_label.grid(row=7, column=14)

pose7_rz_entry = tk.Entry(root, width=10)
pose7_rz_entry.grid(row=7, column=15)

### Pose 8 Inputs
pose8_label = tk.Label(root, text="Pose 8")
pose8_label.grid(row=8, column=3)

pose8_x_label = tk.Label(root, text="X:")
pose8_x_label.grid(row=8, column=4)

pose8_x_entry = tk.Entry(root, width=10)
pose8_x_entry.grid(row=8, column=5)

pose8_y_label = tk.Label(root, text="Y:")
pose8_y_label.grid(row=8, column=6)

pose8_y_entry = tk.Entry(root, width=10)
pose8_y_entry.grid(row=8, column=7)

pose8_z_label = tk.Label(root, text="Z:")
pose8_z_label.grid(row=8, column=8)

pose8_z_entry = tk.Entry(root, width=10)
pose8_z_entry.grid(row=8, column=9)

pose8_rx_label = tk.Label(root, text="RX:")
pose8_rx_label.grid(row=8, column=10)

pose8_rx_entry = tk.Entry(root, width=10)
pose8_rx_entry.grid(row=8, column=11)

pose8_ry_label = tk.Label(root, text="RY:")
pose8_ry_label.grid(row=8, column=12)

pose8_ry_entry = tk.Entry(root, width=10)
pose8_ry_entry.grid(row=8, column=13)

pose8_rz_label = tk.Label(root, text="RZ:")
pose8_rz_label.grid(row=8, column=14)

pose8_rz_entry = tk.Entry(root, width=10)
pose8_rz_entry.grid(row=8, column=15)

### Pose 9 Inputs
pose9_label = tk.Label(root, text="Pose 9")
pose9_label.grid(row=9, column=3)

pose9_x_label = tk.Label(root, text="X:")
pose9_x_label.grid(row=9, column=4)

pose9_x_entry = tk.Entry(root, width=10)
pose9_x_entry.grid(row=9, column=5)

pose9_y_label = tk.Label(root, text="Y:")
pose9_y_label.grid(row=9, column=6)

pose9_y_entry = tk.Entry(root, width=10)
pose9_y_entry.grid(row=9, column=7)

pose9_z_label = tk.Label(root, text="Z:")
pose9_z_label.grid(row=9, column=8)

pose9_z_entry = tk.Entry(root, width=10)
pose9_z_entry.grid(row=9, column=9)

pose9_rx_label = tk.Label(root, text="RX:")
pose9_rx_label.grid(row=9, column=10)

pose9_rx_entry = tk.Entry(root, width=10)
pose9_rx_entry.grid(row=9, column=11)

pose9_ry_label = tk.Label(root, text="RY:")
pose9_ry_label.grid(row=9, column=12)

pose9_ry_entry = tk.Entry(root, width=10)
pose9_ry_entry.grid(row=9, column=13)

pose9_rz_label = tk.Label(root, text="RZ:")
pose9_rz_label.grid(row=9, column=14)

pose9_rz_entry = tk.Entry(root, width=10)
pose9_rz_entry.grid(row=9, column=15)

### Pose 10 Inputs
pose10_label = tk.Label(root, text="Pose 10")
pose10_label.grid(row=10, column=3)

pose10_x_label = tk.Label(root, text="X:")
pose10_x_label.grid(row=10, column=4)

pose10_x_entry = tk.Entry(root, width=10)
pose10_x_entry.grid(row=10, column=5)

pose10_y_label = tk.Label(root, text="Y:")
pose10_y_label.grid(row=10, column=6)

pose10_y_entry = tk.Entry(root, width=10)
pose10_y_entry.grid(row=10, column=7)

pose10_z_label = tk.Label(root, text="Z:")
pose10_z_label.grid(row=10, column=8)

pose10_z_entry = tk.Entry(root, width=10)
pose10_z_entry.grid(row=10, column=9)

pose10_rx_label = tk.Label(root, text="RX:")
pose10_rx_label.grid(row=10, column=10)

pose10_rx_entry = tk.Entry(root, width=10)
pose10_rx_entry.grid(row=10, column=11)

pose10_ry_label = tk.Label(root, text="RY:")
pose10_ry_label.grid(row=10, column=12)

pose10_ry_entry = tk.Entry(root, width=10)
pose10_ry_entry.grid(row=10, column=13)

pose10_rz_label = tk.Label(root, text="RZ:")
pose10_rz_label.grid(row=10, column=14)

pose10_rz_entry = tk.Entry(root, width=10)
pose10_rz_entry.grid(row=10, column=15)

### Button to trigger TCP calculation
button = tk.Button(root, text="Calculate TCP", command=tcp_calc_req)
button.grid(row=5, column=0, columnspan=3)

### Results displayed for TCP calculation
label_result0 = tk.Label(root, text="Populate all 10 positions to calculate")
label_result0.grid(row=1, column=0, columnspan=2)
label_result1 = tk.Label(root, text="the TCP for robot.")
label_result1.grid(row=2, column=0, columnspan=2)
label_result2 = tk.Label(root, text="Then click on [Calculate TCP] Button.")
label_result2.grid(row=3, column=0, columnspan=2)

### WRF Inputs Fields
#label_result3 = tk.Label(root, text="--------------------------------------------------------------------------------------------------------------------------------")
label_result3 = tk.Label(root, text="")
label_result3.grid(row=11, column=0, columnspan=15, padx=0, pady=20)

### Pose 1 Inputs for WRF
poseA_label = tk.Label(root, text="Pose 1")
poseA_label.grid(row=12, column=3)

poseA_x_label = tk.Label(root, text="X:")
poseA_x_label.grid(row=12, column=4)

poseA_x_entry = tk.Entry(root, width=10)
poseA_x_entry.grid(row=12, column=5)

poseA_y_label = tk.Label(root, text="Y:")
poseA_y_label.grid(row=12, column=6)

poseA_y_entry = tk.Entry(root, width=10)
poseA_y_entry.grid(row=12, column=7)

poseA_z_label = tk.Label(root, text="Z:")
poseA_z_label.grid(row=12, column=8)

poseA_z_entry = tk.Entry(root, width=10)
poseA_z_entry.grid(row=12, column=9)

poseA_rx_label = tk.Label(root, text="RX:")
poseA_rx_label.grid(row=12, column=10)

poseA_rx_entry = tk.Entry(root, width=10)
poseA_rx_entry.grid(row=12, column=11)

poseA_ry_label = tk.Label(root, text="RY:")
poseA_ry_label.grid(row=12, column=12)

poseA_ry_entry = tk.Entry(root, width=10)
poseA_ry_entry.grid(row=12, column=13)

poseA_rz_label = tk.Label(root, text="RZ:")
poseA_rz_label.grid(row=12, column=14)

poseA_rz_entry = tk.Entry(root, width=10)
poseA_rz_entry.grid(row=12, column=15) 

### Pose 2 Inputs for WRF
poseB_label = tk.Label(root, text="Pose 2")
poseB_label.grid(row=13, column=3)

poseB_x_label = tk.Label(root, text="X:")
poseB_x_label.grid(row=13, column=4)

poseB_x_entry = tk.Entry(root, width=10)
poseB_x_entry.grid(row=13, column=5)

poseB_y_label = tk.Label(root, text="Y:")
poseB_y_label.grid(row=13, column=6)

poseB_y_entry = tk.Entry(root, width=10)
poseB_y_entry.grid(row=13, column=7)

poseB_z_label = tk.Label(root, text="Z:")
poseB_z_label.grid(row=13, column=8)

poseB_z_entry = tk.Entry(root, width=10)
poseB_z_entry.grid(row=13, column=9)

poseB_rx_label = tk.Label(root, text="RX:")
poseB_rx_label.grid(row=13, column=10)

poseB_rx_entry = tk.Entry(root, width=10)
poseB_rx_entry.grid(row=13, column=11)

poseB_ry_label = tk.Label(root, text="RY:")
poseB_ry_label.grid(row=13, column=12)

poseB_ry_entry = tk.Entry(root, width=10)
poseB_ry_entry.grid(row=13, column=13)

poseB_rz_label = tk.Label(root, text="RZ:")
poseB_rz_label.grid(row=13, column=14)

poseB_rz_entry = tk.Entry(root, width=10)
poseB_rz_entry.grid(row=13, column=15) 

### Pose 3 Inputs for WRF
poseC_label = tk.Label(root, text="Pose 3")
poseC_label.grid(row=14, column=3)

poseC_x_label = tk.Label(root, text="X:")
poseC_x_label.grid(row=14, column=4)

poseC_x_entry = tk.Entry(root, width=10)
poseC_x_entry.grid(row=14, column=5)

poseC_y_label = tk.Label(root, text="Y:")
poseC_y_label.grid(row=14, column=6)

poseC_y_entry = tk.Entry(root, width=10)
poseC_y_entry.grid(row=14, column=7)

poseC_z_label = tk.Label(root, text="Z:")
poseC_z_label.grid(row=14, column=8)

poseC_z_entry = tk.Entry(root, width=10)
poseC_z_entry.grid(row=14, column=9)

poseC_rx_label = tk.Label(root, text="RX:")
poseC_rx_label.grid(row=14, column=10)

poseC_rx_entry = tk.Entry(root, width=10)
poseC_rx_entry.grid(row=14, column=11)

poseC_ry_label = tk.Label(root, text="RY:")
poseC_ry_label.grid(row=14, column=12)

poseC_ry_entry = tk.Entry(root, width=10)
poseC_ry_entry.grid(row=14, column=13)

poseC_rz_label = tk.Label(root, text="RZ:")
poseC_rz_label.grid(row=14, column=14)

poseC_rz_entry = tk.Entry(root, width=10)
poseC_rz_entry.grid(row=14, column=15, padx=20, pady=0)

### Button to trigger TCP calculation
button = tk.Button(root, text="Calculate WRF", command=wrf_calc_req)
button.grid(row=18, column=0, columnspan=3, padx=0, pady=10)

### Results displayed for WRF calculation
label_result3 = tk.Label(root, text="Populate all 3 positions to calculate")
label_result3.grid(row=12, column=0, columnspan=2)
label_result4 = tk.Label(root, text="the WRF for robot.")
label_result4.grid(row=13, column=0, columnspan=2)
label_result5 = tk.Label(root, text="Then click on [Calculate WRF] Button.")
label_result5.grid(row=14, column=0, columnspan=2)
label_result6 = tk.Label(root, text="")
label_result6.grid(row=15, column=0, columnspan=2)
label_result7 = tk.Label(root, text="")
label_result7.grid(row=16, column=0, columnspan=2)
label_result8 = tk.Label(root, text="")
label_result8.grid(row=17, column=0, columnspan=2)

root.mainloop()


