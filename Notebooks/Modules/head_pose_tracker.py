import time
import math
import re
import sys
import os
import argparse

# Setting environment variable to avoid a known issue with OpenMP libraries
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.backends import cudnn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from face_detection import RetinaFace 
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image
matplotlib.use('TkAgg')

from Modules.sixdrepnet.model import SixDRepNet
import utils
from Modules.game_communicator import GameCommunicator

class HeadPoseTracker:

    """
    HeadPoseTracker class for tracking head pose and communicating with a game.

    Args:
    - gpu_id (int): GPU device ID. Set to -1 for CPU mode.
    - cam_id (int): Camera device ID.
    - game_communicator (GameCommunicator): Instance of the GameCommunicator class for sending commands to the game.

    Attributes:
    - gpu_id (int): GPU device ID.
    - cam_id (int): Camera device ID.
    - game_communicator (GameCommunicator): Instance of the GameCommunicator class.
    - zero_pitch (float): Initial pitch angle.
    - zero_yaw (float): Initial yaw angle.
    - zero_roll (float): Initial roll angle.
    - device (torch.device): Device (CPU or GPU) for computations.
    - model (SixDRepNet): Instance of the SixDRepNet model for head pose estimation.
    - detector (RetinaFace): Instance of the RetinaFace detector for face detection.
    - transformations (torchvision.transforms.Compose): Image transformations.

    Methods:
    - initialize_model_and_detector(): Initialize the SixDRepNet model and RetinaFace detector.
    - calibrate(): Calibrate the head pose tracker by determining the initial angles.
    - play(): Continuously track head pose and update the game_communicator with direction commands.
    - update_game_communicator(p_pred_deg, y_pred_deg): Update game_communicator with direction commands based on head pose angles.
    """



    def __init__(self, gpu_id, cam_id, game_communicator):

        """
        Initialize the HeadPoseTracker.

        Args:
        - gpu_id (int): GPU device ID. Set to -1 for CPU mode.
        - cam_id (int): Camera device ID.
        - game_communicator (GameCommunicator): Instance of the GameCommunicator class for sending commands to the game.
        """

        # Initialization of attributes
        self.gpu_id = gpu_id
        self.cam_id = cam_id
        self.game_communicator = game_communicator
        self.zero_pitch = 0
        self.zero_yaw = 0
        self.zero_roll = 0

        # Enable cuDNN for GPU acceleration
        cudnn.enabled = True
        # Set the computational device (CPU or GPU)
        if gpu_id < 0:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(f'cuda:{gpu_id}')

        # Initialize the SixDRepNet model and RetinaFace detector
        self.model, self.detector = self.initialize_model_and_detector()

        # Define image transformations for input to the model
        self.transformations = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])



    def initialize_model_and_detector(self):

        """
        Initialize the SixDRepNet model and RetinaFace detector.

        Returns:
        - model (SixDRepNet): Initialized SixDRepNet model.
        - detector (RetinaFace): Initialized RetinaFace detector.
        """

        # Path to the pre-trained SixDRepNet model
        snapshot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Models", "6DRepNet_300W_LP_AFLW2000.pth")
        
        # Create an instance of SixDRepNet model with the specified backbone
        model = SixDRepNet(backbone_name='RepVGG-B1g2', backbone_file='', deploy=True, pretrained=False)
        # Create an instance of RetinaFace detector
        detector = RetinaFace(gpu_id=self.gpu_id)

        # Load the pre-trained weights into the model
        saved_state_dict = torch.load(os.path.join(snapshot_path), map_location='cpu')
        if 'model_state_dict' in saved_state_dict:
            model.load_state_dict(saved_state_dict['model_state_dict'])
        else:
            model.load_state_dict(saved_state_dict)
        # Move the model to the specified device (CPU or GPU)
        model.to(self.device)

        # Set the model to evaluation mode
        model.eval()
        
        return model, detector



    def calibrate(self):

        """
        Calibrate the head pose tracker by determining the initial angles.
        """

        zero_pitch = zero_roll = zero_yaw = 0
        current_frame = 0

        # Open the camera for capturing video frames
        cap = cv2.VideoCapture(self.cam_id)

        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        with torch.no_grad():
            # Capture and process frames for calibration
            while current_frame < 100:

                if(current_frame % 10 == 0):
                    self.game_communicator.send_to_game(f"{current_frame}")

                ret, frame = cap.read()
                faces = self.detector(frame)

                for box, landmarks, score in faces:
                    if score < 0.95:
                        continue

                    current_frame += 1
                    # Extract bounding box coordinates
                    x_min = int(box[0])
                    y_min = int(box[1])
                    x_max = int(box[2])
                    y_max = int(box[3])
                    bbox_width = abs(x_max - x_min)
                    bbox_height = abs(y_max - y_min)

                    # Expand the bounding box
                    x_min = max(0, x_min-int(0.2*bbox_height))
                    y_min = max(0, y_min-int(0.2*bbox_width))
                    x_max = x_max+int(0.2*bbox_height)
                    y_max = y_max+int(0.2*bbox_width)

                    # Extract the face region
                    img = frame[y_min:y_max, x_min:x_max]
                    img = Image.fromarray(img)
                    img = img.convert('RGB')
                    img = self.transformations(img)

                    # Convert image to tensor and move to the device
                    img = torch.Tensor(img[None, :]).to(self.device)

                    # Predict head pose angles using the model
                    R_pred = self.model(img)

                    # Compute Euler angles from rotation matrices
                    euler = utils.compute_euler_angles_from_rotation_matrices(
                        R_pred)*180/np.pi
                    p_pred_deg = euler[:, 0].cpu()
                    y_pred_deg = euler[:, 1].cpu()
                    r_pred_deg = euler[:, 2].cpu()

                    # Accumulate angles for calibration
                    self.zero_pitch += p_pred_deg
                    self.zero_roll += r_pred_deg
                    self.zero_yaw += y_pred_deg

        # Compute the average angles for calibration
        self.zero_pitch /= current_frame
        self.zero_roll /= current_frame
        self.zero_yaw /= current_frame

        print(self.zero_pitch, self.zero_roll, self.zero_yaw)



    def play(self):

        """
        Continuously track head pose and update the game_communicator with direction commands.
        """

        # Open the camera for capturing video frames
        cap = cv2.VideoCapture(self.cam_id)

        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        with torch.no_grad():
            # Continuously capture and process frames for head pose tracking
            while True:
                ret, frame = cap.read()
                faces = self.detector(frame)

                for box, landmarks, score in faces:
                    if score < 0.95:
                        continue
                    
                    # Extract bounding box coordinates
                    x_min = int(box[0])
                    y_min = int(box[1])
                    x_max = int(box[2])
                    y_max = int(box[3])
                    bbox_width = abs(x_max - x_min)
                    bbox_height = abs(y_max - y_min)

                    # Expand the bounding box
                    x_min = max(0, x_min-int(0.2*bbox_height))
                    y_min = max(0, y_min-int(0.2*bbox_width))
                    x_max = x_max+int(0.2*bbox_height)
                    y_max = y_max+int(0.2*bbox_width)

                    # Extract the face region
                    img = frame[y_min:y_max, x_min:x_max]
                    img = Image.fromarray(img)
                    img = img.convert('RGB')
                    img = self.transformations(img)

                    # Convert image to tensor and move to the device
                    img = torch.Tensor(img[None, :]).to(self.device)

                    # Predict head pose angles using the model
                    R_pred = self.model(img)

                    # Compute Euler angles from rotation matrices
                    euler = utils.compute_euler_angles_from_rotation_matrices(
                        R_pred)*180/np.pi
                    p_pred_deg = euler[:, 0].cpu()
                    y_pred_deg = euler[:, 1].cpu()
                    r_pred_deg = euler[:, 2].cpu()
                    
                    # Update the game_communicator with the new directions
                    self.update_game_communicator(p_pred_deg, y_pred_deg)



    def update_game_communicator(self, p_pred_deg, y_pred_deg):

        """
        Update game_communicator with direction commands based on head pose angles.

        Args:
        - p_pred_deg (float): Predicted pitch angle in degrees.
        - y_pred_deg (float): Predicted yaw angle in degrees.
        """
        
        # Initialize directions as "CENTER"
        vertical_direction = "CENTER"
        horizontal_direction = "CENTER"

        # Check if the pitch angle is above a threshold
        if p_pred_deg > (self.zero_pitch + 10):
            vertical_direction = "UP"
        elif p_pred_deg < (self.zero_pitch - 10):
            vertical_direction = "DOWN"

        # Check if the yaw angle is above a threshold
        if y_pred_deg > (self.zero_yaw + 10):
            horizontal_direction = "RIGHT"
        elif y_pred_deg < (self.zero_yaw - 10):
            horizontal_direction = "LEFT"

        # Send direction commands to the game_communicator
        if(self.game_communicator): 
            self.game_communicator.send_to_game(f"game_command:{vertical_direction}:{horizontal_direction}")
