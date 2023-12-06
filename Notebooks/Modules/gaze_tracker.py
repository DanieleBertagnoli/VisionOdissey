from Modules.ptgaze.common import Face, FacePartsName
from Modules.ptgaze.gaze_estimator import GazeEstimator
from omegaconf import DictConfig, OmegaConf
import os

import cv2
import numpy as np

class GazeTracker:

    def __init__(self, game_communicator):

        self.zero_pitch = 0
        self.zero_yaw = 0
        self.zero_roll = 0
        self.game_communicator = game_communicator

        config_dict = {
            'mode': 'ETH-XGaze',
            'device': 'cpu',
            'model': {
                'name': 'resnet18'
            },
            'face_detector': {
                'mode': 'mediapipe',
                'mediapipe_max_num_faces': 3,
                'mediapipe_static_image_mode': False
            },
            'gaze_estimator': {
                'checkpoint': os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Models', 'eth-xgaze_resnet18.pth'),
                'camera_params': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ptgaze/data/calib/sample_params.yaml'),
                'use_dummy_camera_params': False,
                'normalized_camera_params': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ptgaze/data/normalized_camera_params/eth-xgaze.yaml'),
                'normalized_camera_distance': 0.6,
                'image_size': [224, 224]
            }
        }
        config = OmegaConf.create(config_dict)

        self.gaze_estimator = GazeEstimator(config)

    def calibrate(self):

        cap = cv2.VideoCapture(0)
        current_frame_index = 0

        while current_frame_index < 100:
            
            if cv2.waitKey(1) == 27:
                break

            ret, frame = cap.read()

            _, current_frame_index = self.process_frame(frame, current_frame_index, True)
        
        cap.release()
        cv2.destroyAllWindows()

        self.zero_pitch /= current_frame_index
        self.zero_yaw /= current_frame_index

    def play(self):

        cap = cv2.VideoCapture(0)

        while True:
            
            if cv2.waitKey(1) == 27:
                break

            ret, frame = cap.read()

            __, _ = self.process_frame(frame, None)
        
        cap.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame, current_frame_index, calibration=False):

        camera_matrix = [640., 0., 320.,
           0., 640., 240.,
           0., 0., 1.]
        dist_coefficients = [0., 0., 0., 0., 0.]

        undistorted = cv2.undistort(frame, np.array(camera_matrix).reshape(3, 3), np.array(dist_coefficients).reshape(-1, 1))
        faces = self.gaze_estimator.detect_faces(undistorted)

        if calibration and len(faces) > 0:
            current_frame_index += 1

        for face in faces:
        
            self.gaze_estimator.estimate_gaze(undistorted, face)
            frame = self.draw(frame, face)
            p_pred_deg, y_pred_deg = np.rad2deg(face.vector_to_angle(face.gaze_vector))
            
            if not calibration:
                self.update_game_communicator(p_pred_deg, y_pred_deg)
            else:
                self.zero_pitch += p_pred_deg
                self.zero_yaw += y_pred_deg
        
        return (frame, current_frame_index)
        
    def draw(self, frame, face):

        camera_matrix = [640., 0., 320.,
           0., 640., 240.,
           0., 0., 1.]
        dist_coefficients = [0., 0., 0., 0., 0.]
        length = 0.05

        point_0 = face.center
        point_1 = face.center + length * face.gaze_vector

        points3d = np.vstack([point_0, point_1])
        points2d, _ = cv2.projectPoints(points3d, np.zeros(3, dtype=float), np.zeros(3, dtype=float),np.array(camera_matrix).reshape(3, 3), np.array(dist_coefficients).reshape(-1, 1))
        points2d = points2d.reshape(-1, 2)

        pt0 = self.convert_pt(points2d[0])
        pt1 = self.convert_pt(points2d[1])
        frame = cv2.line(frame, pt0, pt1, (255, 255, 0), 1, cv2.LINE_AA)

        bbox = np.round(face.bbox).astype(np.int64).tolist()
        frame = cv2.rectangle(frame, tuple(bbox[0]), tuple(bbox[1]), (0, 255, 0), 1)

        return frame

    def convert_pt(self, point: np.ndarray):
        return tuple(np.round(point).astype(np.int64).tolist())

    
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