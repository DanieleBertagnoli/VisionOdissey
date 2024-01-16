from Modules.ptgaze.common import Face, FacePartsName
from Modules.ptgaze.gaze_estimator import GazeEstimator
from omegaconf import DictConfig, OmegaConf
import os

import cv2
import numpy as np

class GazeTracker:
    """
    GazeTracker class is responsible for estimating gaze direction and communicating with a game using the provided game communicator.

    Attributes:
    - zero_pitch (float): Initial pitch angle calibration value.
    - zero_yaw (float): Initial yaw angle calibration value.
    - zero_roll (float): Initial roll angle calibration value (not currently used).
    - game_communicator: Object responsible for communicating with the game.
    - gaze_estimator: GazeEstimator instance for estimating gaze direction.

    Methods:
    - __init__(self, game_communicator): Constructor method to initialize the GazeTracker object.
    - calibrate(self): Perform gaze calibration by capturing frames and updating calibration angles.
    - play(self, ret, frame): Process a frame during the gameplay without calibration.
    - process_frame(self, frame, current_frame_index, calibration=False): Process a frame, detect faces, and estimate gaze direction.
    - draw(self, frame, face): Draw gaze direction and face bounding box on the frame.
    - convert_pt(self, point): Convert a 2D point from floating-point to integer format.
    - update_game_communicator(self, p_pred_deg, y_pred_deg): Update game_communicator with direction commands based on head pose angles.
    """

    def __init__(self, game_communicator):
        """
        Initializes a GazeTracker object.

        Args:
        - game_communicator: Object responsible for communicating with the game.
        """

        # Calibration angles
        self.zero_pitch = 0
        self.zero_yaw = 0
        self.zero_roll = 0

        self.game_communicator = game_communicator # Game communicator

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
        """
        Perform gaze calibration by capturing frames and updating calibration angles.
        """

        cap = cv2.VideoCapture(0)
        current_frame_index = 0

        while current_frame_index < 100:
            
            if cv2.waitKey(1) == 27:
                break

            if(current_frame_index % 10 == 0):
                self.game_communicator.send_to_game(f"{current_frame_index}")

            ret, frame = cap.read()

            _, current_frame_index = self.process_frame(frame, current_frame_index, True)
        
        cap.release()
        cv2.destroyAllWindows()

        # Calculate average calibration angles
        self.zero_pitch /= current_frame_index
        self.zero_yaw /= current_frame_index

    def play(self, ret, frame):
        """
        Process a frame during the gameplay without calibration.

        Args:
        - ret: Return value from video capture.
        - frame: Frame captured from the video source.
        """

        self.process_frame(frame, None)
        

    def process_frame(self, frame, current_frame_index, calibration=False):
        """
        Process a frame, detect faces, and estimate gaze direction.

        Args:
        - frame: Frame to be processed.
        - current_frame_index: Index of the current frame.
        - calibration (bool): Flag indicating whether calibration is being performed.

        Returns:
        - Tuple containing the processed frame and updated frame index.
        """

        # Camera parameters for undistortion
        camera_matrix = [640., 0., 320.,
           0., 640., 240.,
           0., 0., 1.]
        dist_coefficients = [0., 0., 0., 0., 0.]

        undistorted = cv2.undistort(frame, np.array(camera_matrix).reshape(3, 3), np.array(dist_coefficients).reshape(-1, 1)) # Undistort the frame
        faces = self.gaze_estimator.detect_faces(undistorted) # Detect faces in the undistorted frame

        if calibration and len(faces) > 0: # Update frame index during calibration
            current_frame_index += 1

        for face in faces: # Process each detected face
        
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
        """
        Draw gaze direction and face bounding box on the frame.

        Args:
        - frame: Frame to draw on.
        - face: Face object containing gaze information.

        Returns:
        - Modified frame with drawn elements.
        """

        # Camera parameters for drawing
        camera_matrix = [640., 0., 320.,
           0., 640., 240.,
           0., 0., 1.]
        dist_coefficients = [0., 0., 0., 0., 0.]
        length = 0.05 # Length of the gaze vector

        # Define points for drawing gaze vector
        point_0 = face.center
        point_1 = face.center + length * face.gaze_vector

        # Project 3D points to 2D for drawing
        points3d = np.vstack([point_0, point_1])
        points2d, _ = cv2.projectPoints(points3d, np.zeros(3, dtype=float), np.zeros(3, dtype=float),np.array(camera_matrix).reshape(3, 3), np.array(dist_coefficients).reshape(-1, 1))
        points2d = points2d.reshape(-1, 2)

        # Convert and draw gaze vector
        pt0 = self.convert_pt(points2d[0])
        pt1 = self.convert_pt(points2d[1])
        frame = cv2.line(frame, pt0, pt1, (255, 255, 0), 1, cv2.LINE_AA)

        # Draw face bounding box
        bbox = np.round(face.bbox).astype(np.int64).tolist()
        frame = cv2.rectangle(frame, tuple(bbox[0]), tuple(bbox[1]), (0, 255, 0), 1)

        return frame

    def convert_pt(self, point: np.ndarray):
        """
        Convert a 2D point from floating-point to integer format.

        Args:
        - point: 2D point in floating-point format.

        Returns:
        - Converted 2D point in integer format.
        """
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