from Modules.emotion_recognition.emotion_cnn import EmotionCNN
import os
import cv2
import random
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN
from torch.utils.data import random_split, ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm
import dlib
from PIL import Image


class EmotionRecognizer:

    def __init__(self, game_communicator):

        self.game_communicator = game_communicator
        
        self.label_mapping = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
        self.model = EmotionCNN(len(self.label_mapping))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "Models", "emotion_recognition_model.pth"), map_location=self.device))
        self.model.to(self.device)
        self.model.eval() 

        # initialize the face detector
        self.detector = dlib.get_frontal_face_detector()


    def recognize(self, ret, frame):        

        # apply the transformations to the face image
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # faces detection
        faces = self.detector(frame)

        # if there is at least one face detected, process the image
        if len(faces) == 0:
            self.game_communicator.send_to_game("emotion:Neutral")
            return
        
        # take only the first face
        face = faces[0]
                
        # cut the face from the frame
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_image = frame[y:y+h, x:x+w]

        # check if the face image is not empty
        if not face_image.size == 0:
            # apply the transformations to the face image
            pil_image = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
            input_image = transform(pil_image).unsqueeze(0)  # Aggiunge una dimensione di batch
            input_image = input_image.to(self.device)

            # model prediction
            with torch.no_grad():
                output = self.model(input_image)

                # get the label predicted by the model
                _, predicted = torch.max(output, 1)
                predicted_emotion = self.label_mapping[predicted.item()]
                self.game_communicator.send_to_game(f"emotion:{predicted_emotion}")