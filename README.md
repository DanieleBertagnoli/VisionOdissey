# Vision Odyssey

We have chosen this name for our project to represent both the difficulties that we faced during the development and the challenges that the user can experience while playing.

# Tasks
The project is composed of 5 different sub-tasks:
1. Face Recognition for user registration and login.
2. Emotion Recognition for adjusting difficulty.
3. Gaze Tracking for controlling the character.
4. Head Pose Tracking for providing a second choice for character control.
5. Game Development

## Face Recognition
The face detection and recognition part has been implemented using the standard and well-known library dlib. It provides two pre-trained models for both detection and recognition. The registration consists of saving a video of the user's face from the webcam. Every frame is processed by applying some computer vision techniques to reduce overall noise and using the dlib's face detector. The frames considered as valid will be saved to build the user's dataset. To recognize the user, a set of features for every user is built and then compared in real time with the face detected in the webcam. The face is associated with the user with the highest number of matches.

## Emotion Recognition
With emotion recognition, we aim to determine the player's emotional state based on their facial expressions, so that the difficulty level of the game can be adjusted accordingly. We believe that modifying difficulty based on the emotion conveyed by the player's face poses a significant challenge, and we are aware, at the same time, that it is not a universal metric for such a purpose.

### Related Research Papers and Dataset
The dataset we utilized is FER-2013: https://www.kaggle.com/datasets/msambare/fer2013. A majority of state-of-the-art (SOTA) projects employ Convolutional Neural Network (CNN) architecture. We drew inspiration from some of these projects, particularly from the following: https://github.com/serengil/deepface, https://ieeexplore.ieee.org/document/9659697. The architecture is inspired by this project. A second dataset was exclusively used for the testing phase: https://www.kaggle.com/datasets/sudarshanvaidya/random-images-for-face-emotion-recognition. The emotions included in the datasets are Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.

The first phase involves pre-processing composed of:
- Dataset augmentation: each image is multiplied by applying the following transformations. Horizontal flip, translation, zoom, contrast and brightness modification, elastic transformation;
- Outliers removal: outliers are removed using DBSCAN clustering algorithm;
- Class balancing: for each class there are the same number of instances;
- Resize (48x48),  conversion to tensor and normalization.

Subsequently, the training phase follows with the specified architecture.
![](https://github.com/DanieleBertagnoli/ComputerVisionProject/blob/main/Plot/cnn.png)

Finally, there is the testing phase using the test set from the training dataset and a dataset dedicated exclusively to the testing phase.
These are the results related to the two test sets. The metrics considered include recall and precision.

Accuracy 70,84 %:
![](https://github.com/DanieleBertagnoli/ComputerVisionProject/blob/main/Plot/plot1.png)

Accuracy 41,67%:
![](https://github.com/DanieleBertagnoli/ComputerVisionProject/blob/main/Plot/plot2.png)


## Gaze Tracking
The game has been designed for individuals with partial or total paralysis. Therefore, we searched on [Papers With Code](https://paperswithcode.com/) for academic articles related to this topic. However, we found only 3-4 papers, and the related codes were using outdated libraries. Only one paper provided a feasible starting point for our project: [Appearance-Based Gaze Estimation in the Wild](https://arxiv.org/pdf/1504.02863.pdf). It can be found on both IEEE and arXiv with around 1k citations. Despite being published in 2015, it remains innovative and suitable for our purposes. In detail, it utilizes a very large dataset combined with a CNN to perform gaze tracking (more details can be found in the linked paper).

However, the available [repository on GitHub](https://github.com/hysts/pytorch_mpiigaze_demo) was using an older Python version. Therefore, the first step of our contribution involved error correction and updating dependencies. Fortunately, the repository contained a demo file that served as a starting point for our work. We created a Python file (`Notebooks/Modules/gaze_tracker.py`) implementing a class that could retrieve all the necessary information from the pre-trained model.

Since the model returned yaw and pitch values (used to determine gaze direction), we also implemented a calibration phase. During this step, the user looks at the center of the screen, and the system calculates the average pitch and yaw. This provides a relative origin from which we can apply a threshold to determine if the user is looking right/left/center and up/down/center. Thanks to this mechanism we can guarantee that even if the player is not perfectly aligned with the camera, the input system will work.


## Head Pose Tracking
To provide a different way to control the character, we decided to implement a head tracker. For this kind of topic, many papers can be found on [Papers With Code](https://paperswithcode.com/), so we chose one of the most recent papers, considering the implementation characteristics as well. [6D Rotation Representation For Unconstrained Head Pose Estimation](https://arxiv.org/abs/2202.12555) is a paper published in 2022 on IEEE. It focuses on predicting head pose without considering facial landmarks. This is an innovative technique since it promises good results even if the face is partially or totally occluded (e.g., the player is rotated by 90 degrees with respect to the camera). The technique is based on a ResVGG deep-neural network that predicts yaw and pitch (further details can be found in the paper above).

Similar to the gaze tracker, the code associated with this paper had several problems due to incompatibilities with the newer Python version. After fixing all the issues, we implemented our standard Python file to access the pre-trained model and retrieve the information (`Notebooks/Modules/head_pose_tracker.py`). As with the previous method, a calibration step is implemented to be camera-independent.

## Game Development
The game has been developed using Unity; it is a Temple Run-like game. The player can move from left to right and slide/jump trying to dodge the obstacles. Communication with the Python part has been implemented through the client/server paradigm. In detail, the Python code acts as a server that analyzes the webcam, computing the movement and difficulty adjustments. All this information is sent to the client (game) using a socket.