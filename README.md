# Expression Odyssey

We have chosen this name for **our** project to represent both the **difficulties** that we faced during the **development** and the **challenges** that the user can **experience** while playing.

# Tasks
The project is composed **of** 5 different sub-tasks:
- Face Recognition for user registration and login.
- Emotion Recognition for **adjusting** difficulty.
- Gaze Tracking for controlling the character.
- Head Pose Tracking for providing a second choice for character control.
- Game Development

## Face Recognition
The face detection and recognition part has been implemented using the standard and well-known library dlib. It provides two pre-trained models for both detection and recognition. The registration consists **of saving** a video of the user's face from the webcam. Every frame is processed by applying some computer vision techniques to reduce overall noise and **using** the dlib's face detector. The frames considered as valid will be saved to build the user's dataset. To recognize the user, a set of features for every user is built and then compared in real time with the face detected in the webcam. The face is associated with the user with the highest number of matches.

## Emotion Recognition
**TODO**

### Related Research Papers
**TODO**

### Dataset
**TODO**

## Gaze Tracking
The game has been designed for **people who have partial or total paralysis**. Therefore, we decided to search on [Papers With Code](https://paperswithcode.com/) for some academic articles related **to** this topic. However, only 3-4 papers have been found, and the related codes were using old libraries. Therefore, we focused mostly on [GitHub](https://github.com/) to find valid repositories that could be used as a starting point. We found this project [Gaze Tracking by antoinelame](https://github.com/antoinelame/GazeTracking); it uses standard computer vision methods (such as binary thresholding and kernel analysis) mixed with the landmark predictor provided by the dlib library. Since there are no deep learning models or large networks involved with this approach, it has good performance with good accuracy too. However, the provided code can be used only for tracking:
- Blinking
- Right Looking
- Left Looking

We don't really need blinking detection, but we need Upward Looking and Downward Looking. Moreover, during the first tests of this repository, we noticed that the tracking is not perfect since even with a small amount of noise, the tracking was not performed in the correct way. Our contribution consists **of** adding the missing features needed in the game and applying some computer vision techniques to improve the robustness against noise.

## Head Pose Tracking
To provide a different way to control the character, we decided to implement a head tracker. **TODO**

## Game Development
The game has been developed using Unity; it is a Temple Run-like game. The player can move from left to right and slide/jump trying to dodge the obstacles. Communication with the Python part has been implemented through the client/server paradigm. In detail, the Python code **acts** as a server that analyzes the webcam, computing the movement and difficulty adjustments. All this information is sent to the client (game) using a socket.