import cv2
import os
import threading
import dlib
import numpy as np
import time



def face_rects(image, face_detector):

    """
    Detects faces in the input image using a face detector.

    Parameters:
    - image: Input image (numpy array).
    - face_detector: Face detector model.

    Returns:
    - rects: List of rectangles representing the detected faces.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert the image to grayscale
    rects = face_detector(gray, 1) # Detect faces in the grayscale image
    return rects



def face_landmarks(image, shape_predictor, face_detector):

    """
    Computes facial landmarks for each face in the input image.

    Parameters:
    - image: Input image (numpy array).
    - shape_predictor: Facial landmark predictor model.
    - face_detector: Face detector model.

    Returns:
    - List of facial landmarks for each detected face.
    """

    # Compute the face landmarks for each face in the image
    return [shape_predictor(image, face_rect) for face_rect in face_rects(image, face_detector)]



def face_encodings(image, face_encoder, shape_predictor, face_detector):

    """
    Computes facial encodings (features) for each face in the input image.

    Parameters:
    - image: Input image (numpy array).
    - face_encoder: Face encoding model.
    - shape_predictor: Facial landmark predictor model.
    - face_detector: Face detector model.

    Returns:
    - List of facial encodings for each detected face.
    """

    # Compute the facial embeddings for each face (128-d vector that describes the face in an image)
    return [np.array(face_encoder.compute_face_descriptor(image, face_landmark)) for face_landmark in face_landmarks(image, shape_predictor, face_detector)]



def learn_faces(face_detector, shape_predictor, face_encoder):

    """
    Learns and stores facial encodings for known faces.

    Parameters:
    - face_detector: Face detector model.
    - shape_predictor: Facial landmark predictor model.
    - face_encoder: Face encoding model.

    Returns:
    - Dictionary containing facial encodings for each known user.
    """

    known_faces = {}
    base_directory = "../UserFaces/"  # Directory containing user faces

    # Iterate through directories
    for user_name in os.listdir(base_directory):
        user_path = os.path.join(base_directory, user_name)

        # Iterate through face images in each user directory
        for filename in os.listdir(user_path):
            image_path = os.path.join(user_path, filename)

            img = cv2.imread(image_path)  # Read the image
            new_encodings = face_encodings(img, face_encoder, shape_predictor, face_detector) # Get the embeddings

            encodings = known_faces.get(user_name, [])
            # Add the embeddings to the already saved ones
            encodings.extend(new_encodings)
            known_faces[user_name] = encodings

    print(known_faces.keys())

    return known_faces



def nb_of_matches(known_encodings, unknown_encoding):

    """
    Calculates the number of matches between the unknown face encoding and the encodings in the database.

    Parameters:
    - known_encodings: Dictionary containing facial encodings for each known user.
    - unknown_encoding: Facial encoding for the unknown face.

    Returns:
    - Number of matches (faces with a distance less than or equal to a threshold) for each known user.
    """

    distances = np.linalg.norm(known_encodings - unknown_encoding, axis=1)
    small_distances = distances <= 0.8  # Keep only the distances that are less than the threshold
    return sum(small_distances)



def set_stop_thread():

    """
    Thread function to set the stop_thread variable to True after 10 seconds.
    """

    global stop_thread
    time.sleep(10)  # Wait for 10 seconds
    with lock:
        stop_thread = True



def face_recognition(known_faces: dict, face_detector, shape_predictor, face_encoder):

    """
    Performs real-time face recognition using the webcam.

    Returns:
    - Recognized username.
    """

    username = "Unknown"

    cap = cv2.VideoCapture(0)  # Open a connection to the webcam

    global stop_thread
    stop_thread = False

    global lock
    lock = threading.Lock()

    # Start the thread that sets the stop_thread variable after 10 seconds
    stop_thread_thread = threading.Thread(target=set_stop_thread)
    stop_thread_thread.start()

    while True:
        with lock:
            # Check if the stop_thread variable is set to True
            if stop_thread:
                break

        ret, frame = cap.read()  # Read a frame from the webcam

        # Get the face encodings of the unknown face
        frame_encodings = face_encodings(frame, face_detector=face_detector, face_encoder=face_encoder, shape_predictor=shape_predictor)

        for encoding in frame_encodings:
            counts = {}

            for (name, known_encodings) in known_faces.items():
                # Compare the encodings between every face in the user dataset and the current one
                counts[name] = nb_of_matches(known_encodings, encoding)

            if all(count == 0 for count in counts.values()):
                # If there are no matches, the user is unknown
                username = "Unknown"
            else:
                # Pick the user with the highest number of matches
                username = max(counts, key=counts.get)
                return username

    # Release the webcam and close all windows
    cap.release()
    stop_thread_thread.join()  # Wait for the stop_thread thread to finish

    return username