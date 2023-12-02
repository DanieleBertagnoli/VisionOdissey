import cv2
import threading
from gaze_tracking import GazeTracking
import time
import sounddevice as sd
import numpy as np

def calibrate_thresholds(gaze, webcam, stop_thread):
    index = 0
    saved = True
    text_directions = ["Look to the right", "Look to the left", "Look up", "Look down"]

    def update_index():
        nonlocal index
        nonlocal saved
        while not stop_thread.is_set():
            time.sleep(1)
            saved = False
            time.sleep(2)
            index = index + 1
            beep()

    def beep():
        # Frequency and duration for the beep sound
        frequency = 1000  # Adjust as needed
        duration = 200  # Adjust as needed
        sd.play(frequency * np.sin(2 * np.pi * np.arange(44100 * duration / 1000) * frequency / 44100), samplerate=44100)
        sd.wait()

    stop_thread.clear()  # Clear the stop_thread flag
    index_thread = threading.Thread(target=update_index, daemon=True)
    index_thread.start()

    while not stop_thread.is_set():
        _, frame = webcam.read()
        gaze.refresh(frame)
        frame = gaze.annotated_frame()

        if gaze.pupils_located and not saved:
            gaze.set_threshold(index)
            saved = True

        if index > 3:
            break 

        text = text_directions[index]
        cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) == 27:
            break

    stop_thread.set()  # Set the stop_thread flag to stop the update_index thread
    index_thread.join()  # Wait for the thread to finish

    cv2.destroyAllWindows()


gaze = GazeTracking()
webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

stop_thread = threading.Event()  # Event to signal the thread to stop
calibrate_thresholds(gaze, webcam, stop_thread)

# Variables to track gaze direction
up_counter = down_counter = left_counter = right_counter = 0

# Flags to indicate gaze direction
looking_up = looking_down = looking_left = looking_right = False

while True:
    _, frame = webcam.read()
    gaze.refresh(frame)
    frame = gaze.annotated_frame()
    text = ""

    # Check gaze direction and update counters
    if gaze.is_right():
        right_counter += 1
        left_counter = up_counter = down_counter = 0
    elif gaze.is_left():
        left_counter += 1
        right_counter = up_counter = down_counter = 0
    else:
        left_counter = right_counter = 0

    if gaze.is_up():
        up_counter += 1
        down_counter = left_counter = right_counter = 0
    elif gaze.is_down():
        down_counter += 1
        up_counter = left_counter = right_counter = 0
    else:
        up_counter = down_counter = 0

    # Check if gaze direction has been held for 10 frames
    if up_counter == 10:
        looking_up = True
        print("Sending UP")
    else:
        looking_up = False

    if down_counter == 10:
        looking_down = True
        print("Sending DOWN")
    else:
        looking_down = False

    if right_counter == 10:
        looking_right = True
        print("Sending RIGHT")
    else:
        looking_right = False

    if left_counter == 10:
        looking_left = True
        print("Sending LEFT")
    else:
        looking_left = False

    if gaze.is_right():
        text = "right"
    elif gaze.is_left():
        text = "left"
    else:
        text = "center"
    
    if gaze.is_up():
        text = text + " up"
    elif gaze.is_down():
        text = text + " down"
    else:
        text = text + " center"

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    cv2.imshow("Demo", frame)

    if cv2.waitKey(1) == 27:
        break

webcam.release()
cv2.destroyAllWindows()
