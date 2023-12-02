import cv2
import threading
from gaze_tracking import GazeTracking
import time

def calibrate_thresholds(gaze, webcam):
    index = 0
    saved = False
    text_directions = ["Look to the right", "Look to the left", "Look up", "Look down"]

    def update_index():
        nonlocal index
        while True:
            time.sleep(2)
            index = (index + 1) % len(text_directions)
            saved = False

    index_thread = threading.Thread(target=update_index, daemon=True)
    index_thread.start()

    while True:
        _, frame = webcam.read()
        gaze.refresh(frame)
        frame = gaze.annotated_frame()

        if(gaze.pupils_located and not saved):
            gaze.set_threshold(index)
            saved = True

        text = text_directions[index]
        cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

calibrate_thresholds(gaze, webcam)

while True:
    _, frame = webcam.read()
    gaze.refresh(frame)
    frame = gaze.annotated_frame()
    text = ""

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
