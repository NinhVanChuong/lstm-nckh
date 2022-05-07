import mediapipe as mp
import pandas as pd
from djitellopy import tello
import keyBoadControlMakeData as kC
import cv2
# Khởi tạo thư viện mediapipe
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils

label='test'
frames_list=[]

def make_landmark_timestep(results):
    c_lm = []
    for lm in results.pose_landmarks.landmark:
            c_lm.append(lm.x)
            c_lm.append(lm.y)
            c_lm.append(lm.z)
            c_lm.append(lm.visibility)
    return c_lm


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections

def draw_styled_landmarks(image, results):
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             )



# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3) as holistic:
    while True:
        frame=kC.camera()
        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        try:
            frame_list=make_landmark_timestep(results)
            frames_list.append(frame_list)
            # Draw landmarks
            draw_styled_landmarks(image, results)

            # Show to screen
            cv2.imshow('OpenCV Feed', image)
        except:
            cv2.imshow('OpenCV Feed', image)
            pass
        value=kC.getKeyboardInput()
        kC.me.send_rc_control(value[0], value[1], value[2], value[3])
        # Break gracefull
        if cv2.waitKey(1) & 0xFF == ord('b'):
            break
cv2.destroyAllWindows()
# Write vào file csv
df = pd.DataFrame(frames_list)
df.to_csv(label + ".txt")
