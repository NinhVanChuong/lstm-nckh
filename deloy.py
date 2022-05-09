import time

import cv2
import mediapipe as mp
import pandas as pd
import threading
import numpy as np
import tensorflow as tf
import keyBoadControlMakeData as kC

# Khởi tạo thư viện mediapipe
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils

label='Unknow'
frames_list=[]
model = tf.keras.models.load_model("model.h5")

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


def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (0, 255, 0)
    thickness = 2
    lineType = 2
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img


def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    results = model.predict(lm_list)
    # print(results[0])
    print(results)
    if results[0][0] > 0.9:
        label = "di"
    elif results[0][1] >0.9:
        label = "chay"
    else:
        label="Unknow"
    return label

kC.me.send_rc_control(0, 0, 25, 0)
time.sleep(5)
kC.me.send_rc_control(0, 0, 0, 0)
n_time_steps=10
# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while True:
        frame = kC.camera()
        # Make detections
        try:
            image, results = mediapipe_detection(frame, holistic)
            frame_list = make_landmark_timestep(results)
            frames_list.append(frame_list)
            # Draw landmarks
            if len(frames_list) == n_time_steps:
                # predict
                t1 = threading.Thread(target=detect, args=(model, frames_list,))
                t1.start()
                frames_list = []
            # Draw landmarks
            draw_styled_landmarks(image, results)
            img = draw_class_on_image(label, image)
            # Show to screen
            cv2.imshow('OpenCV Feed', image)

            # Break gracefull
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except:
            # cv2.imshow('OpenCV Feed2', image)
            pass
        # value = kC.getKeyboardInput()

cv2.destroyAllWindows()
kC.me.land()