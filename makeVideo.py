from djitellopy import tello
from time import sleep
import keyBoadControlMakeData as kC
import cv2
out = cv2.VideoWriter('dat.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))
while True:
    frame=kC.camera()
    out.write(frame)
    cv2.imshow("Image", frame)
    if cv2.waitKey(1) == ord('q'):
        break
