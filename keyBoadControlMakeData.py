import keyPressMove as kp
from djitellopy import tello
import time
import cv2
kp.init()
me= tello.Tello()
me.connect()
print(me.get_battery())
def getKeyboardInput():
    lr,fb,ud,yv=0,0,0,0
    speed=50
    if kp.getKey("LEFT"):lr=-speed
    elif kp.getKey("RIGHT"):lr=speed

    if kp.getKey("UP"):fb=speed
    elif kp.getKey("DOWN"):fb=-speed

    if kp.getKey("w"):ud=speed
    elif kp.getKey("s"):ud=-speed

    if kp.getKey("a"):yv=-speed
    elif kp.getKey("d"):yv=speed

    if kp.getKey("q"):me.land()

    return [lr,fb,ud,yv]
me.streamon()
frame_width=640
frame_height=480
# out = cv2.VideoWriter('dichuyen_chuong1.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),7, (frame_width, frame_height))
def camera():
    frame= me.get_frame_read().frame
    frame = cv2.resize(frame, (640, 480))
    return frame
me.takeoff()
# while True:
#     value= getKeyboardInput()
#     me.send_rc_control(value[0],value[1],value[2],value[3])
#     frame=camera()
#     out.write(frame)
#     cv2.imshow("Image", frame)
#     if cv2.waitKey(1) == ord('q'):
#         break
# me.streamoff()