import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

mpDraw = mp.solutions.drawing_utils

def draw_line(img,a,b):
    cv2.line(img, a, b, (255, 255, 255), thickness=1)
    return img
def draw_landmark_on_image(mpDraw, results, img):
    # Vẽ các điểm nút
    adu=[]
    adus=[]
    for i in range(len(results)):
        adu.append(results[i])
        #lấy ra 4 giá trị của mỗi điểm
        if (i+1)%4==0:
            adus.append(adu)
            adu=[]
    adu=[]
    for lm in adus:
        h, w, c = img.shape
        # Lấy giá trị x,y của 1 điểm
        cx, cy = int(lm[0] * w), int(lm[1] * h)
        adu.append((cx,cy))
        cv2.circle(img, (cx, cy), 2, (0, 0, 255), cv2.FILLED)

    #Draw connection
    S={0, 1, 2, 4, 5, 9, 11, 23}
    S1={3, 0, 15, 16, 27, 28}
    S2={6, 11, 13, 15, 17, 12, 14, 16, 18, 23, 24, 25, 26, 27, 28, 29, 30} #vì chỉ lấy đến điểm 24
    S3={15, 16}
    S4={11, 12}

    for i in range(len(adu)):
        if i in S:
            draw_line(img, adu[i], adu[i+1])
        if i in S1:
            draw_line(img, adu[i], adu[i+4])
        if i in S2:
            draw_line(img, adu[i], adu[i + 2])
        if i in S3:
            draw_line(img, adu[i], adu[i + 6])
        if i in S4:
            draw_line(img, adu[i], adu[i + 12])
    return img


quaytrai_df = pd.read_csv("Data_train/chay.txt")
dataset = quaytrai_df.iloc[:,1:].values
n_sample = len(dataset)
i = 0
list_frame_di=[]
list_frame_chay=[]
label1=f"data_di"
label2=f"data_chay"

f_save_di=False
f_save_chay=False

frame = np.ones((480, 640, 3), np.uint8) * 0
print(len(dataset[1]))
while i<n_sample:
    frame = np.ones((480, 640, 3), np.uint8) * 0
    # ret, frame = cap.read()
    # frame=cv2.flip(frame,1)
    frame = draw_landmark_on_image(mpDraw, dataset[i], frame)

    cv2.putText(frame,str(i),(30,50),4,2,(0,255,0),2);
    cv2.imshow("image", frame)
    # print(i)
    key = cv2.waitKeyEx(1)  # waitKey(300)
    lm=[]
    if key == ord('j'): #left arrow
        # print(i)
        if i>0: i=i-1
    elif key == ord('l'):  #right arrow
        if i < n_sample: i = i + 1
    elif key==ord(' '):  #space
        for j in range (10):
            list_frame_di.append(dataset[i])
            i=i+1
    elif key==ord('b'):
        for j in range (10):
            list_frame_chay.append(dataset[i])
            i=i+1
    elif key==ord('q'):
        break
    if i > n_sample:
        break
if f_save_di:
    df1 = pd.DataFrame(list_frame_di)
    df1.to_csv(label1 + ".txt")
if f_save_chay:
    df2 = pd.DataFrame(list_frame_chay)
    df2.to_csv(label2 + ".txt")