import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

mpDraw = mp.solutions.drawing_utils

def draw_line(img,a,b):
    cv2.line(img, a, b, (0, 0, 0), thickness=2)
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
        cv2.circle(img, (cx, cy), 3, (0, 0, 255), cv2.FILLED)

    #Draw connection
    S={0, 1, 2, 4, 5, 9, 11, 23}
    S1={3, 0, 15, 16, 27, 28}
    S2={6, 11, 13, 15, 17, 12, 14, 16, 18}#, 23, 24, 25, 26, 27, 28, 29, 30} #vì chỉ lấy đến điểm 24
    S3={15, 16}
    S4={11, 12}

    for i in range(len(adu)):
        if i+11 in S:
            draw_line(img, adu[i], adu[i+1])
        if i+11 in S1:
            draw_line(img, adu[i], adu[i+4])
        if i+11 in S2:
            draw_line(img, adu[i], adu[i + 2])
        if i+11 in S3:
            draw_line(img, adu[i], adu[i + 6])
        # if i+11 in S4:
        #     draw_line(img, adu[i], adu[i + 12])
    return img

chay1_df = pd.read_csv("Data1/data_di.txt")
chay2_df = pd.read_csv("Data1/data_di2.txt")
# chay3_df = pd.read_csv("Data1/data_chay3.txt")
label=f"Data_train/di"
f_write=True  #True: cho phép ghi dữ liệu vào filel

dataset=[]
dataset.append(chay1_df.iloc[:,1:].values)
dataset.append(chay2_df.iloc[:,1:].values)
# dataset.append(chay3_df.iloc[:,1:].values)


j=0
count=0
list_frame=[]

frame = np.ones((480, 640, 3), np.uint8) * 255
for dataset1 in dataset:
    i = 0
    n_sample = len(dataset1)
    print(n_sample)
    while i<n_sample:
        frame = np.ones((480, 640, 3), np.uint8) * 255
        # ret, frame = cap.read()
        # frame=cv2.flip(frame,1)
        frame = draw_landmark_on_image(mpDraw, dataset1[i], frame)
        cv2.putText(frame,str(i),(30,50),4,2,(0,255,0),2);
        cv2.putText(frame, str(count), (30, 100), 4, 2, (0, 0,0), 2);
        cv2.imshow("image", frame)
        # print(i)
        key = cv2.waitKeyEx(1)  # waitKey(300)
        lm=[]
        if key == ord('j'): #left arrow
            if i>0:
                i=i-1
        elif key == ord('l'):  #right arrow
            if i < n_sample: i = i + 1
        elif key==ord(' '):
            if i+10> len(dataset1):
                break
            for j in range (10):
                list_frame.append(dataset1[i])
                i=i+1
            count +=1
        elif key==ord('q'):
            break
        if i > n_sample:
            break
if f_write:
    df = pd.DataFrame(list_frame)
    df.to_csv(label + ".txt")