from __future__ import print_function
import sys
import statistics
import cv2
import numpy as np
import time
import math

intercept = 1
xpointbot = 0
xpointtop = 0 
xpointmid = 0
x_C = [0, 0, 0]
B1 = [0,0]
B2 = [0,0]
B3 = [0,0]
B4 = [0,0]
Size_crop = [0,0]
kernel = np.ones((3,3), np.uint8)
lower_yellow = np.uint8([18, 35, 80])
upper_yellow = np.uint8([38, 255, 255])
MODE_COLOR = 0

# Change values below this line, according to blobs number
multiply_y = [0.9,0.85,0.825,0.8,0.75,0.725,0.7] # OG: 0.9-0.8-0.7
blobs = [0,0,0,0,0,0,0]
blob_C = [0,0,0,0,0,0,0]
keypoint = [0,0,0,0,0,0,0]

def Crop_result(cv_image):
    global B1
    global B2
    global B3
    global B4
    global Size_crop
    B1[0] = int(cv_image.shape[1]*0.3125)
    B2[0] = int(cv_image.shape[1]*0.6875)
    B3[0] = int(cv_image.shape[1]*0)
    B4[0] = int(cv_image.shape[1]*1)
    B1[1] = int(cv_image.shape[0]*0.45)
    B2[1] = int(cv_image.shape[0]*0.45)
    B3[1] = int(cv_image.shape[0]*0.694)
    B4[1] = int(cv_image.shape[0]*0.694) 
    Size_crop[0] = int(cv_image.shape[1]*0.4)
    Size_crop[1] = int(cv_image.shape[0])

def Crop_to_Cal(result):
    A1 = [0 , int(result.shape[0])]
    A2 = [int(result.shape[1]*0.4) , int(result.shape[0])]
    A3 = [0, int(result.shape[0]*0.6)]
    A4 = [int(result.shape[1]*0.4) , int(result.shape[0]*0.6)]
    crop_img = result[A3[1]:A1[1], A1[0]:A2[0]]
    return crop_img

def Getcam(cv_image):
    global x_mid
    global y_bot
    global y_top
    global y_mid
    global main_Y
    
    Crop_result(cv_image)
    pts1 = np.float32([[B1], [B2], [B3], [B4]])
    pts2 = np.float32([[0, 0], [Size_crop[0], 0], [0, Size_crop[1]], [Size_crop[0], Size_crop[1]]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(cv_image, matrix, (Size_crop[0], Size_crop[1]))
    result = Crop_to_Cal(result)
    result = cv2.resize(result, (0,0), fx=2, fy=2) 
    if MODE_COLOR == 0 :
      hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
      mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    else :
      hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HLS)
      mask = cv2.inRange(hsv, lower_white, upper_white)
    crop = cv2.bitwise_and(result, result, mask=mask)
    crop = cv2.morphologyEx(crop, cv2.MORPH_OPEN, kernel)
    crop = cv2.dilate(crop,kernel,iterations = 1)
    crop = cv2.morphologyEx(crop, cv2.MORPH_CLOSE, kernel)
    
    x_mid = int(result.shape[1]*0.5)
    y_bot = int(result.shape[0]*(multiply_y[0]+0.025))
    y_top = int(result.shape[0]*(multiply_y[6]+0.025))
    y_mid = int((y_bot+y_top)/2)
    main_Y = [result.shape[0]*(multiply_y[0]+0.025),
                result.shape[0]*(multiply_y[3]+0.025),
                result.shape[0]*(multiply_y[6]+0.025)]
    return result,crop

def Find_slope(point_X, point_Y, y_1, y_2):
    global slope
    global intercept
    global xpointbot
    global xpointtop
    global xpointmid
    slope = 0
    sum_1 = 0
    sum_2 = 0    
    y_mid = (y_1+y_2)/2
    x_bar = statistics.mean(point_X)
    y_bar = statistics.mean(point_Y)
    for i in range(3):
        sum_1 = sum_1 + ((point_X[i]-x_bar) * (point_Y[i]-y_bar))
        sum_2 = sum_2 + ((point_X[i]-x_bar)**2)
    if sum_2 != 0:
        slope = sum_1/sum_2
    if slope != 0:
        intercept = y_bar - (slope*x_bar)
        xpointbot = (y_1-intercept)/slope
        xpointtop = (y_2-intercept)/slope
        xpointmid = (y_mid-intercept)/slope
    return xpointbot, xpointtop, xpointmid

def bloblane(image, y_point,minblobs):
    if minblobs == 0:
        minblobs  = image.shape[1]*0.08     # 0 is height, 1 is width
    xp = int(image.shape[1])
    yp1 = int(image.shape[0] * y_point)
    yp2 = int(image.shape[0] * (y_point+0.05))
    crop = image[yp1:yp2, 0:xp]
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    params = cv2.SimpleBlobDetector_Params()
    params.filterByInertia = False
    params.filterByConvexity = False
    params.filterByColor = False
    params.minArea = minblobs
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(crop)
    blank = np.zeros((1, 1))
    blobs = cv2.drawKeypoints(crop, keypoints, blank, (0, 0, 255),
    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return blobs, keypoints

def Detect_Curve(result,blob_x):   
    global x_C
    x_C[0], x_C[1], x_C[2] = Find_slope(blob_C, main_Y, y_bot, y_top,)    # xpointbot, xpointtop, xpointmid.
    cv2.line(result, (int(x_C[2]), y_mid),(int(result.shape[1]*0.5), y_mid), (180, 0, 255), 1,  8)  # xpointmid, pink the left-right line    
    cv2.line(result, (int(x_C[0]),int(y_bot)),(int(x_C[1]), int(y_top)), (20,255,150), 2,  8)   # xpointtop bright green
    Servo_mortor = ((x_C[1]-x_C[0])*0.3)    # xpointtop - xpointbot
    return Servo_mortor

def Check_LeftRight(key_point,x):
    global blob_C
    global Catch
    if len(keypoint[x]) == 1: 
        blob_C[x] = keypoint[x][0].pt[0]  #will be used in Find_Slope
    else:
        pass

def Cal_Servo(result,mid_point):
    Percent = result.shape[1]/100.0
    Servo_mortor = mid_point/Percent
    return Servo_mortor

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    result, crop = Getcam(frame)

    cv2.line(result, (int(result.shape[1]*0.5), int(y_bot)),(int(result.shape[1]*0.5), int(y_top)), (0, 0, 255), 1,  8)  # red color

    for foo in range(0,7):
        blobs[foo], keypoint[foo] = bloblane(crop, multiply_y[foo],10)
        Check_LeftRight(keypoint,foo)

    BLOB = np.concatenate((blobs[6],blobs[5],blobs[4],blobs[3],blobs[2],blobs[1],blobs[0]),axis=0)
    Detect_Curve(result,blob_C)
    Servo_mortor = Cal_Servo(result,x_C[2])
    result = np.concatenate((result,BLOB), axis=0)

    if Servo_mortor > 100:
        Servo_mortor = 100
    elif Servo_mortor < 0:
        Servo_mortor = 0
    else:
        pass

    crop  = cv2.resize(BLOB, (0,0), fx=0.5, fy=0.5) 
    frame = cv2.resize(result, (0,0), fx=0.5, fy=0.5)
    result = cv2.resize(result, (0,0), fx=0.5, fy=0.5)  
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(Servo_mortor)
    cv2.imshow("result",result)
    k = cv2.waitKey(10) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
cap.release()
