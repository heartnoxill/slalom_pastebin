import cv2
import os
import time

# Read the video from specified path
cam = cv2.VideoCapture(0)

try:
    path = "C:/Users/Heart/Desktop/screenshot11/"
    # creating a folder named data
    if not os.path.exists(path+'data'):
        os.makedirs(path+'data')

    # if not created then raise error
except OSError:
    print('Error: Creating directory of data')

# frame
currentframe = 0

while (True):
    time.sleep(0.1) # take schreenshot every NUMBER seconds
    # reading from frame
    ret, frame = cam.read()

    if ret:
        # if video is still left continue creating images
        name = 'C:/Users/Heart/Desktop/screenshot11/data/frame' + str(currentframe) + '.jpg'
        print('Creating...' + name)

        # writing the extracted images
        cv2.imwrite(name, frame)

        # increasing counter so that it will
        # show how many frames are created
        currentframe += 1
    else:
        break

# Release all space and windows once done
cam.release()
cv2.destroyAllWindows()