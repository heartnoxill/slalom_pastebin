from __future__ import print_function
import cv2
import numpy as np
import statistics
import sys
import rospy
from std_msgs.msg import String,Float64,Int64
import roslib
roslib.load_manifest('usb_cam')
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import time
bridge = CvBridge()

def initial():
    rospy.Subscriber("/usb_cam/image_raw",Image,Getcam_sign)

def Getcam_sign(data):
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    h, w = None, None

    path = "/home/slalom/catkin_ws/src/ros_basics_tutorials/src/"
    with open(path+'traffic/classes.names') as f:
        labels = [line.strip() for line in f]

    network = cv2.dnn.readNetFromDarknet(path+'traffic/yolov3-traffic.cfg',
                                        path+'traffic/yolov3-traffic_final.weights')

    layers_names_all = network.getLayerNames()
    layers_names_output = \
        [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]

    probability_minimum = 0.5
    threshold = 0.3

    colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    if w is None or h is None:
        h, w = cv_image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv_image, 1 / 255.0, (416, 416),
                                    swapRB=True, crop=False)
    network.setInput(blob) 
    start = time.time()
    output_from_network = network.forward(layers_names_output)
    end = time.time()

    print('Current frame took {:.5f} seconds'.format(end - start))

    bounding_boxes = []
    confidences = []
    class_numbers = []

    for result in output_from_network:
        for detected_objects in result:
            scores = detected_objects[5:]
            class_current = np.argmax(scores)
            confidence_current = scores[class_current]
            if confidence_current > probability_minimum:
                box_current = detected_objects[0:4] * np.array([w, h, w, h])
                x_center, y_center, box_width, box_height = box_current
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))

                bounding_boxes.append([x_min, y_min,
                                        int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)

    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                            probability_minimum, threshold)

    if len(results) > 0:
        for i in results.flatten():
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
            colour_box_current = colours[class_numbers[i]].tolist()
            cv2.rectangle(cv_image, (x_min, y_min),
                        (x_min + box_width, y_min + box_height),
                        colour_box_current, 2)
            text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                                confidences[i])
            cv2.putText(cv_image, text_box_current, (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)

    cv2.namedWindow('YOLO v3 Real Time Detections', cv2.WINDOW_NORMAL)
    cv2.imshow('YOLO v3 Real Time Detections', cv_image)

    cv2.waitKey(3)


# camera.release()
# cv2.destroyAllWindows()

def main(args):
  rospy.init_node('image_node', anonymous=True)
  initial()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()



if __name__ == '__main__':
    main(sys.argv)