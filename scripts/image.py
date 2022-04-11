#! /usr/bin/env python
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import sys

bridge = CvBridge()

def image_callback(ros_image):
    #print('Received an Image')
    global bridge

    try:
        cv_image = bridge.imgmsg_to_cv2(ros_image, 'bgr8')
    except CvBridgeError:
        print(CvBridgeError)
    # Canny Edge Detection
    cv_image = cv2.resize(cv_image, (512, 512), interpolation=cv2.INTER_AREA)
    gray_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray_img, 25, 100)
    
    # Green Color Masking
    hsv_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([45, 100, 50])
    upper_green = np.array([75, 255, 255])
    mask = cv2.inRange(hsv_img, lower_green, upper_green)
    img_masked = cv2.bitwise_and(cv_image, cv_image, mask=mask)
    
    # Contour Detection
    _,thr = cv2.threshold(gray_img, 50, 255, cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    img_con = cv2.resize(cv_image, (512, 512), interpolation=cv2.INTER_AREA)
    cv2.drawContours(img_con, contours, -1, (255, 0, 0), 2)

    cv2.imshow('Contours', img_con)
    cv2.imshow('Canny Edge Detection', canny)
    cv2.imshow('Bot Image', cv_image)
    cv2.imshow('Masking', img_masked)
    cv2.waitKey(3)

def main(args):
    rospy.init_node('image_converter', anonymous=True)

    image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, image_callback)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting down')
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)