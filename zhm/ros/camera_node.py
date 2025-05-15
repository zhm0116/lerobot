#!/usr/bin/env python3
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def main():
    rospy.init_node('camera_publisher', anonymous=False)
    pub = rospy.Publisher('/camera/image_raw', Image, queue_size=1)
    bridge = CvBridge()

    cap = cv2.VideoCapture(4)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        rospy.logerr("Unable to open camera")
        return

    rate = rospy.Rate(30)
    while True:#not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            continue
        msg = bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        pub.publish(msg)
        rate.sleep()
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()
