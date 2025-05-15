#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
from ultralytics import YOLO

class OrangeDetector:
    def __init__(self):
        rospy.init_node('orange_detector', anonymous=True)
        self.bridge = CvBridge()

        self.bbox_pub = rospy.Publisher('/orange_bboxes', Float32MultiArray, queue_size=10)
        self.annotated_pub = rospy.Publisher('/orange_annotated', Image, queue_size=1)
        rospy.Subscriber('/camera/image_raw', Image, self.image_callback)

        model_path = "yolov8n.pt"
        self.model = YOLO(model_path).to('cuda')
        rospy.loginfo(f"Loaded model on {self.model.device}")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(f"CV Bridge error: {e}")
            return

        results = self.model(frame, conf=0.3)
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        probs = result.boxes.conf.cpu().numpy()

        orange_bboxes = Float32MultiArray()
        annotated = frame.copy()

        for box, cls, prob in zip(boxes, classes, probs):
            if int(cls) == 49:  # orange class index
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
                label = f"{prob:.2f}"
                cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                orange_bboxes.data.extend([x1, y1, x2, y2, float(prob)])

        self.bbox_pub.publish(orange_bboxes)
        self.annotated_pub.publish(self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8'))

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    OrangeDetector().run()
