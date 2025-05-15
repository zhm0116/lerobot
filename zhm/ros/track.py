import argparse
import cv2
import numpy as np
from ultralytics import YOLO

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="Real-time webcam inference with Ultralytics YOLO models. "
                    "For segmentation models, only the masks are displayed (no bbox)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov3-tiny",
        choices=["yolo11n", "yolov3-tiny", "yolo11n-seg","yolov8"],
        help="Select model to use (default: yolov3-tiny)."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.1,
        help="Confidence threshold (default: 0.5)."
    )
    args = parser.parse_args()

    # 根据选择加载相应的权重文件
    if args.model == "yolo11n":
        model_path = "yolo11n.pt"
    elif args.model == "yolov3-tiny":
        model_path = "yolov3-tiny.pt"
    elif args.model == "yolo11n-seg":
        model_path = "yolo11n-seg.pt"
    elif args.model == "yolov8":
        model_path = "yolov8n.pt"

    print(f"Loading model: {args.model} from {model_path}")
    model = YOLO(model_path)
    model.to('cuda')  # Use GPU
    print("Detect use:",model.device)
  
    # 打开默认摄像头
    cap = cv2.VideoCapture(4)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("Error: Unable to open webcam.")
        return

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break

        results = model.track(frame, conf=args.conf,tracker="bytetrack.yaml")
        
        result = results[0] # 第一张图的结果
        # 复制原图
        annotated_frame = result.orig_img.copy()
        
        # 如果使用的是分割模型，则采用手动绘制掩码的方式
        if args.model.endswith("-seg"):
            # 检查是否存在分割掩码
            if result.masks is not None and result.masks.xy is not None:
                # 遍历每个分割的掩码
                for mask in result.masks.xy:
                    pts = np.array(mask, dtype=np.int32)
                    # 填充掩码（颜色为绿色，可根据需要修改）
                    cv2.fillPoly(annotated_frame, [pts], color=(0, 255, 0))
        else:
            # 如果不是分割模型，则直接使用 plot() 方法
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            probs = result.boxes.conf.cpu().numpy() 

            if result.boxes.is_track:
                track_ids = result.boxes.id.cpu().numpy()
                print(f"Tracking IDs: {track_ids}")
            else:
                print("Tracking is not enabled for these boxes.")

            # Loop over detections and plot only if the class is "person" (class index 0)
            for box, cls,prob,track_id in zip(boxes, classes, probs, track_ids):
                if int(cls) == 49:  # person:0 apple:47 orange 49
                    x1, y1, x2, y2 = map(int, box)
                    # Draw a rectangle (green box) for person detection
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    #
                    label = f"ID: {int(track_id)},orange,{prob:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    # Draw confidence score
                    #label = f"{prob:.2f}"
                    #cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0, 255, 0), 2)
        cv2.imshow("Webcam Inference", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
