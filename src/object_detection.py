from ultralytics import YOLO


class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        """
        Initialize YOLOv8 model.
        :param model_path: Path to the YOLOv8 model file.
        """
        self.model = YOLO(model_path)
        self.classes = self.model.names  # Get class names directly from the YOLOv8 model

    def detect_objects(self, frame):
        """
        Detect objects in a frame using YOLOv8.
        :param frame: Input image frame (numpy array).
        :return: List of bounding boxes, confidences, and class IDs.
        """
        results = self.model(frame)
        detections = results[0].boxes.data.cpu().numpy()

        boxes = []
        confidences = []
        class_ids = []

        for detection in detections:
            x1, y1, x2, y2, confidence, class_id = detection
            boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
            confidences.append(float(confidence))
            class_ids.append(int(class_id))

        return boxes, confidences, class_ids
