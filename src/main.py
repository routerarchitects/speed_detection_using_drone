import cv2
from object_detection import ObjectDetector
import math
import numpy as np


def calculate_speed(prev_position, curr_position, time_delta, pixels_to_meters):
    """
    Calculate the speed of an object in km/h.

    :param prev_position: Tuple (x, y) representing the object's position in the previous frame (in pixels).
    :param curr_position: Tuple (x, y) representing the object's position in the current frame (in pixels).
    :param time_delta: Time difference between the previous and current frames (in seconds).
    :param pixels_to_meters: Conversion factor to translate pixel distances to meters.
    :return: Speed of the object in kilometers per hour (km/h).
    """
    # If either position is None (e.g., no previous position available), return a speed of 0
    if prev_position is None or curr_position is None:
        return 0

    # Calculate the Euclidean distance between the previous and current positions in pixels
    distance_in_pixels = math.sqrt((curr_position[0] - prev_position[0]) ** 2 +
                                   (curr_position[1] - prev_position[1]) ** 2)

    # Convert the distance from pixels to meters using the conversion factor
    distance_in_meters = distance_in_pixels * pixels_to_meters

    # Calculate the speed in meters per second (m/s)
    speed_m_s = distance_in_meters / time_delta

    # Convert the speed from meters per second to kilometers per hour (1 m/s = 3.6 km/h)
    return speed_m_s * 3.6


def estimate_camera_motion(prev_frame, curr_frame):
    """
    Estimate the camera's motion between two frames using optical flow.

    :param prev_frame: The previous frame (color image in BGR format).
    :param curr_frame: The current frame (color image in BGR format).
    :return: Average motion vector (dx, dy) representing the camera's movement in pixels.
    """
    # Convert frames to grayscale for optical flow computation
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Detect good features to track in the previous frame
    # maxCorners: Maximum number of corners to detect
    # qualityLevel: Minimum accepted quality of corners (0.01 = 1%)
    # minDistance: Minimum Euclidean distance between corners
    features_prev = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)

    # Compute the optical flow (track the motion of features) between the two frames
    features_curr, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, features_prev, None)

    # Filter features where the status is 1 (successfully tracked points)
    valid_prev = features_prev[status == 1]
    valid_curr = features_curr[status == 1]

    # Calculate motion vectors as the difference between corresponding feature points
    motion_vectors = valid_curr - valid_prev
    # print("Estimated Camera Motion:", motion_vectors)

    # Compute the average motion vector (dx, dy)
    # If no motion vectors are found, return [0, 0] (no motion)
    avg_motion = np.mean(motion_vectors, axis=0) if len(motion_vectors) > 0 else np.array([0, 0])

    return avg_motion


def main(video_path):
    """
    Main function to process video, detect objects, calculate their speeds,
    and display the results in real-time.

    :param video_path: Path to the input video file.
    """
    print("Initializing YOLOv8 model...")
    detector = ObjectDetector(model_path="yolov10s.pt")
    print("Model initialized successfully.")

    print("Opening video file...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}")
        return
    print("Video file opened successfully.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    time_delta = 1 / fps
    pixels_to_meters = 0.2

    tracked_objects = {}
    prev_frame = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video file.")
            break

        if prev_frame is not None:
            camera_motion = estimate_camera_motion(prev_frame, frame)
        else:
            camera_motion = np.array([0, 0])

        boxes, confidences, class_ids = detector.detect_objects(frame)

        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            confidence = confidences[i]
            current_label = str(detector.classes[class_ids[i]])
            center = (x + w // 2, y + h // 2)

            if center is not None:
                center = (center[0] - camera_motion[0], center[1] - camera_motion[1])

            object_id = i
            prev_data = tracked_objects.get(object_id, {"label": current_label, "confidence": confidence, "center": None})

            if confidence < prev_data["confidence"]:
                current_label = prev_data["label"]
            else:
                prev_data["label"] = current_label
                prev_data["confidence"] = confidence

            prev_position = prev_data["center"]
            tracked_objects[object_id] = {"label": current_label, "confidence": confidence, "center": center}

            # Debug print for positions
            # print(f"Object ID: {object_id}, Previous Position: {prev_position}, Current Center: {center}")

            speed_kmh = calculate_speed(prev_position, center, time_delta, pixels_to_meters)

            # Debug print for speed
            #print(f"Object ID: {object_id}, Speed: {speed_kmh:.2f} km/h")

            cv2.putText(frame, f"{current_label} {speed_kmh:.2f} km/h", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        prev_frame = frame
        frame_count += 1

        cv2.imshow("Object Detection with Speed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "data/test6.mp4"
    main(video_path)
