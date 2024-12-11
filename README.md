# Speed Detection Using Drone

This project is a computer vision application designed to detect objects, estimate their speeds, and adjust for camera motion. The system uses a drone-mounted camera for real-time object detection and speed calculation, compensating for the drone's own motion. It leverages the YOLOv8 object detection model and OpenCV for processing video feeds.

---

## **Project Structure**

```
.
├── data
│   ├── drone_tracking.zip # Download file https://drive.google.com/drive/folders/1_qJwjC5C_aCAK0nM0BO5K3V1p1GaSo10 and unzip the file at this path
│   ├── test1.mp4
│   ├── test2.mp4
│   ├── test3.mp4
│   ├── test4.mp4
│   ├── test5.mp4
│   ├── test6.mp4
│   ├── test7.mp4
│   ├── test8.mp4
│   └── test9.mp4
├── requirements.txt
├── src
│   ├── main.py
│   ├── object_detection.py
│   ├── video_processing.py
```

- **`data/`**: Contains sample video files for testing the system.
- **`requirements.txt`**: Lists the dependencies required to run the project.
- **`src/`**: Contains the source code for object detection, video processing, and the main execution script.
- **`yolov10s.pt`**: The YOLOv8 model weights used for object detection.

---

## **Features**

- **Object Detection**: Utilizes the YOLOv8 model to detect objects in a video frame.
- **Speed Estimation**: Calculates the speed of detected objects in real time, compensating for the drone's movement.
- **Camera Motion Compensation**: Uses optical flow to estimate and adjust for the motion of the drone-mounted camera.
- **Video Processing**: Extracts and saves individual frames from a video for debugging purposes.

---

## **Setup and Installation**

### **Prerequisites**

- Python 3.8+
- pip3

### **Installation Steps**

1. Clone this repository:
   ```bash
   git clone speed_detection_using_drone.git
   cd speed_detection_using_drone
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```

---

## **Usage**

### **Run the Main Script**

To process a video and detect object speeds, run the `main.py` script:
```bash
python src/main.py
```

The default video file is located at `data/test6.mp4`. To use a different video, update the `video_path` variable in `main.py`.

### **Process and Save Video Frames**

To save frames from a video for debugging, use the `video_processing.py` script:
```bash
python src/video_processing.py --video_path data/test6.mp4 --output_dir output_frames
```

---

## **Implementation Details**

### **`main.py`**
- Initializes the YOLOv8 model.
- Opens and processes the video file.
- Estimates camera motion using optical flow.
- Detects objects and calculates their speeds relative to the moving camera.
- Displays results with bounding boxes, object labels, and speed.

![Sample Output](speed_detection.gif)

### **`object_detection.py`**
- Contains the `ObjectDetector` class, which uses YOLOv8 for object detection.
- Outputs bounding boxes, confidence scores, and class IDs for detected objects.

### **`video_processing.py`**
- Processes a video file and saves individual frames as images for debugging purposes.

---

## **Sample Output**

- Detected objects are displayed with bounding boxes, their labels, and speeds (in km/h) overlaid on the video feed.
- Adjustments are made for the drone's motion to ensure accurate speed calculation.

---

## **Future Work**

- Enhance the accuracy of motion compensation.
- Add support for real-time drone video feeds.
- Improve speed calculation for objects moving in complex trajectories.


