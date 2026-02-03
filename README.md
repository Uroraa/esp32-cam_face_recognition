# ESP32-CAM Face Recognition with Liveness Detection

A real-time face recognition system using an ESP32-CAM camera with advanced liveness detection through facial action verification. The system combines face embedding analysis with multi-modal liveness checks including blink detection, head pose estimation, and facial expressions.

## Features

- **Real-Time Face Recognition**: Identifies known faces from a live video stream using InsightFace embeddings
- **Liveness Detection**: Prevents spoofing attacks through multi-step action verification:
  - Blink detection (Eye Aspect Ratio)
  - Head pose estimation (Yaw, Roll, Pitch)
  - Facial expressions (Mouth opening)
  - Head movements (Turn left/right, tilt, look up/down)
- **ESP32-CAM Integration**: Streams video directly from ESP32-CAM module
- **Face Embedding Storage**: Pre-computed face embeddings for efficient recognition
- **MediaPipe Integration**: Advanced facial landmark detection for precise action recognition
- **Real-Time Visualization**: Live display with action indicators and confidence scores

## Project Structure

```
.
├── face_recog.py          # Main face recognition and liveness detection script
├── face_save.py           # Generate face embeddings from reference images
├── known_faces.npz        # Serialized face embeddings and names database
├── ex_faces/              # Directory for captured/processed faces
└── README.md              # This file
```

## System Requirements

### Hardware
- **ESP32-CAM Module**: With WiFi connectivity
- **Computer**: CPU/GPU capable of running InsightFace and MediaPipe
- **Network**: Local network connection between computer and ESP32-CAM

### Software
- Python 3.7+
- OpenCV
- NumPy
- InsightFace
- MediaPipe
- scikit-learn
- SciPy

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Uroraa/esp32-cam_face_recognition
cd esp32-cam_face_recognition
```

### 2. Install Python Dependencies
```bash
pip install opencv-python numpy insightface mediapipe scikit-learn scipy
```

### 3. Set Up ESP32-CAM
- Flash ESP32-CAM with streaming firmware (e.g., using Arduino IDE)
- Connect ESP32-CAM to your WiFi network
- Note the IP address assigned to the ESP32-CAM

## Usage

### Step 1: Prepare Reference Face Images
1. Place clear face images in the `ex_faces/` directory
2. Name files as `{person_name}.jpg` 
3. Ensure each image contains a single, clear face

### Step 2: Generate Face Embeddings
```bash
python face_save.py
```
This script will:
- Read all images from `ex_faces/`
- Extract face embeddings using InsightFace
- Save embeddings and names to `known_faces.npz`

### Step 3: Configure ESP32-CAM Connection
Edit `face_recog.py` and update the ESP32-CAM IP address:
```python
ESP32_URL = "http://YOUR_ESP32_IP:81/stream"
```

### Step 4: Run Face Recognition with Liveness Detection
```bash
python face_recog.py
```

## How It Works

### Face Encoding & Recognition
1. **Reference Processing** (`face_save.py`):
   - Loads all reference images from `ex_faces/`
   - Uses InsightFace's buffalo_sc model to extract 512-dimensional embeddings
   - Stores embeddings and names in `known_faces.npz`

2. **Live Recognition** (`face_recog.py`):
   - Extracts embeddings from detected faces in the video stream
   - Computes cosine similarity with known face embeddings
   - Identifies faces with high similarity scores

### Liveness Detection Protocol
The system requires users to perform a sequence of actions in order:
1. **BLINK**: Close and open eyes (Eye Aspect Ratio < 0.18)
2. **TURN_LEFT**: Rotate head left (Yaw < -25°)
3. **TURN_RIGHT**: Rotate head right (Yaw > 25°)
4. **TILT**: Tilt head side-to-side (|Roll| > 15°)
5. **LOOK_DOWN**: Look downward (Pitch < -15°)
6. **LOOK_UP**: Look upward (Pitch > 18°)
7. **MOUTH_OPEN**: Open mouth (Mouth distance > 30 pixels)

Actions are verified sequentially, and once all actions are completed, the cycle repeats.

### Facial Metrics
- **Eye Aspect Ratio (EAR)**: Measures eye openness for blink detection
- **Head Pose (Yaw, Roll, Pitch)**: 3D orientation angles calculated from facial landmarks
- **Mouth Distance**: Vertical distance between upper and lower lips

## Configuration

### Adjustable Thresholds in `face_recog.py`

## Output Indicators

The live display shows:
- **Green rectangles**: Detected faces
- **Green color text**: Action successfully verified
- **Red color text**: Waiting for action verification
- **Face name and score**: Recognized person with confidence (0.0-1.0)
- **Current action**: Next action to perform for liveness check
- **Point value**: Current metric value for the active action
![Demo](.assets/image.png)

## Advanced Usage

### Adding New Known Faces
1. Add new face image to `ex_faces/` with appropriate naming
2. Re-run `face_save.py` to regenerate embeddings
3. Restart `face_recog.py`

### Adjusting Detection Sensitivity
Modify thresholds in `face_recog.py`:
- Lower thresholds = more sensitive detection
- Higher thresholds = stricter verification

### Custom Action Sequences
Edit the `ACTIONS` list in `face_recog.py` to change the liveness verification sequence.

## Performance Notes

- **Resolution**: Default 640x640 for face detection
- **Frame Rate**: Depends on ESP32-CAM stream quality and network speed
- **Processing**: Runs on CPU by default (set `providers=['CPUExecutionProvider']`)
- **Threading**: Uses separate thread for video capture to prevent frame drops

## References

- [InsightFace](https://github.com/deepinsight/insightface)
- [MediaPipe Face Mesh](https://mediapipe.dev/)
- [OpenCV](https://opencv.org/)
- [ESP32-CAM Documentation](https://github.com/espressif/esp32-camera)

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Support

For issues or questions, please open an issue in the repository or contact the project maintainers.
