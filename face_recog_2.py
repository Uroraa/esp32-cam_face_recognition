import cv2
import numpy as np
import threading, time
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import os
import mediapipe as mp
from scipy.spatial import distance as dist
import sys
sys.stdout.reconfigure(encoding='utf-8')

ESP32_URL = "http://172.17.12.184:81/stream"

app = FaceAnalysis(name="buffalo_sc", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

data = np.load("known_faces.npz", allow_pickle=True)
known_names = data["names"].tolist()
known_embeddings = data["embeddings"]
print(f"Đã nạp {len(known_names)} khuôn mặt mẫu")

mp_face = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

ACTIONS = ["BLINK", "TURN_LEFT", "TURN_RIGHT", "TILT", "MOUTH_OPEN", "LOOK_UP", "LOOK_DOWN"]
             
#ACTIONS = ["TURN_LEFT", "TURN_RIGHT"]

current_action = np.random.choice(ACTIONS)
action_verified = False   
next_action_delay = False       
next_action_time = 0            
DELAY_AFTER_SUCCESS = 1.0

EAR_HISTORY = []
EAR_MAX_LEN = 60

def EAR(eye_points):
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    C = dist.euclidean(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C)

def draw_eye(frame, pts, color=(0, 255, 0)):
    for i in range(len(pts)):
        cv2.circle(frame, pts[i], 2, color, -1)
        cv2.line(frame, pts[i], pts[(i+1) % len(pts)], color, 1)

def get_angle(p1, p2):
    return np.degrees(np.arctan2(p2[1]-p1[1], p2[0]-p1[0]))

def detect_head_pose(landmarks, w, h):
    nose = np.array([landmarks[1].x*w, landmarks[1].y*h])
    forehead = landmarks[10]
    nose     = landmarks[1]
    chin     = landmarks[152]

    face_vec = np.array([
        chin.x - forehead.x,
        chin.y - forehead.y,
        chin.z - forehead.z
    ])

    center = np.array([
        (forehead.x + chin.x) / 2,
        (forehead.y + chin.y) / 2,
        (forehead.z + chin.z) / 2
    ])

    forward = np.array([
        nose.x - center[0],
        nose.y - center[1],
        nose.z - center[2]
    ])

    down_vec = np.array([0, 1, 0])  

    cos_angle = np.dot(face_vec, down_vec) / (
        np.linalg.norm(face_vec) * np.linalg.norm(down_vec)
    )

    pitch = np.degrees(np.arccos(cos_angle))
    if nose.z < forehead.z:
        pitch = +abs(pitch)  
    else:
        pitch = -abs(pitch)

    yaw = np.degrees(np.arctan2(forward[0], -forward[2]))
    roll = get_angle(
        np.array([landmarks[33].x*w, landmarks[33].y*h]),   #point_left_eye_corner
        np.array([landmarks[263].x*w, landmarks[263].y*h])  #point_right_eye_corner
    )
    return yaw, roll, pitch

cap = cv2.VideoCapture(ESP32_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

frame_lock = threading.Lock()
latest_frame = None
running = True

if not cap.isOpened():
    print("Không mở được luồng video từ ESP32")
    exit()

def capture_thread():
    global latest_frame, running
    while running:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                latest_frame = frame

threading.Thread(target=capture_thread, daemon=True).start()

while True:
    with frame_lock:
        frame = latest_frame.copy() if latest_frame is not None else None
    if frame is None:
        continue

    faces = app.get(frame)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face.process(rgb)

    EAR_value = 1.0
    pitch = None
    yaw = None
    roll = None
    mouth_dist = None
    EAR_value = None
    action_verified_now = False

    if results.multi_face_landmarks:
        for lm in results.multi_face_landmarks:
            h, w = frame.shape[:2]

            left_eye_pts = [(int(lm.landmark[i].x * w), int(lm.landmark[i].y * h)) for i in LEFT_EYE]
            right_eye_pts = [(int(lm.landmark[i].x * w), int(lm.landmark[i].y * h)) for i in RIGHT_EYE]

            draw_eye(frame, left_eye_pts)
            draw_eye(frame, right_eye_pts)

            left_EAR = EAR(left_eye_pts)
            right_EAR = EAR(right_eye_pts)
            EAR_value = (left_EAR + right_EAR) / 2.0

            EAR_HISTORY.append(EAR_value)
            if len(EAR_HISTORY) > EAR_MAX_LEN:
                EAR_HISTORY.pop(0)

            yaw, roll, pitch = detect_head_pose(lm.landmark, w, h)

            # Blink
            if current_action == "BLINK" and EAR_value < 0.18:
                action_verified_now = True

            # Head turn
            if current_action == "TURN_LEFT" and yaw < -25:
                action_verified_now = True
            if current_action == "TURN_RIGHT" and yaw > 25:
                action_verified_now = True

            # Tilt head
            if current_action == "TILT" and abs(roll) > 15:
                action_verified_now = True

            # Look up/down
            if current_action == "LOOK_DOWN" and pitch < -15:
                action_verified_now = True
            if current_action == "LOOK_UP" and pitch > 18:
                action_verified_now = True

            # Mouth open
            upper_lip = lm.landmark[13]
            lower_lip = lm.landmark[14]
            mouth_dist = abs((lower_lip.y - upper_lip.y) * h)

            if current_action == "MOUTH_OPEN" and mouth_dist > 30:
                action_verified_now = True

            # Smile
            '''left_m = lm.landmark[61]
            right_m = lm.landmark[291]
            mouth_w = abs((right_m.x - left_m.x) * w)

            if current_action == "SMILE" and mouth_w > 65:
                action_verified_now = True'''

    if action_verified_now and not next_action_delay:
        next_action_delay = True
        next_action_time = time.time() + DELAY_AFTER_SUCCESS

    if next_action_delay and time.time() >= next_action_time:
        current_action = np.random.choice(ACTIONS)
        last_action_time = time.time()
        next_action_delay = False
        print("[NEW ACTION] →", current_action)
    
    for f in faces:
        x1, y1, x2, y2 = [int(v) for v in f.bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        best_match = "Unknown"
        best_score = 0.0

        for name, emb in zip(known_names, known_embeddings):
            score = cosine_similarity([f.embedding], [emb])[0][0]
            if score > best_score:
                best_score = score
                best_match = name

        if not action_verified_now:
            best_match = "Unknown"

        cv2.putText(frame, f"{best_match} ({best_score:.2f})",
                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255,255,255), 2)

    color = (0,255,0) if action_verified_now else (0,0,255)

    cv2.putText(frame, f"Action: {current_action}",
                (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)

    point_value = None

    if current_action == "BLINK":
        point_value = EAR_value
    elif current_action in ["TURN_LEFT", "TURN_RIGHT"]:
        point_value = yaw
    elif current_action == "LOOK_UP" or current_action == "LOOK_DOWN":
        point_value = pitch
    elif current_action == "TILT":
        point_value = roll
    elif current_action == "MOUTH_OPEN":
        point_value = mouth_dist
    '''elif current_action == "SMILE":
        point_value = mouth_w'''

    if point_value is not None:
        cv2.putText(frame, f"Point: {point_value:.2f}",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)

    cv2.imshow("ESP32-CAM Recognition", frame)
    if cv2.waitKey(1) == 27:
        running = False
        break

cap.release()
cv2.destroyAllWindows()