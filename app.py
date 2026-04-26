import cv2 as cv
import time
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

# Connectors

POSE_CONNECTIONS = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # Right arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # Left leg
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # Right leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

# Drawing connectors function
def draw_skeleton(frame, landmarks, visibility_threshold=0.3):
    h, w = frame.shape[:2]

    points = []

    for lm in landmarks:
        if lm.visibility > visibility_threshold:
            points.append((int(lm.x * w), int(lm.y * h)))
        else:
            points.append(None)  # placeholder to keep indices aligned

    for start_idx, end_idx in POSE_CONNECTIONS:
        p1, p2 = points[start_idx], points[end_idx]
        if p1 and p2:  # skip if either endpoint is low-confidence
            cv.line(frame, p1, p2, (0, 255, 0), 2)

    for pt in points:
        if pt:
            cv.circle(frame, pt, 4, (0, 0, 255), -1)

# MediaPipe setup
base_options = mp_tasks.BaseOptions(model_asset_path="./models/pose_landmarker_full.task")

options = PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=RunningMode.VIDEO
)

# Camera Setup
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 640)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 480)

# Draw joints
with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        # Error handler
        if not success:
            print("Something went wrong while capturing frame")
            break

        # Conver CV2 frame into mediaPipe object
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Drawing part
        if result.pose_landmarks:
            draw_skeleton(frame, result.pose_landmarks[0])

        cv.imshow("Pose", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            print('Window closed')
            break

cap.release()
cv.destroyAllWindows()