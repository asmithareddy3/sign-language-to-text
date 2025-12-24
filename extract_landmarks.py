import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

gesture_dir = "gesture_dataset"
output_dir = "landmark_data"
os.makedirs(output_dir, exist_ok=True)

for gesture in os.listdir(gesture_dir):
    gesture_path = os.path.join(gesture_dir, gesture)
    if not os.path.isdir(gesture_path):
        continue

    save_gesture_dir = os.path.join(output_dir, gesture)
    os.makedirs(save_gesture_dir, exist_ok=True)

    for video_name in os.listdir(gesture_path):
        video_path = os.path.join(gesture_path, video_name)
        cap = cv2.VideoCapture(video_path)

        landmarks_all_frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                frame_landmarks = []
                for lm in hand_landmarks.landmark:
                    frame_landmarks.extend([lm.x, lm.y, lm.z])
                landmarks_all_frames.append(frame_landmarks)
            else:
                landmarks_all_frames.append([0]*63)

        cap.release()

        landmarks_all_frames = np.array(landmarks_all_frames)
        base_name = os.path.splitext(video_name)[0]
        npy_path = os.path.join(save_gesture_dir, f"{base_name}.npy")
        np.save(npy_path, landmarks_all_frames)
        print(f"Saved landmarks: {npy_path}")

print("Extraction completed.")
