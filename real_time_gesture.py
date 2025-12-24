import cv2
import mediapipe as mp
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

model = joblib.load('gesture_model.pkl')
print(f"Model expects {model.n_features_in_} features.")

label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('labels.npy', allow_pickle=True)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        landmarks = np.array(landmarks)

        if landmarks.shape[0] == 63:
            X = landmarks.reshape(1, -1)
            pred = model.predict(X)
            gesture = label_encoder.inverse_transform(pred)[0]
            cv2.putText(frame, f'Gesture: {gesture}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        else:
            cv2.putText(frame, 'Landmarks incomplete', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        cv2.putText(frame, 'No hand detected', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow('Real-time Gesture Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

