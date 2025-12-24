import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load scaler and model
scaler = joblib.load('scaler_single.pkl')
model = joblib.load('model_single_gesture.pkl')

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    gesture_text = "Gesture not detected"

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        if len(landmarks) == 63:
            X = np.array(landmarks).reshape(1, -1)
            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled)[0]
            score = model.decision_function(X_scaled)[0]

            print(f"Prediction: {pred}, Score: {score}")

            gesture_text = "Gesture detected" if pred == 1 else "Gesture not detected"

        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, gesture_text, (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("Single Gesture Detection Debug", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
