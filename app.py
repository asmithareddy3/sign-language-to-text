import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load the trained model and labels
model = joblib.load('gesture_model.pkl')
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('labels.npy', allow_pickle=True)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class GestureRecognitionTransformer(VideoTransformerBase):
    def __init__(self) -> None:
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7)
        self.frame_count = 0
        self.last_gesture = "Detecting..."

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # Resize to lower resolution for faster processing
        img = cv2.resize(img, (320, 240))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.frame_count += 1

        # Process only every 3rd frame for speed
        if self.frame_count % 3 == 0:
            results = self.hands.process(img_rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                landmarks = np.array(landmarks).reshape(1, -1)

                if landmarks.shape[1] == 63:
                    pred = model.predict(landmarks)
                    self.last_gesture = label_encoder.inverse_transform(pred)[0]
                else:
                    self.last_gesture = "Landmarks incomplete"

                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            else:
                self.last_gesture = "No hand detected"

        # Show the last recognized gesture
        cv2.putText(img, f'Gesture: {self.last_gesture}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Resize back to original webcam frame size (optional)
        img = cv2.resize(img, (640, 480))

        return img

st.title("Fast Real-time Hand Gesture Recognition")

webrtc_streamer(key="gesture-recognition", video_processor_factory=GestureRecognitionTransformer)
