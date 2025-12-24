import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

X = np.load('X.npy', allow_pickle=True)
y = np.load('y.npy', allow_pickle=True)

X_flat = []
for landmarks_seq in X:
    if len(landmarks_seq) > 0:
        frame = landmarks_seq[0]
        frame = np.array(frame).flatten()
        X_flat.append(frame)
    else:
        X_flat.append(np.zeros(63))

X_flat = np.array(X_flat)
print(f"Training data shape: {X_flat.shape}")

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_flat, y)

print("Model trained successfully.")

joblib.dump(clf, 'gesture_model.pkl')
print("Model saved as 'gesture_model.pkl'.")
