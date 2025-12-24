import numpy as np
import os

landmark_dir = "landmark_data"
gestures = sorted(os.listdir(landmark_dir))

X = []
y = []

for idx, gesture in enumerate(gestures):
    gesture_path = os.path.join(landmark_dir, gesture)
    if not os.path.isdir(gesture_path):
        continue

    for file_name in os.listdir(gesture_path):
        file_path = os.path.join(gesture_path, file_name)
        landmarks = np.load(file_path)
        X.append(landmarks)
        y.append(idx)

X = np.array(X, dtype=object)
y = np.array(y)

np.save('labels.npy', np.array(gestures))
np.save('X.npy', X)
np.save('y.npy', y)

print(f"Prepared dataset with {len(X)} samples and {len(gestures)} gesture classes.")
