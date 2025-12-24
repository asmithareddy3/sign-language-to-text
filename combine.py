import numpy as np

# List your gesture names here exactly as your folders or files are named
gesture_names = ['food', 'hello', 'help']

all_features = []
all_labels = []

for gesture in gesture_names:
    # Change file names if you saved differently
    X = np.load(f'{gesture}_X.npy', allow_pickle=True)
    y = np.array([gesture] * len(X))  # Create labels for these samples
    all_features.append(X)
    all_labels.append(y)

# Combine all features and labels into single arrays
X_combined = np.concatenate(all_features)
y_combined = np.concatenate(all_labels)

# Save combined dataset files for training
np.save('X.npy', X_combined)
np.save('y.npy', y_combined)

print(f"Combined dataset shape: X={X_combined.shape}, y={y_combined.shape}")
