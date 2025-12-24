import numpy as np
from collections import Counter
from sklearn.utils import resample

X = np.load('X.npy', allow_pickle=True)
y = np.load('y.npy', allow_pickle=True)

counts = Counter(y)
print("Sample counts before balancing:")
for gesture, count in counts.items():
    print(f"{gesture}: {count}")

min_count = min(counts.values())

X_balanced = []
y_balanced = []

for gesture in counts.keys():
    gesture_indices = np.where(y == gesture)[0]
    gesture_X = X[gesture_indices]
    gesture_y = y[gesture_indices]

    if len(gesture_X) > min_count:
        gesture_X_resampled, gesture_y_resampled = resample(
            gesture_X, gesture_y,
            replace=False,
            n_samples=min_count,
            random_state=42
        )
    else:
        gesture_X_resampled, gesture_y_resampled = gesture_X, gesture_y

    X_balanced.append(gesture_X_resampled)
    y_balanced.append(gesture_y_resampled)

X_balanced = np.vstack(X_balanced)
y_balanced = np.concatenate(y_balanced)

print("\nSample counts after balancing:")
counts_bal = Counter(y_balanced)
for gesture, count in counts_bal.items():
    print(f"{gesture}: {count}")

np.save('X_balanced.npy', X_balanced)
np.save('y_balanced.npy', y_balanced)

print("\nBalanced dataset saved to 'X_balanced.npy' and 'y_balanced.npy'")
