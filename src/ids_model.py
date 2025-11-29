import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


IMBALANCE_RATIO = 0.05 # 5% intrusion, 95% normal
num_samples = 2000 
num_features = 6

# generate normal data
num_normal = int(num_samples * (1 - IMBALANCE_RATIO))

# using normal distribution (Gaussian) for realism
# features clustered around mean (loc) with low variance (scale)
X_normal = np.random.normal(loc=0.5, scale=0.1, size=(num_normal, num_features))
y_normal = np.zeros(num_normal)

# intrusion data
num_intrusion = num_samples - num_normal

# creating anomalies by increasing variance
# shift in feature-specific readings
X_intrusion = np.random.normal(loc=0.7, scale=0.15, size=(num_intrusion, num_features))
y_intrusion = np.ones(num_intrusion)

# combine and shuffle dataset
X = np.vstack((X_normal, X_intrusion))
y = np.hstack((y_normal, y_intrusion))

# shuffle data to make sure test splits are random
indices = np.arange(num_samples)
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

total_samples = len(y_train)
num_classes = 2

count_normal = np.count_nonzero(y_train == 0)
weight_for_0 = total_samples / (num_classes * count_normal)

count_intrusion = np.count_nonzero(y_train == 1)
weight_for_1 = total_samples / (num_classes * count_intrusion)

class_weight = {0: weight_for_0, 1: weight_for_1}
print(f"Calculated Class Weights: {class_weight}")

model = Sequential([
    Dense(64, activation='relu', input_shape=(num_features,)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Training IDS model...")
model.fit(X_train, y_train, epochs=15, batch_size=16, verbose=1, validation_split=0.2, class_weight=class_weight)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
from sklearn.metrics import precision_score, recall_score, f1_score

# get probability output of the model
y_pred_prob = model.predict(X_test, verbose=0)

# instead of probabilities --> binary predictions
THRESHOLD = 0.30
y_pred_binary = (y_pred_prob > THRESHOLD).astype(int)

# calculate and print new metrics
intrusion_precision = precision_score(y_test, y_pred_binary)
intrusion_recall = recall_score(y_test, y_pred_binary)
intrusion_f1 = f1_score(y_test, y_pred_binary)

print(f"\n*** Intrusion Detection Performance Metrics (Threshold: {THRESHOLD:.2f}) ***")
print(f"Recall (Detection Rate): {intrusion_recall:.4f}")
print(f"Precision (Alert Reliability): {intrusion_precision:.4f}")
print(f"F1-Score: {intrusion_f1:.4f}")


def detect_intrusion(sample):
    """Simulate real-time intrusion detection"""
    sample_scaled = scaler.transform([sample])
    prediction = model.predict(sample_scaled)[0][0]
    if prediction > 0.30:
        print(f"ALERT: Intrusion detected (Confidence: {prediction:.2f})")
    else:
        print(f"Normal activity. (Confidence: {1 - prediction:.2f})")

print("\n*** REAL-TIME SIMULATION ***")
for _ in range(5):
    random_event = np.random.rand(num_features)
    detect_intrusion(random_event)

# for visualizing/generating the confusion matrix
cm = confusion_matrix(y_test, y_pred_binary)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal (0)', 'Intrusion (1)'],
            yticklabels=['Normal (0)', 'Intrusion (1)'])
plt.ylabel("Actual Class")
plt.xlabel('Predicted Class')
plt.title('IDS Confusion Matrix')
plt.savefig('ids_confusion_matrix.png')
plt.show()


