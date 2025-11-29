import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers


IMBALANCE_RATIO = 0.05 # 5% intrusion, 95% normal
num_samples = 3000 
num_features_raw = 6
LOOKBACK_STEPS = 20
num_features = num_features_raw + 1

simulation_history_buffer = np.zeros(LOOKBACK_STEPS)

# generate time-ordered data
num_normal = int(num_samples *  (1 - IMBALANCE_RATIO))
num_intrusion = num_samples - num_normal

X_base_normal = np.random.normal(loc=0.5, scale=0.1, size=(num_normal, num_features_raw))
X_base_intrusion =  np.random.normal(loc=0.8, scale=0.15, size=(num_intrusion, num_features_raw))

X_time_ordered = np.vstack((X_base_normal, X_base_intrusion))
y_time_ordered = np.hstack((np.zeros(num_normal), np.ones(num_intrusion)))

# calculate P_avg (historical augmenting)
P_avg_list = []
history_buffer = np.zeros(LOOKBACK_STEPS)

for i in range(num_samples):
    P_avg_val = history_buffer.mean()
    P_avg_list.append(P_avg_val)

    history_buffer = np.roll(history_buffer, -1)
    history_buffer[-1] = y_time_ordered[i]

P_avg_feature = np.array(P_avg_list).reshape(-1, 1)

# combine + last shuffle
X = np.hstack((X_time_ordered, P_avg_feature))
y = y_time_ordered

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
    Dense(64, activation='relu', input_shape=(num_features,),
          kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.2),
    Dense(32, activation='relu', 
          kernel_regularizer=regularizers.l2(0.001)),
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

# intrusion detection simulation with logging
def detect_intrusion(sample):
    global simulation_history_buffer
    P_avg_current = simulation_history_buffer.mean()

    # augment the sample
    sample_augmented = np.append(sample, P_avg_current)
    sample_scaled = scaler.transform([sample_augmented])

    # prediction
    prediction = model.predict(sample_scaled, verbose=0)[0][0]
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # updating buffer with the new prediction
    simulation_history_buffer = np.roll(simulation_history_buffer, -1)
    simulation_history_buffer[-1] = prediction

    if prediction > 0.30:
        alert_message = f"[{timestamp}] ALERT: Intrusion detected (Confidence: {prediction:.2f})."
        with open("intrusion_log.txt", "a") as f:
            f.write(alert_message + "\n")
        print(alert_message)
    else:
        print(f"Normal activity. (Confidence: {1 - prediction:.2f})")

print("\n*** REAL-TIME SIMULATION ***")
for _ in range(5):
    random_event = np.random.rand(num_features_raw)
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


