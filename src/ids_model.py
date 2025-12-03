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
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# defining here instead of hardcoding
CONFIG = {
    'IMBALANCE_RATIO': 0.05, # 5% intrusion, 95% normal
    'RANDOM_SEED': 42,
    'TEST_SIZE': 0.2,
    'LOOKBACK_STEPS': 40,
    'EPOCHS': 50,
    'BATCH_SIZE': 64,
    'L2_REG_STRENGTH': 0.0005,
    'DROPOUT_RATE': 0.2,
    'THRESHOLD': 0.25,
}

# data loading and generation
data = pd.read_csv('cybersecurity_intrusion_data.csv')
print(f"Loaded {len(data)} samples from dataset.")

data = data.drop('session_id', axis=1)

# separating attack feature and target variable
X = data.drop('attack_detected', axis=1)
y = data['attack_detected'].values

categorical_cols = ['protocol_type', 'encryption_used', 'browser_type']
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

CONFIG['NUM_RAW_FEATURES'] = X.shape[1]
num_features_raw = CONFIG['NUM_RAW_FEATURES']

X = X.values

P_avg_list = []
history_buffer = np.zeros(CONFIG['LOOKBACK_STEPS'])

for i in range(len(y)):
    P_avg_val = history_buffer.mean()
    P_avg_list.append(P_avg_val)

    history_buffer = np.roll(history_buffer, -1)
    history_buffer[-1] = y[i]

P_avg_feature = np.array(P_avg_list).reshape(-1, 1)

# combine P_avg with X
X = np.hstack((X, P_avg_feature))
num_samples = len(y)
num_features = X.shape[1]
print(f"Total features after processing: {num_features}")

indices = np.arange(num_samples)
np.random.seed(CONFIG['RANDOM_SEED'])
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

num_features_for_sim = CONFIG['NUM_RAW_FEATURES'] 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=CONFIG['TEST_SIZE'], 
    random_state=CONFIG['RANDOM_SEED']
)

# scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# calculate class weight
total_samples = len(y_train)
num_classes = 2

count_normal = np.count_nonzero(y_train == 0)
weight_for_0 = total_samples / (num_classes * count_normal)

count_intrusion = np.count_nonzero(y_train == 1)
weight_for_1 = total_samples / (num_classes * count_intrusion)

class_weight = {0: weight_for_0, 1: weight_for_1}
print(f"Calculated Class Weights: {class_weight}")

early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=5,
    restore_best_weights=True
)

model = Sequential([
    Dense(128, activation='relu', input_shape=(num_features,),
          kernel_regularizer=regularizers.l2(CONFIG['L2_REG_STRENGTH'])),
    Dropout(CONFIG['DROPOUT_RATE']),
    Dense(64, activation='relu',
          kernel_regularizer=regularizers.l2(CONFIG['L2_REG_STRENGTH'])),
    Dropout(CONFIG['DROPOUT_RATE']),
    Dense(32, activation='relu'),
    Dropout(CONFIG['DROPOUT_RATE']),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])

print("Training IDS model...")
model.fit(
    X_train, y_train, 
    epochs=CONFIG['EPOCHS'], 
    batch_size=CONFIG['BATCH_SIZE'], 
    verbose=1, 
    validation_split=0.2, 
    callbacks=[early_stopping],
    class_weight=class_weight
)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
from sklearn.metrics import precision_score, recall_score, f1_score

# get probability output of the model
y_pred_prob = model.predict(X_test, verbose=0)

# instead of probabilities --> binary predictions
THRESHOLD = CONFIG['THRESHOLD']
y_pred_binary = (y_pred_prob > THRESHOLD).astype(int)

# calculate and print new metrics
intrusion_precision = precision_score(y_test, y_pred_binary)
intrusion_recall = recall_score(y_test, y_pred_binary)
intrusion_f1 = f1_score(y_test, y_pred_binary)

print(f"\n*** Intrusion Detection Performance Metrics (Threshold: {THRESHOLD:.2f}) ***")
print(f"Recall (Detection Rate): {intrusion_recall:.4f}")
print(f"Precision (Alert Reliability): {intrusion_precision:.4f}")
print(f"F1-Score: {intrusion_f1:.4f}")

simulation_history_buffer = np.zeros(CONFIG['LOOKBACK_STEPS'])

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
    simulation_history_buffer[-1] = (prediction > CONFIG['THRESHOLD']).astype(int)

    if prediction > CONFIG['THRESHOLD']:
        alert_message = f"[{timestamp}] ALERT: Intrusion detected (Confidence: {prediction:.2f})."
        with open("intrusion_log.txt", "a") as f:
            f.write(alert_message + "\n")
        print(alert_message)
    else:
        print(f"Normal activity. (Confidence: {1 - prediction:.2f})")

num_features_raw_for_sim = num_features - 1

print("\n*** REAL-TIME SIMULATION ***")
# specific simulation for in-class demo, can revert later
print("\nTesting normal activity...")
for _ in range(2):
    normal_event = np.random.normal(loc=0.5, scale=0.1, size=(num_features_raw_for_sim))
    detect_intrusion(normal_event)

print("\nTesting intrusion detection...")
intrusion_event = np.random.normal(loc=0.8, scale=0.15, size=(num_features_raw_for_sim))
detect_intrusion(intrusion_event)

print("\nTesting random/edge case...")
random_event = np.random.rand(num_features_raw_for_sim)
detect_intrusion(random_event)

print("\nTesting extended dynamic activity...")
for _ in range(5):
    random_event = np.random.rand(num_features_raw_for_sim)
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


