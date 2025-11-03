import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

np.random.seed(42)
num_samples = 1000
num_features = 6

X = np.random.rand(num_samples, num_features)

y = np.random.choice([0, 1], size=(num_samples,), p=[0.8, 0.2])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential([
    Dense(16, activation='relu', input_shape=(num_features,)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Training IDS model...")
model.fit(X_train, y_train, epochs=15, batch_size=16, verbose=1, validation_split=0.2)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n Model accuracy: {accuracy*100:.2f}%")

def detect_intrusion(sample):
    """Simulate real-time intrusion detection"""
    sample_scaled = scaler.transform([sample])
    prediction = model.predict(sample_scaled)[0][0]
    if prediction > 0.3:
        print(f"ALERT: Intrusion detected (Confidence: {prediction:.2f})")
    else:
        print(f"Normal activity. (Confidence: {1 - prediction:.2f})")

print("\n*** REAL-TIME SIMULATION ***")
for _ in range(5):
    random_event = np.random.rand(num_features)
    detect_intrusion(random_event)
