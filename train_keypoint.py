#!/usr/bin/env python
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Configuration
RANDOM_SEED = 42
NUM_CLASSES = 7
dataset = 'model/keypoint_classifier_new/keypoint.csv'
model_save_path = 'model/keypoint_classifier_new/keypoint_classifier.keras'
tflite_save_path = 'model/keypoint_classifier_new/keypoint_classifier.tflite'

# Load dataset
X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, 43)))
y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))

# Check if dataset exists and has data
try:
    if len(X_dataset) == 0:
        print("No training data found. Please collect data using capture_keypoints.py first.")
        exit(1)
    
    # Labels are already 0-9, no remapping needed
    print(f"Loaded {len(X_dataset)} samples with {len(np.unique(y_dataset))} classes")
    print(f"Classes found: {sorted(np.unique(y_dataset))}")
except:
    print("No training data found. Please collect data using capture_keypoints.py first.")
    exit(1)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED)

# Build model for dual hand keypoints
num_classes = 7
model = tf.keras.models.Sequential([
    tf.keras.layers.Input((42,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Callbacks
cp_callback = tf.keras.callbacks.ModelCheckpoint(model_save_path, verbose=1, save_weights_only=False)
es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)

# Compile and train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=1000, batch_size=128, validation_data=(X_test, y_test), callbacks=[cp_callback, es_callback])

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(tflite_save_path, 'wb') as f:
    f.write(tflite_model)

print("Training completed!")