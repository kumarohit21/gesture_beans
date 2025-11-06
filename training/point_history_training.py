#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def load_dataset(dataset_path):
    """Load point history dataset from CSV file"""
    X_dataset = np.loadtxt(dataset_path, delimiter=',', dtype='float32', usecols=list(range(1, 33)))
    y_dataset = np.loadtxt(dataset_path, delimiter=',', dtype='int32', usecols=(0))
    return X_dataset, y_dataset

def create_model(num_classes=4):
    """Create point history classification model"""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input((32,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def train_model():
    """Train the point history classification model"""
    # Load dataset
    X_dataset, y_dataset = load_dataset('model/point_history_classifier/point_history.csv')
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.75, random_state=42)
    
    # Create and compile model
    model = create_model(num_classes=len(np.unique(y_dataset)))
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    model.fit(
        X_train, y_train,
        epochs=1000,
        batch_size=128,
        validation_data=(X_test, y_test)
    )
    
    # Save model
    model.save('model/point_history_classifier/point_history_classifier.hdf5')
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quantized_model = converter.convert()
    
    with open('model/point_history_classifier/point_history_classifier.tflite', 'wb') as f:
        f.write(tflite_quantized_model)
    
    print("Point history model training completed and saved!")

if __name__ == '__main__':
    train_model()