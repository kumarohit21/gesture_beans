#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training script for both keypoint and point history models
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def train_keypoint():
    import subprocess
    subprocess.run([sys.executable, 'train_keypoint.py'])

def train_point_history():
    from training.point_history_training import train_model
    train_model()

def main():
    print("Training Hand Gesture Recognition Models...")
    
    choice = input("Train which model? (1: Keypoint, 2: Point History, 3: Both): ")
    
    if choice == '1' or choice == '3':
        print("Training keypoint classification model...")
        train_keypoint()
        print("Keypoint model training completed!")
    
    if choice == '2' or choice == '3':
        print("Training point history classification model...")
        train_point_history()
        print("Point history model training completed!")
    
    print("Training completed!")

if __name__ == '__main__':
    main()