#!/usr/bin/env python
import cv2 as cv
import numpy as np
import mediapipe as mp
import copy
import itertools
from utils.camera import CameraCapture
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    
    for index, landmark_point in enumerate(temp_landmark_list):
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    
    def normalize_(n):
        return n / max_value
    
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

def main():
    camera = CameraCapture(1280, 720)
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.8, min_tracking_confidence=0.9)
    
    keypoint_classifier = KeyPointClassifier()
    
    # Gesture labels
    gesture_labels = ['open', 'door', 'close', 'help', 'call', 'me', 'hello']
    
    print("Gesture Recognition Demo")
    print("Available gestures:", ', '.join(gesture_labels))
    print("Press ESC to exit")
    
    while True:
        ret, image = camera.read()
        if not ret:
            break
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)
        
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                hand_label = handedness.classification[0].label
                
                gesture_text = gesture_labels[hand_sign_id] if hand_sign_id < len(gesture_labels) else "Unknown"
                
                # Draw landmarks
                mp.solutions.drawing_utils.draw_landmarks(debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Display gesture
                cv.putText(debug_image, f"{hand_label}: {gesture_text}", (10, 50 if hand_label == "Left" else 80), 
                          cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        cv.imshow('Gesture Recognition Demo', debug_image)
    
    camera.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()