#!/usr/bin/env python
import cv2 as cv
import numpy as np
import mediapipe as mp
import copy
import itertools
import time
import requests
from collections import Counter
from picamera2 import Picamera2
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

def send_to_api(sentence):
    try:
        payload = {"title": "Message", "body": sentence}
        print(f"[API] Sending request to http://172.16.248.252:3000/api/send-notification")
        print(f"[API] Payload: {payload}")
        response = requests.post('http://172.16.248.252:3000/api/send-notification', 
                               json=payload,
                               timeout=5)
        print(f"[API] Response status: {response.status_code}")
        print(f"[API] Response text: {response.text}")
    except Exception as e:
        print(f"[API] Error: {e}")

def main():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (1280, 720)})
    picam2.configure(config)
    picam2.start()
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.9)
    
    keypoint_classifier = KeyPointClassifier()
    gesture_labels = ['Help', 'Me', 'Call', 'fine', 'OK', 'unlock', 'lock', 'I', 'hello', 'bye']
    
    print("Sentence Recording Demo")
    print("Press SPACE to start 10-second recording")
    print("Press ESC to exit")
    
    recording = False
    recorded_gestures = []
    start_time = 0
    
    while True:
        image = picam2.capture_array()
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)
        
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            recording = True
            recorded_gestures = []
            start_time = time.time()
            print(f"[RECORDING] Started at {time.strftime('%H:%M:%S')}")
            print(f"[RECORDING] Will record for 10 seconds...")
        
        if recording:
            elapsed = time.time() - start_time
            if elapsed >= 10:
                recording = False
                print(f"[RECORDING] 10 seconds completed")
                print(f"[RECORDING] Total gestures captured: {len(recorded_gestures)}")
                print(f"[RECORDING] Raw gestures: {recorded_gestures}")
                if recorded_gestures:
                    most_common = Counter(recorded_gestures).most_common()
                    sentence = ' '.join([gesture for gesture, _ in most_common])
                    print(f"[RECORDING] Final sentence: {sentence}")
                    send_to_api(sentence)
                else:
                    print(f"[RECORDING] No gestures recorded")
            else:
                cv.putText(debug_image, f"Recording: {10-int(elapsed)}s", (10, 60), 
                          cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        
        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                gesture_text = gesture_labels[hand_sign_id] if hand_sign_id < len(gesture_labels) else "Unknown"
                
                if recording and gesture_text != "Unknown":
                    recorded_gestures.append(gesture_text)
                    print(f"[GESTURE] Detected: {gesture_text} (Total: {len(recorded_gestures)})")
                
                mp.solutions.drawing_utils.draw_landmarks(debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv.putText(debug_image, gesture_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        cv.imshow('Sentence Demo', debug_image)
    
    picam2.stop()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()