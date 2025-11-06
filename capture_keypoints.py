#!/usr/bin/env python
import csv
import cv2 as cv
import numpy as np
import mediapipe as mp
import itertools
import copy

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
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    mode = 0
    csv_path = 'model/keypoint_classifier_new/keypoint.csv'
    
    print("Press 'k' to enter keypoint logging mode")
    print("Press '0-9' to log keypoint with class ID")
    print("Press 'ESC' to exit")

    # 0 - Help
    # 1 - Me
    # 2 - Call
    # 3 - fine
    # 4 - OK
    # 5 - unlock
    # 6 - lock
    # 7 - I
    # 8 - hello
    # 9 - bye
    
    
    while True:
        ret, image = cap.read()
        if not ret:
            break
            
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)
        
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        elif key == 107:  # k
            mode = 1
        elif key == 110:  # n
            mode = 0
        
        number = -1
        if 48 <= key <= 57:  # 0-9
            number = key - 48
        
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                hand_label = handedness.classification[0].label
                
                if mode == 1 and (0 <= number <= 9):
                    with open(csv_path, 'a', newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([number, *pre_processed_landmark_list])
                    print(f"Saved {hand_label} hand keypoint for class {number}")
                
                mp.solutions.drawing_utils.draw_landmarks(debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        mode_text = "Logging Key Point" if mode == 1 else "Normal"
        cv.putText(debug_image, f"MODE: {mode_text}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv.imshow('Keypoint Capture', debug_image)
    
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()