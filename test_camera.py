#!/usr/bin/env python
import cv2 as cv
from utils.camera import CameraCapture

def main():
    camera = CameraCapture(640, 480)
    
    print("Camera test started. Press 'q' to quit.")
    
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        cv.imshow('Camera Test', frame)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    camera.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()