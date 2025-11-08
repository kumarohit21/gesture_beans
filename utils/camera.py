#!/usr/bin/env python
import cv2 as cv
import platform
import sys

class CameraCapture:
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        self.is_raspberry_pi = self._is_raspberry_pi()
        self.camera = None
        self._init_camera()
    
    def _is_raspberry_pi(self):
        try:
            with open('/proc/cpuinfo', 'r') as f:
                return 'BCM' in f.read()
        except:
            return False
    
    def _init_camera(self):
        if self.is_raspberry_pi:
            try:
                from picamera2 import Picamera2
                self.camera = Picamera2()
                config = self.camera.create_preview_configuration(main={"size": (self.width, self.height)})
                self.camera.configure(config)
                self.camera.start()
                print(f"[CAMERA] Using PiCamera2 ({self.width}x{self.height})")
            except ImportError:
                print("[CAMERA] PiCamera2 not available, falling back to OpenCV")
                self._init_opencv()
        else:
            self._init_opencv()
    
    def _init_opencv(self):
        self.camera = cv.VideoCapture(0)
        self.camera.set(cv.CAP_PROP_FRAME_WIDTH, self.width)
        self.camera.set(cv.CAP_PROP_FRAME_HEIGHT, self.height)
        print(f"[CAMERA] Using OpenCV VideoCapture ({self.width}x{self.height})")
    
    def read(self):
        if self.is_raspberry_pi and hasattr(self.camera, 'capture_array'):
            frame = self.camera.capture_array()
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            return True, frame
        else:
            return self.camera.read()
    
    def release(self):
        if self.is_raspberry_pi and hasattr(self.camera, 'stop'):
            self.camera.stop()
        else:
            self.camera.release()