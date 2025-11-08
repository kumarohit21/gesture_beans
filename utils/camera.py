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
                cpuinfo = f.read()
                print(f"[CAMERA] CPU Info check: {'BCM' in cpuinfo or 'Raspberry Pi' in cpuinfo}")
                return 'BCM' in cpuinfo or 'Raspberry Pi' in cpuinfo
        except Exception as e:
            print(f"[CAMERA] Cannot read /proc/cpuinfo: {e}")
            return False
    
    def _init_camera(self):
        print(f"[CAMERA] Platform: {platform.system()} {platform.machine()}")
        print(f"[CAMERA] Is Raspberry Pi: {self.is_raspberry_pi}")
        
        if self.is_raspberry_pi:
            try:
                print("[CAMERA] Attempting to import PiCamera2...")
                from picamera2 import Picamera2
                print("[CAMERA] PiCamera2 imported successfully")
                
                self.camera = Picamera2()
                print("[CAMERA] PiCamera2 instance created")
                
                config = self.camera.create_preview_configuration(main={"size": (self.width, self.height)})
                print(f"[CAMERA] Configuration created: {self.width}x{self.height}")
                
                self.camera.configure(config)
                print("[CAMERA] Camera configured")
                
                self.camera.start()
                print(f"[CAMERA] ✓ PiCamera2 started successfully ({self.width}x{self.height})")
                
            except ImportError as e:
                print(f"[CAMERA] PiCamera2 import failed: {e}")
                print("[CAMERA] Falling back to OpenCV VideoCapture")
                self._init_opencv()
            except Exception as e:
                print(f"[CAMERA] PiCamera2 initialization failed: {e}")
                print("[CAMERA] Falling back to OpenCV VideoCapture")
                self._init_opencv()
        else:
            print("[CAMERA] Not a Raspberry Pi, using OpenCV VideoCapture")
            self._init_opencv()
    
    def _init_opencv(self):
        print("[CAMERA] Initializing OpenCV VideoCapture...")
        self.camera = cv.VideoCapture(0)
        
        if not self.camera.isOpened():
            print("[CAMERA] ✗ Failed to open camera device 0")
            return
            
        self.camera.set(cv.CAP_PROP_FRAME_WIDTH, self.width)
        self.camera.set(cv.CAP_PROP_FRAME_HEIGHT, self.height)
        
        actual_width = int(self.camera.get(cv.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.camera.get(cv.CAP_PROP_FRAME_HEIGHT))
        
        print(f"[CAMERA] ✓ OpenCV VideoCapture opened (requested: {self.width}x{self.height}, actual: {actual_width}x{actual_height})")
    
    def read(self):
        if self.is_raspberry_pi and hasattr(self.camera, 'capture_array'):
            try:
                frame = self.camera.capture_array()
                frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
                return True, frame
            except Exception as e:
                print(f"[CAMERA] PiCamera2 read error: {e}")
                return False, None
        else:
            return self.camera.read()
    
    def release(self):
        print("[CAMERA] Releasing camera...")
        if self.is_raspberry_pi and hasattr(self.camera, 'stop'):
            try:
                self.camera.stop()
                print("[CAMERA] PiCamera2 stopped")
            except Exception as e:
                print(f"[CAMERA] Error stopping PiCamera2: {e}")
        else:
            try:
                self.camera.release()
                print("[CAMERA] OpenCV VideoCapture released")
            except Exception as e:
                print(f"[CAMERA] Error releasing OpenCV camera: {e}")