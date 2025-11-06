#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demo runner script for hand gesture recognition
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import main

if __name__ == '__main__':
    print("Starting Hand Gesture Recognition Demo...")
    print("Controls:")
    print("- ESC: Exit")
    print("- k: Enter keypoint logging mode")
    print("- h: Enter point history logging mode")
    print("- n: Normal mode")
    print("- 0-9: Log data with corresponding class ID")
    main()