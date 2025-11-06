#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import main

if __name__ == '__main__':
    # Set default arguments for demo
    sys.argv = ['run.py', '--device', '0', '--width', '1280', '--height', '720', 
                '--min_detection_confidence', '0.8', '--min_tracking_confidence', '0.9']
    main()