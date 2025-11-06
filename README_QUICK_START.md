# Hand Gesture Recognition - Quick Start Guide

## Setup
```bash
pip install -r requirements.txt
```

## 1. Capture Training Data

Run the keypoint capture tool:
```bash
python capture_keypoints.py
```

**Controls:**
- Press `k` to enter logging mode
- Press `0-9` to capture gestures for each class
- Press `n` for normal mode
- Press `ESC` to exit

**Gesture Classes:**
- `0` - Help
- `1` - Me  
- `2` - Call
- `3` - fine
- `4` - OK
- `5` - unlock
- `6` - lock
- `7` - I
- `8` - hello
- `9` - bye

**Tips:**
- Capture 50-100 samples per gesture
- Use both left and right hands
- Vary hand positions and angles
- Ensure good lighting

## 2. Train Model

After collecting data, train the model:
```bash
python train_keypoint.py
```

The script will:
- Load your captured keypoints
- Train a neural network
- Save the trained model (.keras and .tflite)
- Show training progress and accuracy

## 3. Run Demo

Test your trained model:
```bash
python demo_gestures.py
```

**Demo Features:**
- Real-time gesture recognition
- Dual hand support
- Live gesture labels
- Press `ESC` to exit

## Alternative Demo
```bash
python run.py
```

## File Structure
```
model/keypoint_classifier_new/
├── keypoint.csv                    # Training data
├── keypoint_classifier.keras       # Trained model
├── keypoint_classifier.tflite      # Optimized model
└── keypoint_classifier_label.csv   # Gesture labels
```

## Troubleshooting

**No training data found:**
- Run `capture_keypoints.py` first
- Ensure you pressed `k` and captured samples

**Low accuracy:**
- Collect more training samples
- Ensure consistent gesture performance
- Check lighting conditions

**Camera issues:**
- Verify camera permissions
- Try different camera device (change device=0 to device=1)