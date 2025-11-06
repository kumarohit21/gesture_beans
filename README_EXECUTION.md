# Hand Gesture Recognition - Execution Guide

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

### Method 1: Using the main runner script
```bash
# Run demo
python run.py demo

# Train models
python run.py train
```

### Method 2: Direct execution
```bash
# Run demo directly
python scripts/run_demo.py

# Train models directly
python scripts/train_models.py

# Original app (with arguments)
python app.py --device 0 --width 960 --height 540
```

## Training Models

### Train Keypoint Classifier
```bash
python training/keypoint_training.py
```

### Train Point History Classifier
```bash
python training/point_history_training.py
```

## Controls During Demo

- **ESC**: Exit application
- **k**: Enter keypoint logging mode
- **h**: Enter point history logging mode  
- **n**: Normal mode
- **0-9**: Log data with corresponding class ID

## File Structure

```
├── training/           # Training scripts
│   ├── keypoint_training.py
│   └── point_history_training.py
├── scripts/           # Execution scripts
│   ├── run_demo.py
│   └── train_models.py
├── model/             # Model files
├── utils/             # Utility functions
├── run.py             # Main runner
├── app.py             # Original application
└── requirements.txt   # Dependencies
```