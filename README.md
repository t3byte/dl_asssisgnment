# Action Recognition with Deep Learning

This project implements an action recognition system using LSTM and CNN (MobileNetV2) for classifying human actions in videos.

## Dataset

UCF50 - 5 selected actions:
- WalkingWithDog
- JumpingJack
- PushUps
- Punch
- Biking

## Model Architecture

- **Base CNN**: MobileNetV2 (pretrained, frozen)
- **Temporal Processing**: LSTM (64 units)
- **Input**: 10 frames per video (160x160 pixels)
- **Output**: 5 action classes

## Project Structure

```
action_recognition/
├── DL_Assignment_4.ipynb    # Training notebook
├── backend/
│   ├── action_model.h5      # Trained model (not in git)
│   ├── label_encoder.pkl    # Label encoder (not in git)
│   └── app.py               # Backend API
├── frontend/
│   └── app.py               # Gradio web interface
└── README.md
```

## Setup

1. Create virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install tensorflow==2.13.1 gradio opencv-python scikit-learn numpy
```

3. Run the Gradio interface:
```bash
cd frontend
python app.py
```

4. Open browser at: `http://127.0.0.1:7860`

## Usage

1. Upload a video file
2. Click "Predict Action"
3. View the predicted action and confidence scores

## Model Training

See `DL_Assignment_4.ipynb` for the complete training process.

## Author

Muhammad Usman Tahir  
Roll No: 221463  
Course: Deep Learning
