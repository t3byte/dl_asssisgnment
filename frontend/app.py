import gradio as gr
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, LSTM, Dense, Dropout, GlobalAveragePooling2D
import pickle
import os
import h5py

# Load label encoder
le_path = os.path.join("..", "backend", "label_encoder.pkl")
le = pickle.load(open(le_path, "rb"))

FRAMES = 10
IMG_SIZE = 160
NUM_CLASSES = 5  # Based on 5 actions

# Rebuild model architecture manually to match the saved model
def create_model():
    """Recreate the model architecture"""
    # Create CNN base
    cnn = MobileNetV2(
        include_top=False,
        weights=None,  # Don't load pretrained weights
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    cnn.trainable = False
    
    # Create full model
    model = Sequential([
        TimeDistributed(cnn, input_shape=(FRAMES, IMG_SIZE, IMG_SIZE, 3)),
        TimeDistributed(GlobalAveragePooling2D()),
        LSTM(64, return_sequences=False),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model

# Create model and load weights
model_path = os.path.join("..", "backend", "action_model.h5")
try:
    # Try loading weights only
    model = create_model()
    model.load_weights(model_path)
    print("Model weights loaded successfully!")
except Exception as e:
    print(f"Error loading model weights: {e}")
    print("Trying full model load...")
    try:
        # Fallback: try loading full model
        model = tf.keras.models.load_model(model_path, compile=False)
        print("Full model loaded successfully!")
    except Exception as e2:
        print(f"Error loading full model: {e2}")
        model = None

FRAMES = 10
IMG_SIZE = 160

def predict_action(video_path):
    """
    Predict action from video file
    Args:
        video_path: Path to uploaded video file
    Returns:
        str: Predicted action label
    """
    if video_path is None:
        return "Please upload a video file"
    
    try:
        # Open video file
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # Extract frames
        while len(frames) < FRAMES:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = frame / 255.0
            frames.append(frame)
        
        cap.release()
        
        # Check if we have enough frames
        if len(frames) != FRAMES:
            return f"Error: Video must have at least {FRAMES} frames. Found {len(frames)} frames."
        
        # Make prediction
        X = np.array([frames])
        pred = model.predict(X)
        label = le.inverse_transform([np.argmax(pred)])
        confidence = np.max(pred) * 100
        
        return f"Predicted Action: {label[0]}\nConfidence: {confidence:.2f}%"
    
    except Exception as e:
        return f"Error processing video: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Action Recognition") as demo:
    gr.Markdown("# 🎬 Action Recognition System")
    gr.Markdown("Upload a video to predict the action being performed")
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload Video", sources=["upload"])
            predict_btn = gr.Button("Predict Action", variant="primary")
        
        with gr.Column():
            output = gr.Textbox(label="Prediction Result", lines=3)
    
    predict_btn.click(
        fn=predict_action,
        inputs=video_input,
        outputs=output
    )
    
    gr.Markdown("### Instructions:")
    gr.Markdown(f"- Upload a video file (MP4, AVI, etc.)")
    gr.Markdown(f"- The model will extract the first {FRAMES} frames")
    gr.Markdown(f"- Each frame will be resized to {IMG_SIZE}x{IMG_SIZE} pixels")
    gr.Markdown("- Click 'Predict Action' to see the result")

if __name__ == "__main__":
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
