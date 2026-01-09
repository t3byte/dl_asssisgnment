from fastapi import FastAPI, UploadFile
import cv2
import numpy as np
import tensorflow as tf
import pickle
import tempfile
import os

app = FastAPI()

model = tf.keras.models.load_model("action_model.h5")
le = pickle.load(open("label_encoder.pkl", "rb"))

FRAMES = 10
IMG_SIZE = 160

@app.post("/predict")
async def predict_action(file: UploadFile):
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(await file.read())
    temp.close()

    cap = cv2.VideoCapture(temp.name)
    frames = []

    while len(frames) < FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = frame / 255.0
        frames.append(frame)

    cap.release()
    os.remove(temp.name)

    if len(frames) != FRAMES:
        return {"error": "Not enough frames"}

    X = np.array([frames])
    pred = model.predict(X)
    label = le.inverse_transform([np.argmax(pred)])

    return {"action": label[0]}