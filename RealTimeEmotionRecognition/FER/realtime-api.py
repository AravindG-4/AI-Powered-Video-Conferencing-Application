import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import timm
import torch.nn as nn
import mediapipe as mp
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import base64
import numpy as np
import json
from io import BytesIO
from .realtime import process_frame  # Assume process_frame is a function in realtime.py

app = FastAPI()

# Update CORS middleware settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize device
device = "cpu"

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the model
model = timm.create_model("tf_efficientnet_b0_ns", pretrained=False)
model.classifier = nn.Sequential(nn.Linear(in_features=1280, out_features=7))
model = torch.load("22.6_AffectNet_10K_part2.pt", map_location=device)
model.to(device)
model.eval()

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.8)

# Labels dictionary
label_dict = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprised",
    7: "No_Face",
}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_bytes()
        frame = np.frombuffer(data, dtype=np.uint8).reshape((480, 640, 3))  # Example shape
        emotion = process_frame(frame)  # Function to process frame and return emotion
        await websocket.send_text(emotion)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)