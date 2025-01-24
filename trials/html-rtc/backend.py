from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Dict, List
import base64
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import mediapipe as mp

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with a specific origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AI Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the emotion detection model
model = torch.load("22.6_AffectNet_10K_part2.pt", map_location=device)
model.to(device)
model.eval()

label_dict = {
    0: "angry", 1: "disgust", 2: "fear", 3: "happy",
    4: "neutral", 5: "sad", 6: "surprised", 7: "No_Face",
}

mp_face_detection = mp.solutions.face_detection


class ConnectionManager:
    def __init__(self):  # Correct method name
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, room_id: str, websocket: WebSocket):
        await websocket.accept()
        if room_id not in self.active_connections:
            self.active_connections[room_id] = []
        self.active_connections[room_id].append(websocket)

    def disconnect(self, room_id: str, websocket: WebSocket):
        if room_id in self.active_connections:
            self.active_connections[room_id].remove(websocket)

    async def broadcast(self, room_id: str, message: str):
        if room_id in self.active_connections:
            for connection in self.active_connections[room_id]:
                await connection.send_text(message)


manager = ConnectionManager()


@app.websocket("/ws/{room_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str):
    await manager.connect(room_id, websocket)
    try:
        while True:
            data = await websocket.receive_text()
            img_data = base64.b64decode(data)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.8) as face_detection:
                results = face_detection.process(rgb_frame)
                emotion = "No_Face"

                if results.detections:
                    for detection in results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, _ = frame.shape
                        x, y, w, h = (
                            int(bboxC.xmin * iw),
                            int(bboxC.ymin * ih),
                            int(bboxC.width * iw),
                            int(bboxC.height * ih),
                        )
                        face = frame[y:y + h, x:x + w]
                        if face.size == 0:
                            continue

                        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                        face_tensor = transform(face_pil).unsqueeze(0).to(device)

                        with torch.no_grad():
                            output = model(face_tensor)
                            _, predicted = torch.max(output, 1)

                        emotion = label_dict[predicted.item()]
                        break

            await manager.broadcast(room_id, f"{websocket.client.host}: {emotion}")
    except WebSocketDisconnect:
        manager.disconnect(room_id, websocket)
        await manager.broadcast(room_id, f"User disconnected: {websocket.client.host}")