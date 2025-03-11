import cv2
import onnxruntime as ort
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from typing import List
import asyncio
import uvicorn
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

app = FastAPI()

# -------------------------------
# 🔹 Initialize ONNX Session (Use GPU with TensorRT)
# -------------------------------
onnx_model_path = "emotion_model.onnx"
onnx_session = ort.InferenceSession(
    onnx_model_path, 
    providers=["CUDAExecutionProvider"]
)

# -------------------------------
# 🔹 Preprocessing Transformations
# -------------------------------
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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

# -------------------------------
# 🔹 Async Batch Processing
# -------------------------------
batch_queue = asyncio.Queue(maxsize=100) 
batch_size = 16  

async def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Preprocess a single frame for the model."""
    face_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    face_tensor = transform(face_pil).unsqueeze(0).numpy().astype(np.float32)
    return face_tensor

async def batch_processor():
    """Continuously process batches from the queue in the background."""
    while True:
        frames = []
        for _ in range(batch_size):
            frame = await batch_queue.get()
            frames.append(frame)

        if frames:
            predictions = batch_inference(frames)
            print(f"Batch Processed: {predictions}")  # Replace with logging

def batch_inference(frames: List[np.ndarray]) -> List[str]:
    """Perform batch inference with ONNX model on GPU."""
    batch = np.vstack(frames)
    onnx_inputs = {onnx_session.get_inputs()[0].name: batch}
    predictions = onnx_session.run(None, onnx_inputs)
    predicted_labels = [label_dict[np.argmax(pred)] for pred in predictions[0]]
    return predicted_labels

# -------------------------------
# 🔹 API Endpoint for Emotion Detection
# -------------------------------
@app.post("/detect-emotions/")
async def detect_emotions(files: List[UploadFile] = File(...)):
    tasks = []
    for file in files:
        content = await file.read()
        frame = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        processed_frame = await preprocess_frame(frame)
        tasks.append(processed_frame)
    
    # Perform inference immediately (instead of queue-based batch processing)
    predictions = batch_inference(tasks)
    
    return {"predictions": predictions}

# -------------------------------
# 🔹 Background Task on Startup
# -------------------------------
async def start_background_tasks():
    loop = asyncio.get_running_loop()
    loop.create_task(batch_processor())

@app.on_event("startup")
async def startup_event():
    """Preload model and start batch processing."""
    await start_background_tasks()
    dummy_input = np.random.rand(1, 3, 224, 224).astype(np.float32)
    _ = onnx_session.run(None, {onnx_session.get_inputs()[0].name: dummy_input})
    print("✅ Model Preloaded on GPU")

# -------------------------------
# 🔹 Run FastAPI Server
# -------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=8)


