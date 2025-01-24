import cv2
import onnxruntime as ort
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from typing import List
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import asyncio
import uvicorn

app = FastAPI()

# Initialize ONNX session with GPU provider
onnx_model_path = "emotion_model.onnx"
onnx_session = ort.InferenceSession(
    onnx_model_path
)

# Preprocessing transformations
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Emotion label dictionary
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

# Queue for batch processing
batch_queue = Queue(maxsize=100)
batch_size = 16  # Define batch size for inference
executor = ThreadPoolExecutor(max_workers=4)


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Preprocesses a single frame for model input."""
    face_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    face_tensor = transform(face_pil).unsqueeze(0).numpy()
    return face_tensor


def batch_inference(frames: List[np.ndarray]) -> List[str]:
    """Perform batch inference on a list of preprocessed frames."""
    batch = np.vstack(frames)
    onnx_inputs = {onnx_session.get_inputs()[0].name: batch}
    predictions = onnx_session.run(None, onnx_inputs)
    predicted_labels = [label_dict[np.argmax(pred)] for pred in predictions[0]]
    return predicted_labels


@app.post("/detect-emotions/")
async def detect_emotions(files: List[UploadFile] = File(...)):
    """Endpoint to handle emotion detection for uploaded images."""
    tasks = []
    for file in files:
        content = await file.read()
        frame = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        tasks.append(preprocess_frame(frame))

    # Process batch
    predictions = await asyncio.get_event_loop().run_in_executor(
        executor, batch_inference, tasks
    )

    return {"predictions": predictions}


def batch_processor():
    """Process batches from the queue."""
    while True:
        if batch_queue.qsize() >= batch_size:
            frames = [batch_queue.get() for _ in range(batch_size)]
            predictions = batch_inference(frames)
            # Log or return predictions as needed
            print(predictions)


# Start batch processor in the background
loop = asyncio.get_event_loop()
loop.run_in_executor(executor, batch_processor)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
