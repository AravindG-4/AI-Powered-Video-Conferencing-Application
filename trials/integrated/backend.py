from fastapi import FastAPI, File, UploadFile
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import mediapipe as mp

# Initialize the FastAPI app
app = FastAPI()

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# Define the transformation to apply to the images
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Load the emotion detection model
model = torch.load("emotion_model.pt", map_location=device)  # Load the full model
model.to(device)  # Move model to GPU if available
model.eval()

# Labels for emotion classes
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

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection

@app.post("/process_frame/")
async def process_frame(file: UploadFile = File(...)):
    """
    Process a single frame and return the detected emotion.
    """
    try:
        # Read the file
        file_bytes = await file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Initialize MediaPipe Face Detection
        with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.8) as face_detection:
            results = face_detection.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    # Get bounding box
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = (
                        int(bboxC.xmin * iw),
                        int(bboxC.ymin * ih),
                        int(bboxC.width * iw),
                        int(bboxC.height * ih),
                    )

                    # Extract the face region
                    face = frame[y: y + h, x: x + w]

                    if face.size == 0:
                        continue

                    # Convert the face to a PIL image
                    face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

                    # Apply transformations
                    face_tensor = transform(face_pil).unsqueeze(0).to(device)  # Move tensor to GPU if available

                    # Predict emotion
                    with torch.no_grad():
                        output = model(face_tensor)
                        _, predicted = torch.max(output, 1)

                    emotion = label_dict[predicted.item()]
                    return {"emotion": emotion}

            return {"emotion": label_dict[7]}  # No face detected

    except Exception as e:
        return {"error": str(e)}
