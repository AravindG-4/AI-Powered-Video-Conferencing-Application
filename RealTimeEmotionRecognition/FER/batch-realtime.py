import cv2
import onnxruntime as ort
import torchvision.transforms as transforms
from PIL import Image
import mediapipe as mp
import time
import numpy as np

# ONNX model path
onnx_model_path = "emotion_model.onnx"

# Use the GPU for inference by setting CUDA execution provider
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
onnx_session = ort.InferenceSession(onnx_model_path, providers=providers)

# Preprocessing pipeline for input image
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# MediaPipe face detection setup
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

label_dict = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprised",
    7: "No_Face", 
    8: "Body",  
}

# Open three camera streams
cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(2)

# Check if cameras are opened
if not cap0.isOpened() or not cap1.isOpened() or not cap2.isOpened():
    print("Error: Could not access one or more webcams.")
    exit()

# Store face detections for batch inference
face_batch = []

# Start timing for emotion change detection
prev_emotion = None
change_list = []
begin = time.time()

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.8) as face_detection:
    while True:
        # Capture frames from all three cameras
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret0 or not ret1 or not ret2:
            print("Error: Unable to capture frames from one or more cameras.")
            break

        # List to store frames from all cameras
        frames = [frame0, frame1, frame2]

        # Process each camera frame
        for idx, frame in enumerate(frames):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame and detect faces
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

                    # Extract face region
                    if w > 0 and h > 0:
                        face = frame[y: y + h, x: x + w]

                        if face.size == 0:
                            continue  # Skip if invalid face region

                        # Convert to PIL image and apply transformations
                        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                        face_tensor = transform(face_pil).unsqueeze(0).numpy()

                        # Add to batch
                        face_batch.append((face_tensor, (x, y, w, h)))

            else:
                # If no face detected, label as No_Face
                label = label_dict[7]  # No_Face
                cv2.putText(
                    frame,
                    f"{label}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

        # If batch size reached, perform inference
        if len(face_batch) >= 8:  # Batch size of 8
            # Stack faces and run inference
            batch_input = np.concatenate([item[0] for item in face_batch], axis=0)
            onnx_inputs = {onnx_session.get_inputs()[0].name: batch_input}
            onnx_outputs = onnx_session.run(None, onnx_inputs)
            predictions = np.argmax(onnx_outputs[0], axis=1)

            # Process the results for each face in the batch
            for i, predicted in enumerate(predictions):
                label = label_dict[predicted]
                (x, y, w, h) = face_batch[i][1]

                # Draw a rectangle around the face and label it with the prediction
                frame_idx = i % 3  # Determine which camera the face came from
                frame = frames[frame_idx]

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(
                    frame,
                    f"{label}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 0, 0),
                    2,
                )

            # Reset the face batch after processing
            face_batch = []

        # Display the frames from all cameras
        cv2.imshow("Camera 0 - Emotion Detection", frame0)
        cv2.imshow("Camera 1 - Emotion Detection", frame1)
        cv2.imshow("Camera 2 - Emotion Detection", frame2)

        # Exit the video playback when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap0.release()
cap1.release()
cap2.release()
cv2.destroyAllWindows()
