import cv2
import onnxruntime as ort
import torchvision.transforms as transforms
from PIL import Image
import mediapipe as mp
import time
import numpy as np

# Load ONNX model
onnx_model_path = "emotion_model.onnx"
# providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
onnx_session = ort.InferenceSession(onnx_model_path)

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# MediaPipe face detection setup
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Emotion labels
label_dict = {
    0: "angry", 1: "disgust", 2: "fear", 3: "happy",
    4: "neutral", 5: "sad", 6: "surprised", 7: "No_Face"
}

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

prev_emotion = None
change_list = []
begin = time.time()

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.8) as face_detection:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture webcam frame.")
            break

        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect faces
        results = face_detection.process(rgb_frame)

        face_tensors = []
        face_coords = []

        if results.detections:
            for detection in results.detections:
                # Get bounding box
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = (int(bboxC.xmin * iw), int(bboxC.ymin * ih),
                              int(bboxC.width * iw), int(bboxC.height * ih))

                if w > 0 and h > 0:
                    # Extract the face region
                    face = frame[y:y+h, x:x+w]

                    if face.shape[0] == 0 or face.shape[1] == 0:
                        continue  # Skip if invalid face

                    # Convert to PIL and apply transformations
                    face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                    face_tensor = transform(face_pil).unsqueeze(0)  # Shape: (1, 3, 224, 224)

                    face_tensors.append(face_tensor)
                    face_coords.append((x, y, w, h))

            if face_tensors:
                # Stack tensors to create a batch
                batch_faces = np.vstack(face_tensors)  # Shape: (batch_size, 3, 224, 224)

                # Run batch inference
                onnx_inputs = {onnx_session.get_inputs()[0].name: batch_faces}
                onnx_outputs = onnx_session.run(None, onnx_inputs)
                predictions = np.argmax(onnx_outputs[0], axis=1)  # Get predicted labels

                # Draw results on the frame
                for (x, y, w, h), predicted in zip(face_coords, predictions):
                    label = label_dict[predicted]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (255, 0, 0), 2)

                    # Detect emotion changes
                    if predicted != prev_emotion:
                        current_time = time.time() - begin
                        if prev_emotion is not None:
                            change_list.append(f"Change detected: {label_dict[prev_emotion]} -> {label} at {current_time}")
                    prev_emotion = predicted
        else:
            # If no face detected, label as No_Face
            cv2.putText(frame, "No_Face", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Emotion Detection", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
