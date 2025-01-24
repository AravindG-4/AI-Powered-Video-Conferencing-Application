import cv2
import onnxruntime as ort
import torchvision.transforms as transforms
from PIL import Image
import mediapipe as mp
import time
import numpy as np

onnx_model_path = "emotion_model.onnx"  
onnx_session = ort.InferenceSession(onnx_model_path)

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

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

                # Check if the bounding box has a valid face region (non-zero width and height)
                if w > 0 and h > 0:
                    # Extract the region of interest (the face)
                    face = frame[y: y + h, x: x + w]

                    if face.size == 0:
                        continue  # Skip this iteration if the face region is invalid

                    # Convert the face to a PIL image
                    face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

                    # Apply transformations
                    face_tensor = transform(face_pil).unsqueeze(0).numpy()

                    # Run inference using ONNX model
                    onnx_inputs = {onnx_session.get_inputs()[0].name: face_tensor}
                    onnx_outputs = onnx_session.run(None, onnx_inputs)
                    predicted = np.argmax(onnx_outputs[0])

                    # Get the predicted label
                    label = label_dict[predicted]

                    # Draw a rectangle around the face and label it with the prediction
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

                    # Detect emotion changes
                    if predicted != prev_emotion:
                        current_time = time.time() - begin
                        if prev_emotion is not None:
                            print(f"{label} at {current_time}")
                            change_list.append(f"Change detected: {label_dict[prev_emotion]} -> {label} at {current_time}")
                    prev_emotion = predicted

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

        # Display the resulting frame in the OpenCV window
        cv2.imshow("Emotion Detection", frame)

        # Exit the video playback when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

