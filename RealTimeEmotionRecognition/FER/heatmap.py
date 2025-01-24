import cv2
import onnxruntime as ort
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch
import torchvision.models as models
from mediapipe import solutions as mp_solutions

# Load a pretrained model for demonstration (use your ONNX model converted to PyTorch if needed)
model = models.resnet18(pretrained=True)
target_layer = model.layer4[-1]  # Grad-CAM target layer

# Initialize Grad-CAM
cam = GradCAM(model=model, target_layers=[target_layer])

# ONNX Session for your model
onnx_model_path = "emotion_model.onnx"
onnx_session = ort.InferenceSession(onnx_model_path)

# Transformation pipeline
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

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

# Mediapipe face detection setup
mp_face_detection = mp_solutions.face_detection
mp_drawing = mp_solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

cv2.namedWindow("Original Video", cv2.WINDOW_NORMAL)
cv2.namedWindow("Cropped Heatmap", cv2.WINDOW_NORMAL)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.8) as face_detection:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture webcam frame.")
            break

        # Convert frame to RGB and preprocess for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        results = face_detection.process(rgb_frame)
        cropped_heatmap = np.zeros_like(frame)

        if results.detections:
            for detection in results.detections:
                # Extract bounding box
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = (
                    int(bboxC.xmin * iw),
                    int(bboxC.ymin * ih),
                    int(bboxC.width * iw),
                    int(bboxC.height * ih),
                )

                # Crop the face region
                face = frame[y: y + h, x: x + w]
                if face.size == 0:
                    continue  # Skip invalid regions

                # Convert face to PIL Image and preprocess for ONNX
                face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                input_tensor = transform(face_pil).unsqueeze(0)

                # Run ONNX model inference
                onnx_inputs = {onnx_session.get_inputs()[0].name: input_tensor.numpy()}
                onnx_outputs = onnx_session.run(None, onnx_inputs)
                predicted = np.argmax(onnx_outputs[0])

                # Grad-CAM processing for the cropped face
                face_rgb = np.float32(face) / 255.0  # Normalize face for Grad-CAM
                face_tensor_for_cam = preprocess_image(face_rgb, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                targets = [ClassifierOutputTarget(predicted)]  # Focus on predicted class
                grayscale_cam = cam(input_tensor=face_tensor_for_cam, targets=targets)[0]  # Grad-CAM for cropped face

                # Generate the Grad-CAM heatmap overlay for the cropped face
                cam_image = show_cam_on_image(face_rgb, grayscale_cam, use_rgb=True)

                # Convert Grad-CAM overlay back to BGR for display
                cropped_heatmap = cv2.cvtColor((cam_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

                # Annotate the original frame with the predicted label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(
                    frame,
                    f"Emotion: {label_dict[predicted]}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 0, 0),
                    2,
                )

        # Display original video and cropped Grad-CAM heatmap
        cv2.imshow("Original Video", frame)
        cv2.resizeWindow("Original Video", 640, 480)  # Set window size to 640x480
        cv2.imshow("Cropped Heatmap", cropped_heatmap)
        cv2.resizeWindow("Cropped Heatmap", 640, 480)  # Set window size to 640x480


        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
