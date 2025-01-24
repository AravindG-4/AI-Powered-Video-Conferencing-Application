import io  # Added this missing import
import base64
import onnxruntime as ort
from ts.torch_handler.base_handler import BaseHandler
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
class ONNXEmotionHandler(BaseHandler):
    def __init__(self):
        super(ONNXEmotionHandler, self).__init__()
        self.onnx_session = None
        self.transform = None
        self.label_dict = {
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

    def initialize(self, context):
        # Load the ONNX model
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        onnx_model_path = f"{model_dir}/emotion_model.onnx"
        self.onnx_session = ort.InferenceSession(onnx_model_path)

        # Define preprocessing transformations
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def preprocess(self, data):
        # Process the input data into a format suitable for the model
        image = data[0].get("data") or data[0].get("body")
        if isinstance(image, (bytearray, bytes)):
            image = Image.open(io.BytesIO(image)).convert("RGB")

        # Save the original image for visualization
        self.original_image = image.copy()

        # Apply transformations
        image_tensor = self.transform(image).unsqueeze(0).numpy()
        return image_tensor

    def inference(self, data):
        # Perform inference using ONNX model
        onnx_inputs = {self.onnx_session.get_inputs()[0].name: data}
        onnx_outputs = self.onnx_session.run(None, onnx_inputs)
        self.feature_map = onnx_outputs[0]  # Assuming this is the feature map
        predicted = np.argmax(onnx_outputs[0])
        return predicted

    def visualize_feature_map(self):
        # Generate a heatmap from the feature map
        feature_map = self.feature_map[0]  # Assuming feature_map is [1, C, H, W]
        heatmap = np.mean(feature_map, axis=0)  # Average across channels
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)  # Normalize to [0, 1]

        # Resize heatmap to match the original image size
        heatmap = Image.fromarray(np.uint8(cm.jet(heatmap) * 255))  # Apply colormap
        heatmap = heatmap.resize(self.original_image.size, Image.BILINEAR)

        # Overlay heatmap on the original image
        overlay_image = Image.blend(self.original_image, heatmap, alpha=0.5)

        # Convert the image to base64
        buffered = io.BytesIO()
        overlay_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def postprocess(self, data):
        # Generate the visualization
        heatmap_overlay = self.visualize_feature_map()

        # Convert the output to a human-readable label and include the visualization
        predicted_label = self.label_dict.get(data, "Unknown")
        return {
            "label": predicted_label,
            "heatmap_overlay": heatmap_overlay
        }
