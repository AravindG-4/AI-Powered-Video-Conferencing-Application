import io  # Added this missing import
import logging
import onnxruntime as ort
import torch
from ts.torch_handler.base_handler import BaseHandler
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)

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
        """
        Initializes the ONNX model and sets up the transformation pipeline.
        """
        try:
            self.manifest = context.manifest
            properties = context.system_properties
            model_dir = properties.get("model_dir")
            onnx_model_path = f"{model_dir}/emotion_model.onnx"

            self.onnx_session = ort.InferenceSession(onnx_model_path)
            logging.info(f"Loaded ONNX model from {onnx_model_path}")
        except Exception as e:
            logging.error(f"Error loading ONNX model: {e}")
            raise RuntimeError("Failed to load ONNX model.")

        # Define preprocessing transformations
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        logging.info("Preprocessing transformations set up successfully.")

    def preprocess(self, data):
        """
        Preprocess the input data for inference.
        Handles both single and batch inputs.
        """
        logging.info("Starting preprocessing...")
        try:
            images = []
            for idx, item in enumerate(data):
                # Extract image data
                image = item.get("data") or item.get("body")
                if not image:
                    raise ValueError(f"No image data found in input at index {idx}.")

                if isinstance(image, (bytearray, bytes)):
                    image = Image.open(io.BytesIO(image)).convert("RGB")
                else:
                    raise TypeError(f"Unsupported image type: {type(image)} at index {idx}.")

                # Apply transformations
                images.append(self.transform(image).unsqueeze(0))

            # Stack all images into a batch
            batch = torch.cat(images, dim=0).numpy()
            logging.info("Preprocessing completed successfully.")
            return batch
        except Exception as e:
            logging.error(f"Error during preprocessing: {e}")
            raise RuntimeError("Failed during preprocessing.")

    def inference(self, data):
        """
        Perform inference using the ONNX model.
        Handles batch inputs and returns predictions for all inputs.
        """
        logging.info("Starting inference...")
        try:
            # Pass the batch to the ONNX model
            onnx_inputs = {self.onnx_session.get_inputs()[0].name: data}
            onnx_outputs = self.onnx_session.run(None, onnx_inputs)

            # Get predictions for each input in the batch
            predictions = np.argmax(onnx_outputs[0], axis=1)
            logging.info(f"Inference completed. Predictions: {predictions}")
            return predictions
        except Exception as e:
            logging.error(f"Error during inference: {e}")
            raise RuntimeError("Failed during inference.")

    def postprocess(self, data):
        """
        Convert the model's output to a human-readable format.
        Handles batch predictions.
        """
        logging.info("Starting postprocessing...")
        try:
            # Map predictions to corresponding labels
            results = [{"label": self.label_dict.get(pred, "Unknown")} for pred in data]
            logging.info(f"Postprocessing completed. Results: {results}")
            return results
        except Exception as e:
            logging.error(f"Error during postprocessing: {e}")
            raise RuntimeError("Failed during postprocessing.")
