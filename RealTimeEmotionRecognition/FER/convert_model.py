import torch
import tensorflow as tf
import torch.nn as nn
import timm
import numpy as np
import os

def convert_pytorch_to_tfjs():
    # Create PyTorch model architecture
    pytorch_model = timm.create_model("tf_efficientnet_b0_ns", pretrained=False)
    pytorch_model.classifier = nn.Sequential(nn.Linear(in_features=1280, out_features=7))
    
    # Load trained weights
    pytorch_model.load_state_dict(torch.load("22.6_AffectNet_10K_part2.pt", map_location="cpu"))
    pytorch_model.eval()
    
    # Create equivalent TensorFlow model
    tf_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
        tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
        tf.keras.layers.experimental.preprocessing.Normalization(
            mean=[0.485, 0.456, 0.406],
            variance=[0.229**2, 0.224**2, 0.225**2]
        )
    ])
    
    # Copy weights from PyTorch to TensorFlow model
    with torch.no_grad():
        # Create a dummy input to trace the PyTorch model
        dummy_input = torch.randn(1, 3, 224, 224)
        torch_output = pytorch_model(dummy_input)
        
        # Convert PyTorch model to ONNX format first
        torch.onnx.export(pytorch_model,
                         dummy_input,
                         "temp_model.onnx",
                         input_names=['input'],
                         output_names=['output'],
                         dynamic_axes={'input': {0: 'batch_size'},
                                     'output': {0: 'batch_size'}})
        
        # Convert ONNX to TensorFlow
        os.system("python -m tf2onnx.convert --input temp_model.onnx --output model.pb --inputs input:0 --outputs output:0")
        
        # Load the TensorFlow model
        tf_model = tf.saved_model.load("model.pb")
        
        # Convert to TensorFlow.js format
        output_dir = 'public/models/emotion_model'
        os.makedirs(output_dir, exist_ok=True)
        os.system(f"tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model model.pb {output_dir}")
        
        # Clean up temporary files
        os.remove("temp_model.onnx")
        os.remove("model.pb")

if __name__ == "__main__":
    # First, install required dependencies
    os.system("pip install onnx tf2onnx")
    convert_pytorch_to_tfjs()