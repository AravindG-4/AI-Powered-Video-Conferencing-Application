import torch

# Load the full PyTorch model
model_path = "emotion_model.pt"
pytorch_model = torch.load(model_path, map_location="cpu")
pytorch_model.eval()

# Define dummy input for the model
dummy_input = torch.randn(1, 3, 224, 224)

# Define the ONNX file path
onnx_model_path = "emotion_model.onnx"

# Export the PyTorch model to ONNX
torch.onnx.export(
    pytorch_model,
    dummy_input,
    onnx_model_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)

print(f"Model has been successfully converted to ONNX and saved at {onnx_model_path}")
