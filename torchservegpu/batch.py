import requests

# TorchServe prediction endpoint
TORCHSERVE_URL = "http://127.0.0.1:8080/predictions/emotion_model"

# Path to the test image
image_path = "surprised.jpg"

# Create a batch of 100 images
files = [("data", ("image.jpg", open(image_path, "rb").read())) for _ in range(100)]

# Send the batch request
response = requests.post(TORCHSERVE_URL, files=files)

# Check the response
if response.status_code == 200:
    try:
        predictions = response.json()
        if isinstance(predictions, list):
            for idx, prediction in enumerate(predictions):
                print(f"Image {idx + 1}: {prediction['label']}")
        else:
            print("Unexpected response format:", predictions)
    except Exception as e:
        print("Error parsing response:", e)
        print("Response Text:", response.text)
else:
    print(f"Error: {response.status_code}, {response.text}")
