from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import librosa
import torch
import numpy as np
import uvicorn

# Load model and feature extractor during app initialization for efficiency
model_id = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
model = AutoModelForAudioClassification.from_pretrained(model_id)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, do_normalize=True)
id2label = model.config.id2label

# FastAPI app instance
app = FastAPI()

def preprocess_audio(audio_path: str, feature_extractor, max_duration: float = 30.0):
    audio_array, sampling_rate = librosa.load(audio_path, sr=feature_extractor.sampling_rate)
    max_length = int(feature_extractor.sampling_rate * max_duration)
    if len(audio_array) > max_length:
        audio_array = audio_array[:max_length]
    else:
        audio_array = np.pad(audio_array, (0, max_length - len(audio_array)))

    inputs = feature_extractor(
        audio_array,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    return inputs


async def predict_emotion(file_path: str, model, feature_extractor, id2label, max_duration: float = 30.0):
    inputs = preprocess_audio(file_path, feature_extractor, max_duration)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_id = torch.argmax(logits, dim=-1).item()
    predicted_label = id2label[predicted_id]

    return predicted_label


class EmotionPredictionRequest(BaseModel):
    file_paths: list[str]


@app.post("/predict-emotions/", response_model=dict)
async def batch_predict_emotions(request: EmotionPredictionRequest):
    responses = {}

    for file_path in request.file_paths:
        try:
            # Predict the emotion for each file
            predicted_emotion = await predict_emotion(file_path, model, feature_extractor, id2label)
            responses[file_path] = predicted_emotion
        except Exception as e:
            responses[file_path] = f"Error: {str(e)}"

    return JSONResponse(content=responses)


@app.post("/predict-emotion/")
async def predict_emotion_endpoint(file: UploadFile = File(...)):
    try:
        file_location = f"./{file.filename}"
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())

        # Perform prediction
        predicted_emotion = await predict_emotion(file_location, model, feature_extractor, id2label)
        return {"file_name": file.filename, "predicted_emotion": predicted_emotion}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
