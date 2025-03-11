# AI-Powered Video Conferencing Application

## Overview
This project develops an **advanced emotion recognition system** for video conferences using a **multi-modal AI model** that processes both **video and audio** data to analyze participants' emotions in real-time. The system enhances virtual interactions by providing **live emotion heatmaps**, meeting summaries, and emotion analytics.

## Features 🚀
- **Real-time Emotion Recognition** 🧠🎭
  - Uses **EfficientNet-B0** trained on the **AffectNet dataset** for facial emotion detection.
  - Implements **OpenAI Whisper** for audio emotion recognition.
- **High Performance & Low Latency** ⚡
  - Built with **React + SimplePeer + TorchServe** for optimized real-time inference.
  - Handles **batch inference** and **auto-scaling** for multiple concurrent video conferences.
- **Meeting Summaries & Q&A** 📜🤖
  - Integrates **Retrieval-Augmented Generation (RAG)** to provide contextual meeting summaries and Q&A.
- **Scalability & Auto Scaling** 🏗️
  - Supports **high-throughput** processing with **TorchServe**.
  - Efficient handling of **multiple sessions simultaneously**.
- **Seamless Video Conferencing** 🎥
  - Uses **SimplePeer** for real-time video/audio streaming.
  - Optimized for **low-latency** interactions.

## Tech Stack 🛠️
- **Frontend:** React, SimplePeer
- **Backend:** TorchServe (Optimized Backend)
- **AI Models:**
  - **EfficientNet-B0** (Facial Emotion Recognition - Trained on AffectNet)
  - **OpenAI Whisper** (Audio Emotion Recognition)
  - **RAG-based System** (Meeting Summaries & Q&A)
- **Deployment & Scalability:**
  - Docker, Kubernetes, TorchServe, Auto Scaling

## Architecture 🏗️
1. **Video & Audio Streaming:** Captured via **SimplePeer** and sent to the backend.
2. **Inference Pipeline:**
   - **EfficientNet-B0** processes video frames for facial emotion recognition.
   - **Whisper Model** analyzes audio signals for tone-based emotion detection.
3. **Real-time Processing:**
   - TorchServe optimizes **low-latency inference**.
   - Results are displayed as **live emotion heatmaps**.
4. **Meeting Summarization & Q&A:**
   - RAG system extracts meaningful insights from conversations.
5. **Scalability & Multi-Session Support:**
   - **Batch inference** & auto-scaling enable efficient resource utilization.

## Performance Optimization ⚡
### **Approach 1: Streamlit + Streamlit WebRTC** ❌
- **High latency & slow inference** made it impractical for real-time video processing.

### **Approach 2: React + SimplePeer + FastAPI** ✅
- **Significant improvements** in inference speed but lacked advanced scalability features.

### **Final Approach: React + SimplePeer + TorchServe** 🚀
- **Low latency, high throughput, and optimized inference time**.
- Supports **batch inference & auto-scaling** for enterprise-level scalability.

## Installation & Setup 🛠️
### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/ai-video-conferencing.git
cd ai-video-conferencing
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

#### Frontend (React + SimplePeer)
```bash
cd frontend
npm install
```

### **3. Run the Application**
#### Start Frontend
```bash
cd frontend
npm start
```

#### Start Peer Server
```bash
cd peerserver
node peer-server.js
```

#### Start TorchServe Backend
```bash
cd torchserve
torchserve --start --ncs --model-store model_store --models emotion_model.mar --ts-config config.properties
```

### **4. Test the Model Locally**
```bash
curl -X POST http://127.0.0.1:8080/predictions/emotion_model -T surprised.jpg
```

## Initial Development Phases 🛠️
### **Phase 1: Streamlit + FastAPI (Initial Prototype - High Latency)**
- Used **Streamlit + FastAPI** for emotion recognition, but faced **high latency and slow inference.**

### **Phase 2: React + SimplePeer + FastAPI (Improved Inference Speed)**
- Transitioned to **React + SimplePeer + FastAPI**, achieving **better real-time performance.**

### **Final Phase: React + SimplePeer + TorchServe (Optimized Solution)**
- Implemented **TorchServe** as the backend, enabling **low-latency, high-throughput, and batch inference**.
- Achieved **auto-scaling** for multiple concurrent video conferences.

## Future Enhancements 🌟
- Fine-tuning models for higher emotion detection accuracy.
- Integration with **cloud AI services (AWS/GCP/Azure)** for enterprise-scale deployments.
- Expanding support for **more languages & cultures** in emotion recognition.

## Contributors 🤝
- **Vishwa J** – ML Engineer
- **Aravind G** - Full Stack Developer
- **Sivakumar A V** - ML Engineer

## License 📜
This project is licensed under the **MIT License**.

---
### 🚀 Feel free to contribute, fork, and improve the project!