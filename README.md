# SignWeaver: AI-Powered Sign Language Recognition & Translation System

SignWeaver is an AI-powered, real-time sign language recognition and translation system designed to bridge the communication gap between individuals who use sign language and those who do not [cite: 476].

---

## 📖 Overview
SignWeaver processes continuous video streams or individual images to recognize sign language gestures and convert them into text, with an optional text-to-speech (TTS) feature [cite: 477, 503]. It is built to operate at interactive frame rates (target latency < 200 ms), making it suitable for live scenarios like medical consultations, customer service, and educational settings [cite: 478].

## ✨ Key Features
* **Real-Time Processing:** Gesture recognition from webcam inputs with low-latency performance [cite: 490].
* **Flexible Output:** Real-time text display and optional text-to-speech (TTS) for verbal communication [cite: 491, 492].
* **Dynamic & Static Support:** Recognizes both single-frame (static) and multi-frame (dynamic) gestures [cite: 493].
* **Scalable Architecture:** Modular microservice design using FastAPI [cite: 494].
* **Deployment Ready:** Containerized for cloud deployment with support for auto-scaling [cite: 497].

## 🛠 Tech Stack
| Component | Technology |
| :--- | :--- |
| **Language** | Python 3.11 [cite: 668] |
| **Computer Vision** | OpenCV, MediaPipe [cite: 668] |
| **Deep Learning** | PyTorch (primary), TensorFlow/Keras (alternate) [cite: 668] |
| **Backend** | FastAPI, Uvicorn [cite: 668] |
| **Frontend** | React.js, WebRTC, Web Speech API [cite: 668] |
| **Deployment** | Docker, Kubernetes (K8s), AWS (EC2/ECS/S3) [cite: 668] |
| **Monitoring** | Prometheus, Grafana [cite: 668] |

## 🚀 Pipeline Workflow
1.  **Input:** Captures live video via MediaDevices API or accepts uploaded video files [cite: 499, 587].
2.  **Processing:** Extracts frames, detects landmarks via MediaPipe, and constructs feature vectors [cite: 501].
3.  **Inference:** A hybrid CNN+LSTM model processes sequences to predict gesture classes [cite: 598].
4.  **Output:** Serializes predictions into JSON, displays text on the UI overlay, and provides TTS audio playback [cite: 608, 609].

## 📊 Performance
The system is optimized for speed. Under LAN conditions with CPU-based inference, total end-to-end latency is approximately **55–90 ms** [cite: 612]. The hybrid CNN+LSTM architecture achieves **91.7% accuracy** on the WLASL-100 benchmark [cite: 680].

## ⚖️ Ethical Considerations
* **Privacy:** Biometric landmark data is processed server-side and discarded upon disconnection; raw video is not stored [cite: 682, 683].
* **Transparency:** Users are informed about video processing, and data is only used for training with explicit opt-in consent [cite: 684, 685].
* **Safety:** The system is intended as an assistive tool to supplement—not replace—human interpreters in high-stakes environments [cite: 690].

---

*Project developed by Abhinav Gangwar.*