# AI Mood DJ - Emotion-Based Music Recommender

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-Backend-black)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![Spotify](https://img.shields.io/badge/API-Spotify-1DB954)

> **Turn your vibe into a playlist.** AI Mood DJ is a full-stack web application that scans your facial expressions in real-time and curates the perfect Spotify playlist to match (or fix) your mood using Deep Learning.

![Project Banner](path/to/your/snapshot_image.jpg) 
*(Place one of your generated snapshots here)*

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
- [The Team](#-the-team)

---

## Project Overview
This project evolves the concept of an "AI DJ" from a simple script into a **complete web platform**. Instead of manually searching for music, the system utilizes a custom Convolutional Neural Network (CNN) to detect 7 distinct emotions via webcam. It offers a "Therapy Mode" based on the Iso-principle of music therapy to either match the user's mood or uplift it.

---

## Key Features

### Real-Time Emotion Recognition
* Low-latency detection using **OpenCV** and **PyTorch**.
* Classifies 7 emotions: *Happy, Sad, Angry, Neutral, Surprise, Fear, Disgust*.

### Therapy Mode (Iso-Principle)
The user can choose the AI's strategy:
1.  **Match Vibe:** Feeling sad? The AI plays melancholic tracks to help you cathart.
2.  **Cheer Up:** Feeling down? The AI intelligently selects uplifting tracks to gradually boost your mood.

### Viral Social Sharing
* Generates an **Instagram-ready Story (9:16)** image.
* Includes your photo with a neon glow, detected mood, sound waves, and a **scannable QR Code** linking directly to the generated Spotify playlist.

### Secure Authentication
* Full user system with **Login/Register** capabilities.
* Secure password hashing and session management using **Flask-Login** and **SQLAlchemy**.

### Modern UI/UX
* Responsive **Neon Glassmorphism** design.
* Smooth animations and tabbed navigation.

---

## System Architecture

### 1. The AI Model (`CustomDeepEmotionNet`)
A VGG-style CNN optimized for 128x128 input size:
* **Architecture:** 5 Convolutional Blocks + Global Average Pooling (GAP).
* **Performance:** ~91% Training Accuracy / ~88% Validation Accuracy on RAF-DB.
* **Inference:** runs at ~27 FPS on GPU.

### 2. Backend & Integration
* **Flask:** Handles routing, video streaming, and API logic.
* **Spotipy:** Bridges the AI with Spotify's Web API for playlist creation.
* **Pillow (PIL):** dynamically draws the viral snapshot images.

---

## Installation & Setup

### Prerequisites
* Python 3.8+
* Spotify Developer Account

### 1. Clone the Repository
```bash
git clone [https://github.com/m-o-a-z-e/AI_DJ_Mood.git](https://github.com/m-o-a-z-e/AI_DJ_Mood.git)
cd AI_DJ_Mood
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration (.env)

Create a file named .env in the root directory and add your credentials:

```bash
SPOTIPY_CLIENT_ID=your_spotify_client_id
SPOTIPY_CLIENT_SECRET=your_spotify_client_secret
SPOTIPY_REDIRECT_URI=[http://127.0.0.1:5000/callback](http://127.0.0.1:5000/callback)
FLASK_SECRET_KEY=your_random_secret_key
```

### 4. Run the Application
```bash
python app.py
```

Visit http://127.0.0.1:5000 in your browser.

### Usage Guide

Sign Up/Login: Create an account to access the experience.

Start Experience: Click "Start Experience" to enable the camera.

Choose Strategy: Select Match Vibe or Cheer Up.

Generate: Click "Generate Playlist" once your mood is stable.

Share: Once the playlist is ready, a Camera Icon will appear. Click it to generate your viral Vibe Card with a QR code.

### Performance

The model was trained for 45 epochs using AdamW optimizer and ReduceLROnPlateau scheduler.

| Metric | Value |
|-----|----------|
| **Training Accuracy** | 90.91% |
| **Validation Accuracy** | 87.99% |
| **Inference Speed** | ~27 FPS (on GPU) |

## Future Roadmap & Business Strategy

### Advanced AI & Personalization
* **Reinforcement Learning (RLHF):** Implement a feedback loop where the model learns from user actions (Skips/Likes) to refine future recommendations.
* **Multimodal Analysis:** Combine facial expressions with **Speech Emotion Recognition (SER)** for higher confidence detection.
* **Context-Aware Engine:** Integrate Weather & Time APIs to suggest tracks based on the environment (e.g., "Rainy Lo-Fi" vs "Sunny Pop").

### Performance & Engineering
* **Edge AI Deployment:** Convert the PyTorch model to **ONNX Runtime** to run inference directly in the user's browser, reducing server latency to zero.
* **Model Quantization:** Optimize model size for mobile devices and low-power environments.

### Product Features
* **Mood Analytics Dashboard:** Visualize user's emotional trends over weeks/months.
* **Vibe Match (Multi-User):** Detect multiple faces and generate a "Consensus Playlist" that balances the mood of everyone in the room.
* **Voice Commands:** Hands-free control to change genres or restart detection.

### Business Value & Analytics
*Transforming emotional data into actionable business intelligence to drive retention and revenue.*

* **Emotional Retention Rate (ERR):**
    * A critical metric measuring the effectiveness of the **"Iso-Principle"** logic in reducing session termination compared to random shuffling.
    * **Goal:** Minimize "Mood Mismatch Churn" and maximize User Lifetime Value (LTV).

* **Dynamic Ad-Targeting Engine:**
    * Leveraging real-time emotional context to optimize Ad conversion rates:
    * **Happy State:** Serve high-energy, high-conversion ads (e.g., Travel, Events, Fashion).
    * **Sad/Anxious State:** Activate **"Churn Protection Mode"**—suppress intrusive ads to prevent app exit, or strategically serve comfort-oriented ads.

* **Artist Mood Score (AMS):**
    * A proprietary metric evaluating an artist's efficacy in *shifting* or *sustaining* a user's emotional state.
    * **B2B Value:** High-value data asset for Record Labels to optimize marketing campaigns and release strategies.


    ### Creators


| Role | Name | Contribution |
|-----|----------|-------------|
| **Team Lead & Full Stack AI** | **Moaz Hany** | Orchestrated system integration, developed Flask backend logic, implemented Spotify API bridge, and engineered the UI/UX. |
| **Data Engineer** | **Mohamed Tamer** | Designed the data pipeline, handled dataset curation, cleaning, preprocessing, and implemented data augmentation strategies. |
| **AI Architect** | **Asmahan Saad** | Designed the custom Deep CNN architecture, optimizing layer depth and feature extraction for high-accuracy classification. |
| **Machine Learning Engineer** | **Hager Moustafa** | Developed the training loop, managed hyperparameter tuning, loss function optimization, and model validation. |
| **Computer Vision Engineer** | **Nour Mohamed** | Built the real-time inference engine using OpenCV, optimizing video processing for low-latency performance. |