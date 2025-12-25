# Emotion-Based Music Recommendation System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![Spotify](https://img.shields.io/badge/API-Spotify-1DB954)

A real-time deep learning application that bridges Computer Vision and Music Recommendation. The system detects user emotions via webcam feed and curates dynamic Spotify playlists to match the detected mood using a custom-trained CNN.

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [System Architecture](#-system-architecture)
- [Technical Approach](#-technical-approach)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Performance](#-performance)

---

## 📖 Project Overview
This project aims to create an interactive "AI DJ" experience. Instead of manual playlist selection, the system utilizes facial expression recognition (FER) to automate the process. The pipeline consists of face detection, emotion classification (7 classes), and API integration for real-time playlist generation.

**Key Features:**
* **Real-Time Inference:** Low-latency emotion detection using OpenCV and PyTorch.
* **Custom VGG-Style CNN:** Designed from scratch for optimal trade-off between accuracy and speed.
* **Robust Preprocessing:** Implements advanced augmentation techniques (RandomErasing, Affine Transformations) to handle lighting and pose variations.
* **Spotify Integration:** Automated OAuth authentication and playlist creation via Spotipy.

---

## 🏗️ System Architecture

### 1. Data Pipeline
The dataset is processed using a robust pipeline to ensure high generalization:
* **Cleaning:** corrupted images are filtered out during loading.
* **Augmentation:** `RandomRotation`, `ColorJitter`, and `RandomErasing` are applied to the training set to prevent overfitting.
* **Normalization:** Inputs are normalized to standard mean/std deviations.

### 2. Model: `CustomDeepEmotionNet`
A specific CNN architecture inspired by VGG-Net, optimized for the 128x128 input size:
* **Feature Extractor:** 5 Convolutional Blocks (Conv2d -> BatchNorm -> ReLU -> MaxPool).
* **Parameter Reduction:** Utilizes **Global Average Pooling (GAP)** instead of fully connected layers to minimize parameter count and reduce model size.
* **Loss Function:** `LabelSmoothingCrossEntropy` is used to handle noisy labels and improve calibration.

---

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.8+
* NVIDIA GPU (Recommended for training)
* Spotify Developer Account

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/emotion-music-player.git](https://github.com/your-username/emotion-music-player.git)
cd emotion-music-player
```
### 2. Install Dependencies
```bash
pip install torch torchvision opencv-python spotipy matplotlib tqdm numpy
```
### 3. Spotify Configuration

To enable playlist generation, you must set up your Spotify credentials:

Go to the Spotify Developer Dashboard.

Create a new App.

Set Redirect URI to http://127.0.0.1:8888/callback.

Update the config section in the script:

```bash
CLIENT_ID = 'YOUR_CLIENT_ID'
CLIENT_SECRET = 'YOUR_CLIENT_SECRET'
```

### 🚀 Usage

Inference (Run the App)

To launch the real-time detector:
```bash
python main.py
```

## 🎮 Controls

| Key | Function |
|-----|----------|
| **P** | Generate Playlist: Captures current mood & opens Spotify |
| **Q** | Quit: Closes the application |

---

### 📊 Performance

The model was trained for 45 epochs using AdamW optimizer and ReduceLROnPlateau scheduler.

| Metric | Value |
|-----|----------|
| **Training Accuracy** | 92.91% |
| **Validation Accuracy** | 84.99% |
| **Inference Speed** | ~27 FPS (on GPU) |

### 👤 Creators

Moaz Hany 

| Role | Name | Contribution |
|-----|----------|-------------|
| **Team Lead & Full Stack AI** | **Moaz Hany** | Orchestrated system integration, developed Flask backend logic, implemented Spotify API bridge, and engineered the UI/UX. |
| **Data Engineer** | **Mohamed Tamer** | Designed the data pipeline, handled dataset curation, cleaning, preprocessing, and implemented data augmentation strategies. |
| **AI Architect** | **Asmahan Saad** | Designed the custom Deep CNN architecture, optimizing layer depth and feature extraction for high-accuracy classification. |
| **Machine Learning Engineer** | **Hager Moustafa** | Developed the training loop, managed hyperparameter tuning, loss function optimization, and model validation. |
| **Computer Vision Engineer** | **Nour Mohamed** | Built the real-time inference engine using OpenCV, optimizing video processing for low-latency performance. |

### 🔮 Future Improvements

**Dockerization**: Containerizing the application for easier deployment.

**Cloud Deployment**: Hosting the inference model on AWS/GCP.

**User Preferences**: Adding functionality to learn user music taste over time.