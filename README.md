# 🚀 AI Face Attribute Analyzer & CNN Benchmark

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Latest-orange)
![Gradio](https://img.shields.io/badge/Gradio-Web%20UI-success)

## 📌 Overview
This project implements an end-to-end **Multi-Label Image Classification** pipeline to detect multiple facial attributes simultaneously from a single image. It was designed to simulate real-world Computer Vision applications, such as demographic extraction for E-KYC (Electronic Know Your Customer) and automated image tagging.

Instead of building separate models for each feature, this project utilizes a single Convolutional Neural Network (CNN) to output joint probabilities for multiple targets: **Gender (Male/Female), Age (Young), Expression (Smiling), and Accessories (Eyeglasses)**.

## ✨ Key Features
* **Multi-Architecture Benchmarking:** Trains and evaluates 5 state-of-the-art CNNs:
  * ResNet50
  * VGG16
  * GoogLeNet (Inception v1)
  * AlexNet
  * ResNeXt50
* **Advanced Training Techniques:** Implements `OneCycleLR` scheduling and End-to-End Transfer Learning (Fine-tuning) using `BCEWithLogitsLoss` for stable multi-label convergence.
* **Comprehensive Evaluation:** Generates Learning Curves, gender-specific Confusion Matrices, and detailed classification reports (Precision, Recall, F1-Score per attribute).
* **Interactive Deployment:** Includes a built-in Web UI using **Gradio** for real-time inference on uploaded images, featuring dynamic human-readable labels (e.g., dynamically converting raw probability into "Gender: Female").
* **Production-Ready Export:** Saves the best-performing model in both native PyTorch (`.pth`) and ONNX (`.onnx`) formats for cross-platform deployment.

## 🛠️ Tech Stack
* **Deep Learning Framework:** PyTorch, Torchvision
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Evaluation:** Scikit-Learn
* **Deployment UI:** Gradio

## 📂 Dataset Structure
The model is trained on a subset of celebrity facial images (e.g., CelebA format). To run this code, ensure your dataset is structured as follows:

```text
Dataset/
├── Images/                 # Folder containing 5000+ .jpg files
└── list_attribute.txt      # Text file containing image IDs and binary attribute labels
