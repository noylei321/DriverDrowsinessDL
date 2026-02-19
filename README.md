# 🚗 Driver Drowsiness Detection using Deep Learning

A robust Computer Vision project utilizing a Custom Convolutional Neural Network (CNN) to detect driver drowsiness from images. Built with PyTorch, this model aims to enhance road safety by accurately classifying a driver's state as either 'Drowsy' or 'Non-Drowsy'.

## 📌 Project Overview
Driver fatigue is a leading cause of traffic accidents. This project tackles the problem by analyzing facial features (specifically eye closure) using Deep Learning. We developed and trained a custom CNN architecture from scratch and compared it against a baseline model to demonstrate the effectiveness of modern regularization techniques.

## 📊 Dataset
The project uses the **Driver Drowsiness Dataset (DDD)** sourced from Kaggle.
* **Classes:** 2 (`Drowsy`, `Non Drowsy`)
* **Total Images:** ~41,000 images
* **Split Strategy:** 60% Train / 20% Validation / 20% Test (with fixed random seeds for reproducibility).

## 🏗️ Architecture & Methodology
The project implements a deep learning pipeline in **PyTorch**:
* **Baseline Model:** A simple CNN used to establish foundational performance.
* **Proposed Architecture:** An enhanced Custom CNN featuring:
  * 3 Convolutional Layers with **Batch Normalization** for faster and stable convergence.
  * **MaxPooling** & **ReLU** activations.
  * **Dropout** layers to prevent overfitting.
* **Optimization:** Adam Optimizer with `StepLR` (Learning Rate Scheduler).
* **Loss Function:** Cross Entropy Loss.
* **Data Augmentation:** Rotation and color jitter applied to the training set to ensure robustness to different lighting and head positions.

## 🚀 Key Results
The proposed architecture achieved near-perfect classification on the test set:
* **Test Accuracy:** 99.9% (Only 2 misclassifications out of 8,360 test samples).
* **AUC Score:** 1.0000
* Visualizations including Loss/Accuracy curves, Confusion Matrix, and ROC Curve are available in the notebook.

## 🛠️ How to Run
1. Open the `DriverDrowsinessProj.ipynb` notebook in Google Colab.
2. Ensure the Runtime is set to **GPU** (`Tesla T4` recommended).
3. Upload your `kaggle.json` file when prompted in Part 1 to automatically download the dataset via the Kaggle API.
4. Run the cells sequentially to train and evaluate the model.

## 🔮 Future Work
* **Video-Based Analysis:** Upgrading the system to analyze sequences of frames using LSTMs or Transformers to detect the *duration* of eye closure.
* **Mobile Integration:** Optimizing and exporting the model (via TorchScript or ONNX) for real-time deployment on Android devices.

---
**Contributors:** Noy Leibovitch, Matan Dalal  
**Framework:** PyTorch
