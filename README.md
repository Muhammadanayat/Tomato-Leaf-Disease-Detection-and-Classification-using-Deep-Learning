# Tomato Leaf Disease Detection and Classification

This repository contains the code and resources for the **Tomato Leaf Disease Detection and Classification using Deep Learning** project. The project leverages deep learning techniques, specifically Convolutional Neural Networks (CNNs), to identify and classify tomato leaf diseases. A Streamlit-based web application is also included for real-time user interaction with the trained model.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Modeling](#modeling)
5. [Web Application](#web-application)
6. [Installation](#installation)
9. [Results](#results)

---

## Introduction

Tomato plants are highly susceptible to various diseases, causing significant yield loss if not diagnosed early. Traditional methods of disease detection are often inaccurate and time-consuming. This project implements a deep learning-based approach to detect and classify tomato leaf diseases, offering an efficient, accurate, and scalable solution. 

Key Features:
- **CNN Models**: Custom and pre-trained MobileNetV2 models for classification.
- **Streamlit App**: A user-friendly interface to upload leaf images and receive disease diagnoses with confidence scores.

---

## Dataset

The project uses the **PlantVillage Dataset**, which includes:
- 10 categories of tomato leaf diseases and healthy leaves.
- 16,011 labeled images of tomato leaves.
- Publicly available on [Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease/data).

---

## Methodology

1. **Data Preprocessing**:
   - Images resized to 128x128 pixels.
   - Augmentation techniques: rotation, flipping, zooming, and shearing.
   - Normalization to scale pixel values between 0 and 1.

2. **Model Development**:
   - Custom CNN architecture with convolutional, pooling, and dense layers.
   - MobileNetV2 for transfer learning.

3. **Evaluation Metrics**:
   - Accuracy, precision, recall, F1-score.
   - ROC-AUC analysis and confusion matrix visualization.

4. **Deployment**:
   - Streamlit-based app for real-time predictions.

---

## Modeling

The repository includes the following models:
1. **Custom CNN**:
   - Designed for efficient disease detection.
   - Incorporates dropout layers to prevent overfitting.
   
2. **MobileNetV2**:
   - Lightweight pre-trained model fine-tuned for this project.

---

## Web Application

The Streamlit app provides the following functionalities:
- **Image Upload**: Users can upload images of tomato leaves.
- **Disease Prediction**: Displays the disease class with a confidence score.
- **Visualization**: Shows the uploaded image along with prediction results.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Muhammadanayat/Tomato-Leaf-Disease-Detection-and-Classification-using-Deep-Learning.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Tomato-Leaf-Disease-Detection-and-Classification-using-Deep-Learning
   ```
3. Install dependencies:
   ```bash
   pip install -r APP/requirements.txt
   ```

---

## Results

- Custom CNN achieved an accuracy of **96.4%**.
- MobileNetV2 achieved an accuracy of **90.2%**.
- Visualizations include confusion matrix and ROC curves.
