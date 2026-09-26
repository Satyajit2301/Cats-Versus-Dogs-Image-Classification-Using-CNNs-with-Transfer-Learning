# Cats vs Dogs - Image Classification Using CNNs with Transfer Learning

## 📋 Project Description
This project implements an image classification system that automatically classifies input images as either **Cat** or **Dog** using Convolutional Neural Networks (CNNs) with **Transfer Learning**.

## 🏗️ Architecture
- **Model:** EfficientNetB0 (Pre-trained on ImageNet)
- **Approach:** Transfer Learning with Fine-Tuning
- **Input Size:** 128 × 128 × 3 (RGB)
- **Output:** Binary Classification (Cat=0, Dog=1)

## 📊 Results
| Model | Test Accuracy | F1-Score | AUC | Parameters |
|-------|:---:|:---:|:---:|:---:|
| VGG16 | 86.43% | 0.8690 | 0.9551 | 14.8M |
| ResNet50 | 90.71% | 0.9103 | 0.9659 | 24.1M |
| MobileNetV2 | 87.86% | 0.8889 | 0.9804 | 2.6M |
| **EfficientNetB0** | **89.29%** | **0.9020** | **0.9584** | **4.4M** |
| Xception | 87.14% | 0.8714 | 0.9522 | 21.4M |

## 🛠️ Tech Stack
- Python, TensorFlow/Keras
- Streamlit (Deployment)
- Google Colab (Training)

## 📂 Dataset
[Cats and Dogs Image Classification](https://www.kaggle.com/datasets/samuelcortinhas/cats-and-dogs-image-classification) — 697 images (349 Cats, 348 Dogs)

## 👨‍💻 Author
**Satyajit** — Lab Assignment 03
