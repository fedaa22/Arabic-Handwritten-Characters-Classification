# Arabic Handwritten Characters Classification

## Project Overview
This project classifies 28 Arabic handwritten characters using deep learning.

## Dataset
- **Source**: https://github.com/HossamBalaha/HMBD-v1 
- **Total Images**: 13,075
- **Classes**: 28 Arabic letters (isolated form)
- **Image Size**: 64×64 pixels (grayscale)

## Models

### 1. Custom CNN (Built from Scratch)
- 3 convolutional blocks with BatchNormalization and Dropout
- GlobalAveragePooling2D to reduce parameters
- L2 regularization to prevent overfitting
- Test Accuracy: 34%  
### 2. Transfer Learning (MobileNetV2)
- Pretrained on ImageNet (1.4M images)
- Custom classification head added manually
- Two-phase training: frozen base → fine-tuning
- Test Accuracy: **84.45%**

## Results Comparison

| Metric | Custom CNN | Transfer Learning |
|--------|------------|-------------------|
| Test Accuracy | 34.96% | 84.45% |
| Validation Accuracy | 89.63% | 84.29% |
| Training Time |  90 min |  70 min |

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
