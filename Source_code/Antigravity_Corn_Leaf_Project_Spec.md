
# Corn Leaf Disease Classification System

## Project Overview
Title: Corn Leaf Disease Classification Using Deep Learning

Goal:
Build a web application that classifies corn leaf diseases from an uploaded image.

Input:
- Corn leaf image

Output:
- Predicted disease
- Confidence score
- Probability distribution for all classes

## Final Model
Model: Vision Transformer (ViT)

## Dataset
Total Images: 9847

Classes:
- blight: 1834
- common_rust: 1802
- gray_spot: 2119
- healthy: 1677
- not_corn_leaf: 2415

Split:
- Train: 70%
- Validation: 15%
- Test: 15%

Image Size:
224 x 224 x 3

Normalization:
image = image / 255.0

## Classes

blight
common_rust
gray_spot
healthy
not_corn_leaf

## Performance

Test Accuracy: 94.19%
Best Validation Accuracy: 95.59%

## Model File

ViT_Scratch_best.keras

## Website Requirements

- Upload image
- Preview image
- Predict disease
- Show confidence
- Show probabilities for all classes
- Show disease descriptions

## UI

Theme:
- Modern AI Dashboard
- Agriculture Theme

Colors:
- Green
- White
- Dark Gray

Tech Stack:
- React
- TailwindCSS
- FastAPI
- TensorFlow / Keras


