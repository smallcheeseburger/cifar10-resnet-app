# CIFAR-10 ResNet Image Classification Web App

This project is a deep learning web application that classifies images from the CIFAR-10 dataset using a custom-built ResNet model. The application is built with PyTorch, Streamlit, and Docker.

## Project Overview

The web app allows users to upload CIFAR-10 images and receive instant classification results. The model is trained locally and the system provides a training dashboard including validation accuracy and training loss visualization.

## Features

- ResNet-based custom convolutional neural network
- Streamlit-powered web interface
- Real-time image upload and prediction
- Training progress visualization
- Dockerized deployment for portability

## Technologies Used

- Python
- PyTorch
- Streamlit
- Docker
- Optuna (for hyperparameter tuning)
- Matplotlib (for training curve plotting)

## Project Structure

```text
├── resnet/              # Model and training scripts
│   ├── model.py
│   ├── train.py
│   └── tuner.py
├── app.py               # Streamlit web app
├── get_pic.py           # Image downloader (optional)
├── Dockerfile           # Docker configuration
├── requirements.txt     # Python dependencies
├── best_params.json     # Optimized hyperparameters
├── .gitignore           # Files excluded from version control
└── README.md            # Project documentation
```