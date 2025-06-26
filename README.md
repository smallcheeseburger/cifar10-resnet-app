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




## Setup and Usage
### Local Docker Deployment
1. Clone the repository:
```bash
git clone https://github.com/smallcheeseburger/cifar10-resnet-app.git
cd cifar10-resnet-app
```

2. Build the Docker image:
```bash
docker build -t cifar10-app .
```
3. Run the Docker container:
```bash
docker run -p 8501:8501 cifar10-app
```
4. Access the web app in your browser:
```bash
http://localhost:8501
```
## Notes
The trained model file (.pth) and CIFAR-10 dataset files are excluded from the repository due to GitHub’s file size limitations.

The CIFAR-10 dataset will be automatically downloaded by torchvision.datasets if it is not found locally. By default, the dataset will be stored in a data/ folder in the project directory.

Please ensure that you have trained the model or provided a pre-trained model file in the correct path before running the app.

The optimized hyperparameters are saved in best_params.json.