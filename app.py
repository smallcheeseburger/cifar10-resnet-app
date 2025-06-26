import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import os
import json
from resnet.model import ResNet

# CIFAR-10 label list
class_labels = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Device detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load best hyperparameters
with open('best_params.json', 'r') as f:
    best_params = json.load(f)

# Load model
model_path = './CIFAR_ResNet.pth'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

model = ResNet()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Streamlit page title
st.title("CIFAR-10 Image Classifier with Training Dashboard")

# Sidebar: hyperparameter info
st.sidebar.header("Training Configuration")
st.sidebar.write(f"Epochs: 100")
st.sidebar.write(f"Batch Size: {best_params['batch_size']}")
st.sidebar.write(f"Learning Rate: {best_params['lr']}")
st.sidebar.write(f"Patience: 10")
st.sidebar.write(f"Loaded Model: {model_path}")

# Show training curve
st.header("Training Progress")
if os.path.exists('training_curve.png'):
    st.image('training_curve.png', caption="Training Loss and Validation Accuracy")
else:
    st.warning("Training curve not found. Please train the model first.")

# Image upload and prediction
st.header("Image Upload and Prediction")
uploaded_file = st.file_uploader("Upload a CIFAR-10 image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        input_tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output.data, 1)
            predicted_class = class_labels[predicted.item()]

        st.success(f"Predicted Class: {predicted_class}")

