import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
import os
import json
import matplotlib.pyplot as plt
from model import ResNet
import argparse
import mlflow

current_dir = os.path.dirname(__file__)
data_dir = os.path.join(current_dir, 'data')

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, required=True)
parser.add_argument('--params', type=str, required=True)

args = parser.parse_args()

mode = args.mode
param = args.params

if mode == "best": 
    # load best parameter
    params_path = os.path.join(current_dir, '..', 'best_params.json')
else:
    params_path = os.path.join(current_dir, '..', 'params', param)

with open(params_path, 'r') as f:
    best_params = json.load(f)

lr = best_params['lr']
batch_size = best_params['batch_size']



# data preprocessing
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    # 50% probability to flip the image horizontally
    transforms.RandomRotation(10),
    # Randomly rotate the image within ±10 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    # Apply random color distortion (brightness, contrast, saturation, hue)
    transforms.RandomCrop(32, padding=2),
    # Pad the image by 2 pixels on each side and randomly crop back to 32x32 to simulate off-center images
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5),
    # 50% probability to randomly erase a rectangle area in the image
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

root = os.path.join(current_dir, 'data')
# load data
trainset = torchvision.datasets.CIFAR10(
    root=root,
    train=True,
    download=True,
    transform=transform_train
)

train_size = int(0.8 * len(trainset))
val_size = len(trainset) - train_size
train_set, val_set = random_split(trainset, [train_size, val_size])

trainloader = DataLoader(train_set, batch_size=256, shuffle=True)
valloader = DataLoader(val_set, batch_size=256, shuffle=True)

testset = torchvision.datasets.CIFAR10(
    root=root,
    train=False,
    download=True,
    transform=transform_test
)
testloader = DataLoader(testset, batch_size=256, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet().to(device)
save_path = os.path.join(current_dir, '..', 'CIFAR_ResNet.pth')
if os.path.exists(save_path):
    model.load_state_dict(torch.load(save_path))
    print("find exist model")
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=2, 
    verbose=True)

accuracy_path = os.path.join(current_dir, '..', 'best_accuracy.pkl')
if os.path.exists(accuracy_path):
    with open(accuracy_path, 'rb') as f:
        best_accuracy = pickle.load(f)
    print(f"Loaded previous best_accuracy: {best_accuracy:.2f}%")
else:
    best_accuracy = 0
    print("No previous best_accuracy found, starting from 0%.")

epochs = 3
loss_list = []
val_accuracy_list = []
patience = 10
epochs_no_improve = 0
early_stop = False
criterion = nn.CrossEntropyLoss()
history = {'loss': [], 'val_acc': []}

mlflow.start_run()
mlflow.log_param('lr', lr)
mlflow.log_param('batch_size', batch_size)
mlflow.log_param('mode', mode)
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in trainloader:
        optimizer.zero_grad()
        images, labels = images.to(device), labels.to(device)
        # send model to gpu
        output = model(images)
        loss = criterion(output, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(output.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    accuracy = correct/total * 100
    print("train accuracy:",accuracy)
    loss_list.append(total_loss/len(trainloader))
    val_correct = 0
    val_total = 0
    model.eval()
    for images, labels in valloader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        _, predicted = torch.max(output.data, 1)
        val_correct += (predicted == labels).sum().item()
        val_total += labels.size(0)
    val_accuracy = val_correct/val_total * 100
    scheduler.step(val_accuracy)
    print("val accuracy:",val_accuracy)
    val_accuracy_list.append(val_accuracy)

    mlflow.log_metric('train_accuracy', accuracy, step=epoch)
    mlflow.log_metric('val_accuracy', val_accuracy, step=epoch)
    mlflow.log_metric('train_loss', total_loss / len(trainloader), step=epoch)

    history['loss'].append(total_loss / len(trainloader))
    history['val_acc'].append(val_accuracy)

    history_path = os.path.join(current_dir, '..', 'training_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)

    if best_accuracy < val_accuracy:
        best_accuracy = val_accuracy
        torch.save(model.state_dict(), save_path)
        print(f"New best model saved with val_accuracy: {best_accuracy:.2f}%")
        epochs_no_improve = 0

        with open(accuracy_path, 'wb') as f:
            pickle.dump(best_accuracy, f)
    else:
        epochs_no_improve+=1
        if epochs_no_improve >= patience:
            early_stop = True
    if early_stop:
        print("early stop triggered")
        break

plt.figure()
plt.plot(range(len(loss_list)), loss_list, label='Training Loss', color='red')
plt.plot(range(len(val_accuracy_list)), val_accuracy_list, label='Validation Accuracy', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss / Accuracy')
plt.title('Training Loss and Validation Accuracy')
plt.legend()
plt.grid(True)
save_fig_path = os.path.join(current_dir, '..', 'training_curve.png')
plt.savefig(save_fig_path)

mlflow.log_artifact(save_fig_path)
mlflow.log_artifact(save_path)
mlflow.log_artifact(history_path)

mlflow.end_run()

print("training finish")