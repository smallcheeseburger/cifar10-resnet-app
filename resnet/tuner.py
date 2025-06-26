import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import optuna
from model import ResNet
import json

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomCrop(32, padding=2),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    train_set, val_set = random_split(trainset, [train_size, val_size])

    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=False)

    criterion = nn.CrossEntropyLoss()

    patience = 5
    epochs_no_improve = 0
    best_val_accuracy = 0

    for epoch in range(5):  # train 5 epoch in each trial
        model.train()
        for images, labels in trainloader:
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

        val_correct = 0
        val_total = 0
        model.eval()
        with torch.no_grad():
            for images, labels in valloader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                _, predicted = torch.max(output.data, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_accuracy = val_correct / val_total * 100
        scheduler.step(val_accuracy)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    return 1 - (best_val_accuracy / 100)  # goal is to minimize validation error


pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
# first 5 trials will not be pruned, each trial will train at least 2 epochs
study = optuna.create_study(direction='minimize', pruner=pruner)
study.optimize(objective, n_trials=10)

print('Best params:', study.best_params)
print('Best validation accuracy:', (1 - study.best_value))

with open('best_params.json', 'w') as f:
    json.dump(study.best_params, f)