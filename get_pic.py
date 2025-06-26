import os
import random
import torchvision
from torchvision import transforms
from PIL import Image

def generate_random_cifar10_image(save_path='./project/random_image.png'):
    # load CIFAR dataset
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

    # randomly choose a pic
    index = random.randint(0, len(testset) - 1)
    image, label = testset[index]

    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"Removed existing file: {save_path}")

    # save pic
    image.save(save_path)
    print(f"Saved random CIFAR-10 image as '{save_path}' with label: {label}")
    return save_path, label

generate_random_cifar10_image()