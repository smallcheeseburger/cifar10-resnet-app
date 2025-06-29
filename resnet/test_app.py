import unittest
import torch
from model import ResNet
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn

class TestModel(unittest.TestCase):
    def test_output_shape(self):
        model = ResNet()
        x = torch.randn(1, 3, 32, 32)
        y = model(x)
        self.assertEqual(y.shape, (1, 10))

    def test_dataloader(self):
        transform = transforms.ToTensor()
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        image, _ = trainset[0]
        self.assertTrue(torch.is_tensor(image))
        self.assertEqual(image.shape, (3, 32, 32))

    def test_train(self):
        model = ResNet()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        x = torch.randn(4, 3, 32, 32)
        y = torch.randint(0, 10, (4,))
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        self.assertGreater(loss.item(), 0)

    def test_invalid_input_type(self):
        model = ResNet()
        with self.assertRaises(Exception):
            model("invalid input")

    def test_empty_input(self):
        model = ResNet()
        x = torch.empty(0)
        with self.assertRaises(Exception):
            model(x)

if __name__ == '__main__':
    unittest.main()