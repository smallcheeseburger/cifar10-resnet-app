import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels,stride):
        # in_channels represents the depth of the input image(3 for RGB images)
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride = stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels or stride !=1:
        # input and output channel must equal in able apply residual algorithm
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            # when in_channnels equal out_channels and stride is 1, no projection needed

    def forward(self, x):
        identity = self.shortcut(x)  
        # Apply shortcut to x to adjust dimensions if needed
        # The output 'out' and 'identity' must have the same shape to enable residual addition
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        # Residual connection
        return self.relu(out)
    
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.layer1 = self.make_layer(64,64,2,1)
        # Feature map size: 32x32
        self.layer2 = self.make_layer(64,128,2,2)
        # Feature map size: 16*16
        self.layer3 = self.make_layer(128,256,2,2)
        # Feature map size: 8*8
        self.layer4 = self.make_layer(256,512,2,2)
        # Feature map size: 4*4
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # Downsample the feature map to size 1x1
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512,10)
        self.dropout = nn.Dropout(p=0.4)
        # Dropout layer to reduce overfitting

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        # [batch_size, 512, 1, 1]
        x = self.flatten(x)
        # flatten to[batch_size, 512]
        x = self.dropout(x)
        x = self.fc1(x)
        return x
    
    def make_layer(self, in_channels, out_channels_, num_blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels,out_channels_,stride = stride))

        for i in range(num_blocks):
            layers.append(BasicBlock(out_channels_,out_channels_,stride=1))
        return nn.Sequential(*layers)
        # Return the layer sequence as a single module