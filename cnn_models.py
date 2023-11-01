import torch
from torch import nn

class BaseCNN(nn.Module):
    def __init__(self):
        super(BaseCNN, self).__init__()
        self.keep_prob = 0.5
        # (3, 224, 224)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # (32, 112, 112)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # (32, 56, 56)
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # (64, 28, 28)
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # (128, 14, 14)
        self.fc1 = nn.Linear(128*14*14, 512, bias=True)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(512, 512, bias=True)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.layer5 = nn.Sequential(
            self.fc1,
            nn.Dropout(p=self.keep_prob),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            self.fc2,
            nn.Dropout(p=self.keep_prob),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        
        self.fc3 = nn.Linear(512, 8, bias=True)
        
        nn.init.xavier_uniform_(self.fc3.weight)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.layer5(out)
        out = self.fc3(out)
        return out

class CNN_2(nn.Module):
    def __init__(self):
        super(CNN_2, self).__init__()
        self.keep_prob = 0.5
        # (3, 224, 224)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # (32, 112, 112)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # (32, 56, 56)
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # (64, 28, 28)
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # (128, 14, 14)
        self.fc1 = nn.Linear(128*14*14, 512, bias=True)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.layer5 = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(p=self.keep_prob)
        )
        self.fc2 = nn.Linear(512, 8, bias=True)
        nn.init.xavier_uniform_(self.fc2.weight)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.layer5(out)
        out = self.fc2(out)
        return out
    
class CNN_3(nn.Module):
    def __init__(self):
        super(CNN_3, self).__init__()
        self.keep_prob = 0.5
        # (3, 224, 224)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # (32, 112, 112)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # (32, 56, 56)
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # (64, 28, 28)
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # (128, 14, 14)
        self.fc1 = nn.Linear(128*14*14, 512, bias=True)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(512, 512, bias=True)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.layer5 = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(p=self.keep_prob),
            self.fc2,
            nn.ReLU(),
            nn.Dropout(p=self.keep_prob)
        )
        
        self.fc3 = nn.Linear(512, 8, bias=True)
        
        nn.init.xavier_uniform_(self.fc3.weight)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.layer5(out)
        out = self.fc3(out)
        return out
    
# Pytorch SENet implementation
class SEBlock(nn.Module):
    def __init__(self, in_channels, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // r, in_channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        se = self.squeeze(x).view(b,c)
        se = self.excitation(se).view(b, c, 1, 1)
        return x * se.expand_as(x)
    
class SEBasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SEBlock(planes, reduction)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        return out

class SE_CNN(nn.Module):
    def __init__(self):
        super(SE_CNN, self).__init__()
        self.keep_prob = 0.5
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            SEBasicBlock(32, 32),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        )
        
        self.layer2 = nn.Sequential(
            SEBasicBlock(32, 32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        )
        
        self.layer3 = nn.Sequential(
            SEBasicBlock(64, 64),
            SEBasicBlock(64, 64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        )
        
        self.layer4 = nn.Sequential(
            SEBasicBlock(128, 128),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc1 = nn.Linear(128, 512, bias=True)
        nn.init.xavier_uniform_(self.fc1.weight)
        
        self.fc2 = nn.Linear(512, 8, bias=True)
        nn.init.xavier_uniform_(self.fc2.weight)
        
        self.layer5 = nn.Sequential(
            self.fc1,
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            self.fc2
        )
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.flatten(out, 1)
        out = self.layer5(out)
        return out
    
# Modified ResNet from "Expert-level prenatal detection of complex congenital heart disease from screening ultrasound using deep learning"
class M_resnet(nn.Module):
    class BasicBlock(nn.Module):
        def __init__(self, in_, out_, stride=1):
            super().__init__()

            self.residual_function = nn.Sequential(
                nn.BatchNorm2d(in_),
                nn.ReLU(),
                nn.Conv2d(in_, out_, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_)
            )

            self.shorcut = nn.Sequential()

            self.relu = nn.ReLU()

        def forward(self,x):
            x = self.residual_function(x) + self.shorcut(x)
            x = self.relu(x)
            return x
    def __init__(self):
        super(M_resnet, self).__init__()
        size2 = 2
        num_classes = 8
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        )
        
        self.layer2 = self.BasicBlock(64, 64)
        
        self.layer3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        
        self.layer4 = self.BasicBlock(128, 128)
        
        self.layer5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        
        self.layer6 = self.BasicBlock(256, 256)
        
        self.GAP = nn.AdaptiveAvgPool2d(1)
        
        self.layer7 = nn.Sequential(
            nn.Linear(256, 512, bias=True),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes, bias=True)
        )
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.layer5(x)
        x = self.layer6(x)
        
        x = self.GAP(x)
        x = torch.flatten(x, 1)
        x = self.layer7(x)
        return x
        
# Modified VGG16 from "Fast and accurate view classification of echocardiograms using deep learning"
class M_vgg16(nn.Module):
    def __init__(self):
        super().__init__()
        num_classes = 8
        # (3, 224, 224)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # (32, 112, 112)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # (64, 56, 56)
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # (128, 28, 28)
        self.layer4 = nn.Sequential(
            nn.Linear(128*28*28, 1028, bias=True),
            nn.Dropout(0.3),
            nn.BatchNorm1d(1028),
            nn.ReLU(),
            nn.Linear(1028, 512, bias=True),
            nn.Dropout(0.3),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.classifier = nn.Linear(512, num_classes, bias=True)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.layer4(x)
        x = self.classifier(x)
        return x
        