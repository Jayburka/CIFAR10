
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.data

#定义一个简单的神经网络

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        #定义第一层卷积神经网络，输入通道维度=1，输出通道维度=6，卷积核大小为3*3
        self.conv1 = nn.Conv2d(3, 6, 5)
        #定义第二层卷积神经网络，输入通道维度=6，输出通道维度=16，卷积核大小为3*3
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 定义三层全连接层网络
        
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        # 在（2,2）的池化窗口下执行最大池化操作
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2), 2)
        
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        # 计算size, 除了第0个维度上的batch_size
        size = x.size()[1:]
        print('size大小为',size)
        num_features = 1
        for s in size:
            num_features *=s
            
        return num_features
    
net = Net()

import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt 

import numpy as np

# 构建展示图片的函数

def imshow(img):
    img = img / 2 + 0.5 
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 从数据迭代器中读取一张图片

dataiter = iter(trainloader)

images, labels = next(dataiter)

#展示图片

imshow(torchvision.utils.make_grid(images))

#打印logo

print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

