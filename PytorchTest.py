import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

#定义一个简单的神经网络
import multiprocessing


# class Net(nn.Module):
    
#     def __init__(self):
#         super(Net, self).__init__()
#         #定义第一层卷积神经网络，输入通道维度=1，输出通道维度=6，卷积核大小为3*3
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         #定义第二层卷积神经网络，输入通道维度=6，输出通道维度=16，卷积核大小为3*3
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         # 定义三层全连接层网络
        
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
        
#     def forward(self, x):
#         # 在（2,2）的池化窗口下执行最大池化操作
#         x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
#         x = F.max_pool2d(F.relu(self.conv2), 2)
        
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
    
#     def num_flat_features(self, x):
#         # 计算size, 除了第0个维度上的batch_size
#         size = x.size()[1:]
#         print('size大小为',size)
#         num_features = 1
#         for s in size:
#             num_features *=s
            
#         return num_features
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()



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

# import matplotlib.pyplot as plt 

# import numpy as np

# # 构建展示图片的函数

def imshow(img):
    img = img / 2 + 0.5 
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# # 从数据迭代器中读取一张图片

# dataiter = iter(trainloader)

# images, labels = next(dataiter)

# #展示图片

# imshow(torchvision.utils.make_grid(images))

# #打印logo

# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

#采用交叉熵损失函数 和 随机梯度下降优化器 
#首先要定义设备, 如果CUDA是可用的则被定义成GPU, 否则被定义成CPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net.to(device)
#采用交叉熵损失函数
criterion = nn.CrossEntropyLoss()

#随机梯度下降优化器
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(10):
    # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        #enumerate enumerate 是一个内置函数，用于将一个可迭代对象（如列表、元组、字符串等）转换为一个枚举对象
        
        #data中包含输入图片张量inputs, 标签张量labels
        # inputs, labels = data
        
        # 将输入的图片张量和标签张量转移到GPU上
        inputs, labels = data[0].to(device), data[1].to(device)
        # 首先将优化器梯度归零
        optimizer.zero_grad()
        
        #输入图片张量进网络，得到输出张量outputs
        outputs = net(inputs)
        
        # 利用网络的输出outputs 和标签labelsl计算损失值
        #(计算目标值和输出值损失函数)
        loss = criterion(outputs, labels)
        
        # 反向传播 + 参数更新， 是标准的代码流程
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()
        if (i + 1) % 2000 == 0:
            print('[%d, %5d] loss : %.3f' %
                (epoch + 1 , i + 1, running_loss / 2000))
                #这行代码的作用是将epoch + 1、i + 1和running_loss / 2000这三个变量的值格式化后插入到字符串[%d, %5d] loss: %.3f中，并打印输出到控制台。
                # %(epoch + 1 , i + 1, running_loss / 2000) 这个前面的%  是用来进行字符串格式化的操作符，被称为格式化操作符。在这里，% 将一个格式化字符串和一个元组结合起来
            running_loss = 0.0
            
print('Finished Training')

# 首先设定模型的保存路径
PATH = './cifar_net.pth'
# 保存模型的状态字典
torch.save(net.state_dict, PATH)


dataiter = iter(testloader)

# 迭代器改版调用方法后 为next(xx)
images, labels = next(dataiter)


# 打印原始图片

imshow(torchvision.utils.make_grid(images))

print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))