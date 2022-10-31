import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


def load_data(data_path="data", batch_size=4):
    transform = transforms.Compose(
        # 这里只对其中的一个通道进行归一化的操作
        [transforms.ToTensor(),
         transforms.Normalize((0.5, ), (0.5,))])
    # 加载数据据
    trainset = torchvision.datasets.MNIST(root=data_path, train=True, download=True,
                                            transform=transform)
    testset = torchvision.datasets.MNIST(root=data_path, train=False, download=True,
                                         transform=transform)
    # 构建数据加载器
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    return trainloader, testloader


# 构造网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1 = nn.Linear(22*22*20, 500)
        self.fc2 = nn.Linear(500, 10)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 22*22*20)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train(epochs=5, save_path='models/mnist_net41.pth', batch_size=4, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    print("程序执行设备：{}".format(device))
    trainloader, testloader = load_data(batch_size=batch_size)
    net = Net()
    print(net)
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # 训练模型
    for epoch in range(epochs):
        loss_record = []
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 1000 == 999:
                print("Train Epoch:{} [{}/60000 ({}%)]    Loss:{}".format(epoch, (i+1)*batch_size, (i+1)*batch_size*100/60000, running_loss/1000))
                loss_record.append(running_loss/1000)
                running_loss = 0.0
        # 测试模型
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        per = 100 * correct / total
        epoch_loss = np.mean(np.array(loss_record))
        print()
        print("Test set: Average loss:{} Accuracy:{}/{} ({}%) ".format(epoch_loss, correct, total, per))
        print()

    # 保存模型
    torch.save(net.state_dict(), save_path)
    print('Finished Training And model saved in {}'.format(save_path))


if __name__ == '__main__':
    # load_data()
    # train(device=torch.device("cpu"))
    train(batch_size=16)
