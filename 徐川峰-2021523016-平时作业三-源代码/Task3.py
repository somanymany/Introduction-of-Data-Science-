import torchvision


def load_data(data_path="data", batch_size=4):
    # transform = transforms.Compose(
    #     # 这里只对其中的一个通道进行归一化的操作
    #     [transforms.ToTensor(),
    #      transforms.Normalize((0.5, ), (0.5,))])
    # 加载数据
    trainset = torchvision.datasets.MNIST(root=data_path, train=True, download=True,
                                            transform=None)
    testset = torchvision.datasets.MNIST(root=data_path, train=False, download=True,
                                         transform=None)
    print("训练集大小：{}".format(len(trainset)))
    print("测试集大小：{}".format(len(testset)))

    for img, label in trainset:
        print("图片数组的size是{}".format(img.size))
        img.show()
        break


if __name__ == '__main__':
    load_data()