import torch
import torchvision

import matplotlib.pyplot as plt

from torch.autograd import Variable
from tqdm import tqdm


class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2, kernel_size=2))
        self.dense = torch.nn.Sequential(torch.nn.Linear(14 * 14 * 128, 1024),
                                         torch.nn.ReLU(),
                                         torch.nn.Dropout(p=0.5),
                                         torch.nn.Linear(1024, 10))

    def forward(self, x):
        x = self.conv1(x)
        # x = self.conv2(x)
        x = x.view(-1, 14 * 14 * 128)
        x = self.dense(x)
        return x


def mnist_train():
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(mean=[0.5, ],
                                                                                 std=[0.5, ])])

    data_train = torchvision.datasets.MNIST(root="./data/",
                                            transform=transform,
                                            train=True,
                                            download=True)

    data_test = torchvision.datasets.MNIST(root="./data/",
                                           transform=transform,
                                           train=False)

    data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                    batch_size=64,
                                                    shuffle=True,
                                                    num_workers=2)

    data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                                   batch_size=64,
                                                   shuffle=True,
                                                   num_workers=2)

    # print(data_loader_train)

    # print(next(iter(data_loader_train)))

    images, labels = next(iter(data_loader_train))
    img = torchvision.utils.make_grid(images)

    img = img.numpy().transpose(1, 2, 0)
    std = [0.5, 0.5, 0.5]
    mean = [0.5, 0.5, 0.5]
    img = img * std + mean
    model = Model()

    cost = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    n_epochs = 5

    for epoch in range(n_epochs):
        running_loss = 0.0
        running_correct = 0
        print("Epoch {}/{}".format(epoch, n_epochs))
        print("-" * 10)
        for data in tqdm(data_loader_train):
            x_train, y_train = data
            x_train, y_train = Variable(x_train), Variable(y_train)
            outputs = model(x_train)
            _, pred = torch.max(outputs.data, 1)
            optimizer.zero_grad()
            loss = cost(outputs, y_train)

            loss.backward()
            optimizer.step()

            running_loss += loss.data.item()
            running_correct += torch.sum(pred == y_train.data)
        testing_correct = 0
        for data in data_loader_test:
            x_test, y_test = data
            x_test, y_test = Variable(x_test), Variable(y_test)
            outputs = model(x_test)
            _, pred = torch.max(outputs.data, 1)
            testing_correct += torch.sum(pred == y_test.data)
        print(
            "Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Accuracy is:{:.4f}".format(running_loss / len(data_train),
                                                                                        100 * running_correct / len(
                                                                                            data_train),
                                                                                        100 * testing_correct / len(
                                                                                            data_test)))
    torch.save(model.state_dict(), "model_parameter.pkl")


if __name__ == "__main__":
    mnist_train()
