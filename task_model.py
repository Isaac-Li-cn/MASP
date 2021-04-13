import os
import torch
from torch import nn, optim
from tqdm import tqdm
import numpy as np

from torch.utils.data import DataLoader

from utils.data_io import coco_c
from utils.preprocess import resnet_18_encoder

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(512, 32)
        self.fc2 = nn.Linear(32, 4)

    def forward(self, x):
        x = x.view([-1, 512])
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def train():
    batch_size = 8192
    learning_rate = 0.02
    epoch_num = 15

    coco_c_train = coco_c('../coco_c/classify_folder_train/*.jpg')
    coco_c_val = coco_c('../coco_c/classify_folder_val/*.jpg')

    train_loader = DataLoader(coco_c_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(coco_c_val, batch_size=batch_size, shuffle=True)

    model = Net()

    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epoch_num):

        print_loss_list = []

        for data in tqdm(train_loader):
            img, label = data

            img = img.cuda()
            label = torch.as_tensor(label).cuda()

            img_feature = resnet_18_encoder(img)

            img_feature = torch.as_tensor(img_feature).cuda()
            out = model(img_feature)

            loss = criterion(out, label)

            print_loss = loss.data.item()

            print_loss_list.append(print_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch: {}, loss: {:.4}'.format(epoch, np.mean(np.array(print_loss_list))))

    torch.save(model.state_dict(), './classify.pt')


if __name__ == "__main__":
    train()
