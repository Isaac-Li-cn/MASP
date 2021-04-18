import os
import torch
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import wandb

from torch.utils.data import DataLoader

from utils.data_io import coco_c
from utils.preprocess import resnet_18_encoder
from config import Config

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(512, 32)
        self.fc2 = nn.Linear(32, 4)

    def forward(self, x):
        x = x.view([-1, 512])
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def train():
    wandb.init(project='task_model')

    batch_size = Config.batch_size
    learning_rate = Config.learning_rate
    epoch_num = Config.epoch_num

    config = wandb.config

    config.batch_size = batch_size
    config.learning_rate = learning_rate
    config.epoch_num = epoch_num
    config.train_set = Config.classify_folder_train
    config.val_set = Config.classify_folder_val

    coco_c_train = coco_c(Config.classify_folder_train)
    coco_c_val = coco_c(Config.classify_folder_val)

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

        test_acc = test(weight=model.state_dict())

        wandb.log({"train_loss": np.mean(np.array(print_loss_list)), "test_acc": test_acc})

        print('epoch: {}, loss: {:.4}'.format(epoch, np.mean(np.array(print_loss_list))))

    torch.save(model.state_dict(), './classify_mini.pt')
    wandb.save('./classify_mini.pt')


def test(weight=None):
    model = Net()
    if weight is None:
        model.load_state_dict(torch.load('classify.pt'))

    model.load_state_dict(weight)

    model = model.cuda()

    model.eval()

    coco_c_val = coco_c(Config.classify_folder_val)
    val_loader = DataLoader(coco_c_val, batch_size=Config.batch_size, shuffle=True)

    eval_acc = 0
    for data in tqdm(val_loader):
        img, label = data

        img = img.cuda()
        label = torch.as_tensor(label).cuda()

        img_feature = resnet_18_encoder(img)

        img_feature = torch.as_tensor(img_feature).cuda()
        out = model(img_feature)

        _, pred = torch.max(out, 1)

        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()

    print('Acc: {:.6f}'.format(
            eval_acc / (coco_c_val.__len__())
    ))

    return eval_acc / (coco_c_val.__len__())


if __name__ == "__main__":
    train()
