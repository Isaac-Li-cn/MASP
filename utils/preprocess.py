import torch
import torchvision
import numpy as np

from tqdm import tqdm


def resnet_18_encoder(imgs):

    img_features = []

    model = torchvision.models.resnet18(pretrained=True)

    model = torch.nn.Sequential(*list(model.children())[:-1])  # 去掉网络的最后一层

    model = model.cuda()

    model.eval()

    for img in imgs:
        img = torch.unsqueeze(img, 0)
        img_feature = model(img)
        img_feature = img_feature.cpu()
        img_feature = img_feature.detach().numpy()
        img_features.append(img_feature)

    return img_features


def feature_getting(train_loader):
    print("开始代表集特征提取：")
    feature_set = np.zeros(shape=(0, 512))  # 初始化一个空数组
    for data in tqdm(train_loader):
        img, _ = data
        img = img.cuda()

        img_feature = np.array(resnet_18_encoder(img)).reshape(-1, 512)
        feature_set = np.concatenate((feature_set, img_feature), axis=0)
    return feature_set


if __name__ == "__main__":
    resnet_18_encoder()
