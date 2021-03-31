import torch
import torchvision


def resnet_18_encoder(imgs):

    img_features = []

    model = torchvision.models.resnet18(pretrained=True)

    for img in imgs:
        img = torch.unsqueeze(img, 0)
        img_feature = model(img)
        img_feature = img_feature.detach().numpy()
        img_features.append(img_feature)

    return img_features


if __name__ == "__main__":
    resnet_18_encoder()
