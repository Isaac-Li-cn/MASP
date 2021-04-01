import os

from torchvision import transforms
from PIL import Image


def load_dataset(dataset_dir):
    """
    读取NICO中的单个concept的不同context数据，全都混起来，得到一个统一的image集
    """
    transform = transforms.Compose([
        transforms.Resize(256),  # 将图像调整为256×256像素
        transforms.CenterCrop(224),  # 将图像中心裁剪出来，大小为224×224像素
        transforms.ToTensor(),  # 将图像转换为PyTorch张量（tensor）数据类型
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]  # 通过将图像的平均值和标准差设置为指定的值来正则化图像
        )])

    imgs = []
    for context_name in os.listdir(dataset_dir):
        print(context_name)
        context_dir = os.path.join(dataset_dir, context_name)
        for img_name in os.listdir(context_dir):
            figure_dir = os.path.join(context_dir, img_name)
            img = Image.open(figure_dir).convert('RGB')  # 数据集中有jpg也有png，做一个转换
            img_t = transform(img)
            imgs.append(img_t)
        print(len(imgs))

    return imgs
