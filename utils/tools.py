import torch

from PIL import Image
from torchvision import transforms, models
from torchviz import make_dot
import matplotlib.pyplot as plt


def network_visualization(model):
    """
    使用graphviz进行神经网络可视化
    :param model: 要可视化的神经网络
    :return: 无返回值，生成一个神经网络的pdf文件
    """
    x = torch.rand(8, 3, 256, 512)
    y = model(x)
    g = make_dot(y)
    g.render('espnet_model', view=False)


def get_img_to_inference(img_dir="../Animal/dog/eating/3.jpg"):
    transform = transforms.Compose([  # [1]
        transforms.Resize(256),  # [2]
        transforms.CenterCrop(224),  # [3]
        transforms.ToTensor(),  # [4]
        transforms.Normalize(  # [5]
            mean=[0.485, 0.456, 0.406],  # [6]
            std=[0.229, 0.224, 0.225]  # [7]
        )])

    img = Image.open(img_dir)

    img_t = transform(img)

    plt.imshow(img)
    plt.show()

    batch_t = torch.unsqueeze(img_t, 0)

    batch_t_gpu = batch_t.cuda()

    return batch_t_gpu


def network_evaluation(__model, __batch_t_gpu):
    __model.eval()

    out = __model(__batch_t_gpu)

    _, index = torch.max(out, 1)

    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

    print(percentage[index[0]], index)


def network_encoder(_model, _batch_t_gpu):

    _model = torch.nn.Sequential(*list(_model.children())[:-1])  # 去掉网络的最后一层

    _model.eval()

    out = _model(_batch_t_gpu)

    print(out.shape)


if __name__ == "__main__":
    model = models.resnet18(pretrained=True).cuda()

    batch_t_gpu = get_img_to_inference()

    # network_evaluation(model, batch_t_gpu)

    network_encoder(model, batch_t_gpu)
