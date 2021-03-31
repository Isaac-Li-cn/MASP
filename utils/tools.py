import torch

from torchviz import make_dot


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
