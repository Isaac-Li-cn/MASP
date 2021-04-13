import torch

import numpy as np

from config import Config
from context_classification import context_classification_by_kmeans, context_cluster_by_hierarchy_cluster
from context_classification import get_features
from context_classification import context_cluster_by_dbscan
from utils.data_io import gen_class, load_yolo
from imagecorruptions import get_corruption_names

if __name__ == "__main__":

    # get_features(Config)

    gen_class()

    # for corruption in get_corruption_names():
    #     print(corruption)

    # for index in range(7, 11):
    #     print(get_corruption_names()[index])

    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    #
    # print(model)
