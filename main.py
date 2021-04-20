import os
import sys
import argparse
import joblib

sys.path.append('../yolov5')

from test_for_model import test

import torch

import numpy as np

from torch.utils.data import DataLoader

from config import Config
from my_utils.data_io import coco_c, Feature_set
from my_utils.preprocess import feature_getting
from task_model import model_list_train


def model_eval():
    map_list = []
    for model_data in Config.model_list:
        map_list_for_model = []
        for weight in Config.weight_list:
            result, _, _ = test(data=model_data, weights=weight, opt=opt)
            map_list_for_model.append(result[2])
        map_list.append(map_list_for_model)

    map_list = np.array(map_list)
    np.save('map_list.npy', map_list)
    return map_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')

    opt = parser.parse_args()

    # 获取图片代表集
    sample_train = coco_c(Config.sample_train)
    sample_val = coco_c(Config.sample_val)

    train_loader = DataLoader(sample_train, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(sample_val, batch_size=Config.batch_size, shuffle=True)

    # 代表集特征抽取
    if os.path.exists('feature_set_train.npy') and os.path.exists('label_set_train.npy') and \
            os.path.exists('feature_set_val.npy') and os.path.exists('label_set_val.npy'):
        feature_set_train = np.load('feature_set_train.npy')
        label_set_train = np.load('label_set_train.npy')
        feature_set_val = np.load('feature_set_val.npy')
        label_set_val = np.load('label_set_val.npy')
    else:
        feature_set_train, label_set_train = feature_getting(train_loader, train=True)
        feature_set_val, label_set_val = feature_getting(val_loader, train=False)

    # 模型性能验证
    if os.path.exists('map_list.npy'):
        map_list = np.load('map_list.npy')
    else:
        map_list = model_eval()

    model_map = map_list.T

    # 模型选择器生成

    if os.path.exists('model_chooser_list.m'):
        model_chooser_list = joblib.load('model_chooser_list.m')
    else:
        model_chooser_list = model_list_train(model_map, feature_set_train, label_set_train,
                                              feature_set_val, label_set_val)

    # 模型选择结果

    data_val = coco_c(Config.dataset_val)
