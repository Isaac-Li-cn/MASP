import sys

from torch.utils.data import DataLoader

from config import Config
from utils.data_io import coco_c
from utils.preprocess import feature_getting

sys.path.append('./yolov5')
from yolo_test import test


if __name__ == "__main__":

    # 获取图片代表集
    sample_train = coco_c(Config.sample_train)
    sample_val = coco_c(Config.sample_val)

    train_loader = DataLoader(sample_train, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(sample_train, batch_size=Config.batch_size, shuffle=True)

    # 代表集特征抽取
    feature_set = feature_getting(train_loader)
