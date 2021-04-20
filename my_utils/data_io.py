import glob
import os
import shutil
import random

import numpy as np
from PIL import Image
from imagecorruptions import corrupt
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from config import Config


class coco_c(Dataset):
    def __init__(self, root_dir, size=(416, 416)):
        self.files = glob.glob(root_dir)
        self.size = size
        self.transform = transforms.Compose([transforms.Resize(256),  # 将图像调整为256×256像素
                                             transforms.CenterCrop(224),  # 将图像中心裁剪出来，大小为224×224像素
                                             transforms.ToTensor()  # 将图像转换为PyTorch张量（tensor）数据类型
                                             ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx])
        img = self.transform(img)
        img = np.array(img)
        label = int(self.files[idx][-18])  # todo 标签暂时为倒数18位
        return img, label


class Feature_set(Dataset):
    def __init__(self, feature_dir, label_dir, model_performance, map_threshold):
        self.feature = feature_dir
        self.label = label_dir
        self.model_performance = model_performance
        self.map_threshold = map_threshold

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, item):
        feature = self.feature[item]
        if self.model_performance[int(self.label[item])] > self.map_threshold:
            label = 1
        else:
            label = 0
        return feature, label


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


# def load_corruption_dataset():
#     # via number:
#     for i in range(15):
#         for severity in range(5):
#             corrupted = corrupt(image, corruption_number=i, severity=severity + 1)


def coco_loading():
    train_set = 'images/train2017'
    val_set = 'images/val2017'

    train_dir = os.path.join(Config.coco_dir, train_set)

    for img_train_name in os.listdir(train_dir):
        print(img_train_name)
        img_train = Image.open(os.path.join(train_dir, img_train_name)).convert('RGB')

    # img_train = Image.open('../bus.jpg')
    #
    # img_train.save('o.jpg')
    #
    # img_train = np.array(img_train)
    #
    # for i in range(15):
    #     for severity in range(5):
    #         corrupted = corrupt(img_train, corruption_number=i, severity=severity + 1)
    #
    #         corrupted_image = Image.fromarray(corrupted, 'RGB')
    #
    #         corrupted_image.save(str(i) + str(severity) + '.jpg')


def save_dataset(corrupted_image, img_name, corrupt_type, severity_type, img_type='train'):
    img_dir = os.path.join(Config.coco_c, img_type + str(corrupt_type) + '-' + str(severity_type))

    if not os.path.exists(img_dir):
        # 目录不存在创建，makedirs可以创建多级目录
        os.makedirs(img_dir)

    corrupted_image.save(os.path.join(img_dir, img_name))


def gen_each_class_train(corrupt_type, severity_type):
    train_dir = os.path.join(Config.coco_dir, Config.train_set)

    for img_train_name in tqdm(os.listdir(train_dir)):
        img_train = Image.open(os.path.join(train_dir, img_train_name)).convert('RGB')
        img_train = np.array(img_train)
        corrupted = corrupt(img_train, corruption_number=corrupt_type, severity=severity_type + 1)
        corrupted_image = Image.fromarray(corrupted, 'RGB')

        save_dataset(corrupted_image, img_train_name, corrupt_type, severity_type, img_type='train')


def gen_each_class_val(corrupt_type, severity_type):
    val_dir = os.path.join(Config.coco_dir, Config.val_set)

    for img_val_name in tqdm(os.listdir(val_dir)):
        img_val = Image.open(os.path.join(val_dir, img_val_name)).convert('RGB')
        img_val = np.array(img_val)
        corrupted = corrupt(img_val, corruption_number=corrupt_type, severity=severity_type + 1)
        corrupted_image = Image.fromarray(corrupted, 'RGB')

        save_dataset(corrupted_image, img_val_name, corrupt_type, severity_type, img_type='val')


def gen_class():
    for i in range(7, 11):  # 数字固定，与imagecorruptions库保持一致 7， 11
        for severity in range(2, 3):  # 3
            print("Start generating style %d and %d" % (i, severity))
            gen_each_class_train(i, severity)


def load_yolo(model):
    for k, v in model.named_parameters():
        print(k)


def classify_folder_gen_train():
    task_folder_list = ['../../coco_c/images/train7-2', '../../coco_c/images/train8-2',
                        '../../coco_c/images/train9-2', '../../coco_c/images/train10-2']
    task_label_list = ['../../coco_c/labels/train7-2', '../../coco_c/labels/train8-2',
                       '../../coco_c/labels/train9-2', '../../coco_c/labels/train10-2']
    sample_set_train_images = '../../coco_c/sample_set_train_images'
    sample_set_train_labels = '../../coco_c/sample_set_train_labels'
    length = 2000
    classify_folder_gen(sample_set_train_images, sample_set_train_labels, task_folder_list, task_label_list, length)


def classify_folder_gen(sample_set_images, sample_set_labels, task_folder_list, task_label_list, length):
    """
    采用文件拷贝的方式，构建代表集文件夹；同时拷贝图片和标签
    :param sample_set_images: 代表集图片地址，应当为新地址或者空文件夹
    :param sample_set_labels: 代表集标签地址，应当为新地址或者空文件夹
    :param task_folder_list: 源任务地址
    :param task_label_list: 源标签地址
    :param length: 需要的文件长度，即代表集的大小
    :return: 直接执行shell操作，无直接输出
    """

    # 检查目录是否存在
    if not os.path.exists(sample_set_images):
        os.makedirs(sample_set_images)

    if not os.path.exists(sample_set_labels):
        os.makedirs(sample_set_labels)

    for task_folder_index in range(len(task_label_list)):
        # label存在不全的情况，改用label作为索引去找image
        file_list = os.listdir(task_label_list[task_folder_index])

        random.shuffle(file_list)

        for file_obj_index in range(length):  # 随机选择样本构建特征集
            file_obj = file_list[file_obj_index]
            file_path = os.path.join(task_folder_list[task_folder_index], file_obj[:-4] + '.jpg')
            label_path = os.path.join(task_label_list[task_folder_index], file_obj)

            new_name_image = str(task_folder_index) + '_' + file_obj[:-4] + '.jpg'
            new_name_label = str(task_folder_index) + '_' + file_obj

            newfile_path = os.path.join(sample_set_images, str(new_name_image))
            newlabel_path = os.path.join(sample_set_labels, str(new_name_label))

            try:
                shutil.copyfile(file_path, newfile_path)
                shutil.copyfile(label_path, newlabel_path)
            except IOError:
                print("文件不存在", label_path)


def classify_folder_gen_val():
    task_folder_list = ['../../coco_c/images/val7-2', '../../coco_c/images/val8-2',
                        '../../coco_c/images/val9-2', '../../coco_c/images/val10-2']
    task_label_list = ['../../coco_c/labels/val7-2', '../../coco_c/labels/val8-2',
                       '../../coco_c/labels/val9-2', '../../coco_c/labels/val10-2']
    sample_set_val_images = '../../coco_c/sample_set_val_images'
    sample_set_val_labels = '../../coco_c/sample_set_val_labels'

    length = 500
    classify_folder_gen(sample_set_val_images, sample_set_val_labels, task_folder_list, task_label_list, length)


def check_file(file):
    # Search for file if not found
    if os.path.isfile(file) or file == '':
        return file
    else:
        files = glob.glob('./**/' + file, recursive=True)  # find file
        assert len(files), 'File Not Found: %s' % file  # assert file was found
        assert len(files) == 1, "Multiple files match '%s', specify exact path: %s" % (file, files)  # assert unique
        return files[0]  # return file


if __name__ == "__main__":
    classify_folder_gen_train()
    classify_folder_gen_val()