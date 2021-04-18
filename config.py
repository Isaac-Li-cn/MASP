
class Config:

    # ############################################################################
    # # 一些曾经的参数
    #
    # dataset_dir = 'Animal/dog/'
    #
    # coco_dir = '../coco'
    #
    # train_set = 'images/train2017'
    # val_set = 'images/val2017'
    #
    # coco_c = '../coco_c/images'
    #
    # # 场景分类器的超参数
    # batch_size = 1024
    # learning_rate = 0.04
    # epoch_num = 25
    #
    # # 场景分类器的训练集验证集的地址
    # classify_folder_train = '../coco_c/classify_folder_train_mini/*.jpg'
    # classify_folder_val = '../coco_c/classify_folder_val_mini/*.jpg'

    ################################################################################
    # 目前使用的参数

    # 图片集的地址
    dataset_train = '../coco_c/classify_folder_train/*.jpg'
    dataset_val = '../coco_c/classify_folder_val/*.jpg'

    # 代表集的地址
    sample_train = '../coco_c/classify_folder_train_mini/*.jpg'
    sample_val = '../coco_c/classify_folder_val_mini/*.jpg'

    # batch大小
    batch_size = 1024
