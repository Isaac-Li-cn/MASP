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

    # 代表集样本及标签
    sample_set = '/yolov5/data/sample.yaml'

    #########################################################
    # test参数

    model_list = ['base_model/coco_c_7_2.yaml', 'base_model/coco_c_8_2.yaml',
                  'base_model/coco_c_9_2.yaml', 'base_model/coco_c_10_2.yaml']

    weight_list = ['base_model/model_7_2.pt', 'base_model/model_8_2.pt',
                   'base_model/model_9_2.pt', 'base_model/model_10_2.pt']

    ##########################################################
    # 场景分类器参数
    learning_rate = 0.02
    epoch_num = 25

    ##########################################################
    # 算法超参数
    map_threshold = 0.42

    ##########################################################
    # 验证集大小
    val_set_len = 2000  # todo 应当作为参数传入，当前实现先人工计算
