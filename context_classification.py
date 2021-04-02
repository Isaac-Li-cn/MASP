import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, DBSCAN

from utils.data_io import load_dataset
from utils.preprocess import resnet_18_encoder
from utils.tools import non_iid_index


def get_features(Config):
    dataset_dir = Config.dataset_dir

    imgs = load_dataset(dataset_dir)

    img_features = resnet_18_encoder(imgs)

    img_features = np.array(img_features)

    img_features = np.squeeze(img_features)

    print(img_features.shape)

    # np.save("img_features.npy", img_features)


def context_classification_by_kmeans(img_features):
    # print(img_features)

    n_class = int(len(img_features) / 100)

    print(n_class)

    y_pred = KMeans(n_clusters=n_class, random_state=2316).fit_predict(img_features)

    # print(y_pred)

    kmeans_array = save_kmeans_array(img_features, y_pred)

    # # 画图代码
    #
    # x = np.arange(len(y_pred))
    #
    # plt.scatter(x, y_pred, alpha=0.6, s=1)
    # plt.axvline(x=255, color='r', linestyle='-')
    # plt.axvline(x=398, color='r', linestyle='-')
    # plt.axvline(x=542, color='r', linestyle='-')
    # plt.axvline(x=629, color='r', linestyle='-')
    # plt.axvline(x=909, color='r', linestyle='-')
    # plt.axvline(x=1072, color='r', linestyle='-')
    # plt.axvline(x=1194, color='r', linestyle='-')
    # plt.axvline(x=1481, color='r', linestyle='-')
    # plt.axvline(x=1582, color='r', linestyle='-')
    # plt.axvline(x=1675, color='r', linestyle='-')
    # plt.show()

    # # 保存结果
    #
    # dataframe = pd.DataFrame({'y_pred': y_pred})
    #
    # dataframe.to_csv("y_pred.csv", index=False, sep=',')

    return kmeans_array


def save_kmeans_array(img_features, cluster_result):
    array_len = len(np.unique(cluster_result))  # 数组的长度为类数

    # 初始化一个数组，其每个元素都是一个kmeans的聚类结果
    kmeans_array = [[] for _ in range(array_len)]

    for img_index in range(len(img_features)):
        kmeans_array[cluster_result[img_index]].append(img_features[img_index])

    return kmeans_array


def context_cluster_by_dbscan(kmeans_array):
    # 计算距离矩阵

    cluster_len = len(kmeans_array)

    distance_matrix = np.zeros((cluster_len, cluster_len))

    for i in range(cluster_len):
        for j in range(cluster_len):
            if i == j:
                distance_matrix[i][j] = 0
            else:
                distance_matrix[i][j] = non_iid_index(kmeans_array[i], kmeans_array[j])

    print(distance_matrix[0])

    sns.heatmap(data=distance_matrix, vmin=10, vmax=20, cmap='Blues')

    plt.show()

    # clustering = DBSCAN(eps=13, min_samples=5, metric='precomputed').fit(distance_matrix)
    #
    # print(len(clustering.labels_))
    # print(clustering.labels_)
