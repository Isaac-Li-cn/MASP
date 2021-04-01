import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from utils.data_io import load_dataset
from utils.preprocess import resnet_18_encoder


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

    y_pred = KMeans(n_clusters=10, random_state=2316).fit_predict(img_features)

    x = np.arange(len(y_pred))

    plt.scatter(x, y_pred, alpha=0.6, s=1)
    plt.axvline(x=255, color='r', linestyle='-')
    plt.axvline(x=398, color='r', linestyle='-')
    plt.axvline(x=542, color='r', linestyle='-')
    plt.axvline(x=629, color='r', linestyle='-')
    plt.axvline(x=909, color='r', linestyle='-')
    plt.axvline(x=1072, color='r', linestyle='-')
    plt.axvline(x=1194, color='r', linestyle='-')
    plt.axvline(x=1481, color='r', linestyle='-')
    plt.axvline(x=1582, color='r', linestyle='-')
    plt.axvline(x=1675, color='r', linestyle='-')
    plt.show()

    dataframe = pd.DataFrame({'y_pred': y_pred})

    dataframe.to_csv("y_pred.csv", index=False, sep=',')
