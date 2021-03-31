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

    np.save("img_features.npy", img_features)


def context_classification_by_kmeans(img_features):

    img_features = np.squeeze(img_features)

    # print(img_features)

    y_pred = KMeans(n_clusters=10, random_state=1999).fit_predict(img_features)

    x = np.arange(len(y_pred))

    plt.scatter(x, y_pred, alpha=0.6)
    plt.show()

    dataframe = pd.DataFrame({'y_pred': y_pred})

    dataframe.to_csv("y_pred.csv", index=False, sep=',')
