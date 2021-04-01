import numpy as np

from config import Config
from context_classification import context_classification_by_kmeans
from context_classification import get_features

if __name__ == "__main__":

    # get_features(Config)

    img_features = np.load("img_features.npy")

    print(img_features.shape)

    context_classification_by_kmeans(img_features)
