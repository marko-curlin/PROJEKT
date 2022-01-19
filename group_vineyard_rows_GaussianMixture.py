from os.path import basename

import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

from lib import util


def group_vineyard_rows(point_cloud_data, nr_of_groups):
    classifier = KMeans(nr_of_groups).fit(point_cloud_data[:, 1].reshape(-1, 1))

    y = classifier.predict(point_cloud_data[:, 1].reshape(-1, 1))

    return point_cloud_data, y


def main():
    ply_fle_path = util.get_ply_file_path(util.SLICE_01_INLIERS_K_20_DEV_3_ROTATED)
    point_cloud_data = util.read_ply_file_as_numpy_array(ply_fle_path)

    nr_of_groups = 6
    classifier = KMeans(nr_of_groups).fit(point_cloud_data[:, 1].reshape(-1, 1))

    y = classifier.predict(point_cloud_data[:, 1].reshape(-1, 1))

    plt.scatter(point_cloud_data[:, 0], point_cloud_data[:, 1], c=y)
    plt.title(f'file={basename(ply_fle_path)}\nclassifier={type(classifier)} k={nr_of_groups}', fontsize=10)
    plt.show()

    np.savetxt(util.construct_path(util.CLASSIFICATION_FOLDER, "slice_01_inliers_k=20_dev=3_k=6_rotated_y-axis.txt"),
               np.hstack((point_cloud_data[:, 0:2], y.reshape(y.shape[0], 1))))


if __name__ == '__main__':
    main()
