from os.path import basename

import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

from lib import util


def main():
    ply_fle_path = util.get_ply_file_path(util.SLICE_01_INLIERS_K_20_DEV_3_ROTATED)
    point_cloud_data = util.read_ply_file_as_numpy_array(ply_fle_path)

    nr_of_groups = 6
    classifier = KMeans(nr_of_groups).fit(point_cloud_data[:, 1].reshape(-1, 1))

    y = classifier.predict(point_cloud_data[:, 1].reshape(-1, 1))

    plt.scatter(point_cloud_data[:, 0], point_cloud_data[:, 1], c=y)
    plt.title(f'file={basename(ply_fle_path)}\nclassifier={type(classifier)} k={nr_of_groups}', fontsize=10)
    plt.show()


if __name__ == '__main__':
    main()
