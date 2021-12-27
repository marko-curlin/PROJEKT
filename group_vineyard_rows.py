# import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

from lib import util


def main():
    ply_fle_path = util.get_ply_file_path(util.ABOVE_GROUND_06)
    point_cloud_data = util.read_ply_file_as_numpy_array(ply_fle_path)

    point_cloud_data_2d = util.remove_z_axis(point_cloud_data)

    nr_of_groups = 6
    classifier = GaussianMixture(nr_of_groups).fit(point_cloud_data)

    y = classifier.predict(point_cloud_data)

    plt.scatter(point_cloud_data_2d[:, 0], point_cloud_data_2d[:, 1], c=y)
    plt.show()


if __name__ == '__main__':
    main()
