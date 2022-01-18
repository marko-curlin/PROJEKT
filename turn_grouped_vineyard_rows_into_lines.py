import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import ransac, LineModelND

from lib import util


def main():
    classified_data = np.loadtxt(util.construct_path(util.CLASSIFICATION_FOLDER,
                                                     "slice_01_inliers_k=20_dev=3_k=6_rotated_y-axis.txt"))

    point_cloud_2d: np.ndarray = classified_data[:, 0:2]
    y: np.ndarray = classified_data[:, 2].astype(int)

    data = point_cloud_2d

    s = 0.1

    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], s)

    # line_x = np.arange(data[:, 0].min()-1, data[:, 0].max()+1)

    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(np.unique(y)+1))))

    for i in np.unique(y):
        points_in_row = point_cloud_2d[classified_data[:, 2] == i]

        model, inliers = ransac(points_in_row, LineModelND, min_samples=2, residual_threshold=0.1, max_trials=100)

        ax.scatter(points_in_row[inliers, 0], points_in_row[inliers, 1], s, next(color), alpha=0.6, label=f'classified with {i}')

    plt.show()


if __name__ == '__main__':
    main()
