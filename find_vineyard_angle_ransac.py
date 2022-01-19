import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import ransac, LineModelND

from lib import util


def find_vineyard_angle_ransac(point_cloud_data, residual_threshold=0.1, max_trials=500):
    point_cloud_2d = util.remove_z_axis(point_cloud_data)

    model, _ = ransac(point_cloud_2d, LineModelND, min_samples=2,
                      residual_threshold=residual_threshold, max_trials=max_trials)

    origin, direction = model.params
    return origin, direction


def main():
    ply_file_path = util.get_ply_file_path(util.SLICE_01_INLIERS_K_20_DEV_3)
    point_cloud = util.read_ply_file_as_numpy_array(ply_file_path)

    point_cloud_2d = util.remove_z_axis(point_cloud)

    model_robust, inliers = ransac(point_cloud_2d, LineModelND, min_samples=2,
                                   residual_threshold=0.1, max_trials=500)
    outliers = inliers == False

    data = point_cloud_2d

    line_x = np.arange(-45, -10)
    line_y_robust = model_robust.predict_y(line_x)

    origin, direction = model_robust.params

    # region plot results
    fig, ax = plt.subplots()
    ax.plot(data[inliers, 0], data[inliers, 1], '.b', alpha=0.6,
            label='Inlier data')
    ax.plot(data[outliers, 0], data[outliers, 1], '.r', alpha=0.6,
            label='Outlier data')
    ax.plot(line_x, line_y_robust, '-b', label='Robust line model')
    ax.legend(loc='upper left')

    plt.show()
    # endregion

    print("origin: ", origin)
    print("direction: ", direction)


if __name__ == '__main__':
    main()
