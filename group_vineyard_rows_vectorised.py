import math

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

from lib import util


def get_direction_vectors(n):
    for i in range(n+1):
        radian_angle = np.pi / n * i
        yield np.asarray((np.cos(radian_angle), np.sin(radian_angle)))


def mean_square_error(point, direction_vector, neighbours):
    second_point = point + direction_vector
    mse = 0
    for neighbour in neighbours:
        distance_to_point = norm(np.cross(second_point - point, point - neighbour)) / norm(second_point - point)
        mse += np.square(distance_to_point)

    return mse


def find_best_direction_vector(point, neighbours):
    smallest_mse, best_direction_vector = np.inf, None
    # counter_best_vectors = 0
    for direction_vector in get_direction_vectors(180):
        mse = mean_square_error(point, direction_vector, neighbours)
        if mse < smallest_mse:
            smallest_mse = mse
            best_direction_vector = direction_vector
            # counter_best_vectors += 1

    # print(f'Decided on a new best vector {counter_best_vectors} times')

    return best_direction_vector


def find_direction(point_cloud_data):
    from scipy.spatial import KDTree

    sample_size = 3

    kdt = KDTree(point_cloud_data)
    sum_direction_vectors = np.asarray((0, 0), dtype=float)
    best_directions = []
    for point in point_cloud_data[:sample_size]:
        nearest_neighbours_indices = kdt.query(point, k=1001)[1]
        nearest_neighbours = point_cloud_data[nearest_neighbours_indices[1:]]  # first one is always the point itself

        best_direction = find_best_direction_vector(util.remove_z_axis(point), util.remove_z_axis(nearest_neighbours))
        best_directions.append(best_direction)
        sum_direction_vectors += best_direction

    # direction_vector = sum_direction_vectors / sample_size
    return sum_direction_vectors, best_directions


def main():
    ply_fle_path = util.get_ply_file_path(util.SLICE_01_INLIERS_K_20_DEV_3)
    point_cloud_data = util.read_ply_file_as_numpy_array(ply_fle_path)

    point_cloud_data_2d = util.remove_z_axis(point_cloud_data)

    nr_of_groups = 6
    classifier = None  # create vector classifier, and give it the vector

    direction_vector, best_directions = find_direction(point_cloud_data)
    print(direction_vector)
    # y = classifier.predict(point_cloud_data_2d)

    plt.scatter(point_cloud_data_2d[:, 0], point_cloud_data_2d[:, 1])
    plt.scatter(direction_vector[0], direction_vector[1], marker='^')
    plt.scatter(0, 0, marker='^')
    plt.show()


if __name__ == '__main__':
    main()
