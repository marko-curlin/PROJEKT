import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import ransac, LineModelND

from lib import util


def find_entrances_from_grouped_vineyard_rows(point_cloud_2d, y, residual_threshold=0.1, max_trials=50):
    lines = []
    for i in np.unique(y):
        points_in_row = point_cloud_2d[y == i]

        model, inliers = ransac(points_in_row, LineModelND, min_samples=2,
                                residual_threshold=residual_threshold, max_trials=max_trials)

        inlier_points = points_in_row[inliers]

        min_index, max_index = np.argmin(inlier_points[:, 0]), np.argmax(inlier_points[:, 0])
        lines.append((inlier_points[min_index], inlier_points[max_index]))

    lines.sort(key=lambda line: line[0][1])

    entrances = find_vineyard_row_entrances(lines)

    return entrances


def find_vineyard_row_entrances(sorted_lines):
    entrances = []
    for i in range(len(sorted_lines)-1):
        first_left_point = sorted_lines[i][0]
        second_left_point = sorted_lines[i+1][0]

        left_entrance = [0, 0]
        left_entrance[0] = (first_left_point[0] + second_left_point[0])/2
        left_entrance[1] = (first_left_point[1] + second_left_point[1])/2

        first_right_point = sorted_lines[i][1]
        second_right_point = sorted_lines[i+1][1]

        right_entrance = [0, 0]
        right_entrance[0] = (first_right_point[0] + second_right_point[0])/2
        right_entrance[1] = (first_right_point[1] + second_right_point[1])/2

        entrances.append(left_entrance)
        entrances.append(right_entrance)

    return entrances


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

    lines = []
    for i in np.unique(y):
        points_in_row = point_cloud_2d[classified_data[:, 2] == i]

        model, inliers = ransac(points_in_row, LineModelND, min_samples=2, residual_threshold=0.1, max_trials=50)

        ax.scatter(points_in_row[inliers, 0], points_in_row[inliers, 1], s, next(color), alpha=0.6, label=f'classified with {i}')

        inlier_points = points_in_row[inliers]

        min_index, max_index = np.argmin(inlier_points[:, 0]), np.argmax(inlier_points[:, 0])
        lines.append((inlier_points[min_index], inlier_points[max_index]))

    lines.sort(key=lambda line: line[0][1])

    entrances = find_vineyard_row_entrances(lines)

    ax.scatter(list(zip(*entrances))[0], list(zip(*entrances))[1], 1)

    origin = (-28.23891965, -3.89433624)
    direction = (-0.99870772,  0.05082218)

    angle = np.math.atan2(direction[1], direction[0])
    if angle > np.pi/2:
        angle = angle - np.pi
    angle_degrees = np.degrees(angle)

    real_entrances = util.rotate_points(entrances, origin=origin, degrees=angle_degrees)

    ax.scatter(list(zip(*real_entrances))[0], list(zip(*real_entrances))[1], 1.5)

    plt.show()

    print(real_entrances)
# [[-42.67066412  -2.32738735]
#  [-12.14289997  -3.89336355]
#  [-42.59338645  -0.52349363]
#  [-12.94748769  -1.82434489]
#  [-42.32114541   1.66136394]
#  [-14.20885938   0.4228698 ]
#  [-42.11761097   3.83111381]
#  [-15.17969796   2.2806417 ]
#  [-41.45527506   5.69998256]
#  [-17.12025386   4.39097713]]


if __name__ == '__main__':
    main()
