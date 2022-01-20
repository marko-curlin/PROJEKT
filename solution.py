from remove_noise import remove_noise
from find_plane import find_plane
from remove_ground import remove_ground
from find_vineyard_angle_ransac import find_vineyard_angle_ransac
from rotate_point_cloud import level_off_point_cloud
from group_vineyard_rows_GaussianMixture import group_vineyard_rows
from find_entrances_from_grouped_vineyard_rows import find_entrances_from_grouped_vineyard_rows

from lib import util


def show_entrances(points_above_plane, entrances):
    import matplotlib.pyplot as plt

    data = points_above_plane

    s = 0.1

    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], s, marker='.')

    ax.scatter(list(zip(*entrances))[0], list(zip(*entrances))[1], 10, marker='x')

    plt.show()


def find_entrances(point_cloud, nr_of_rows):
    # region remove_noise - nr_neighbours, std_ratio
    print("Remove starting noise. Using default values for nr_neighbours and std_ratio")

    inlier_point_cloud = remove_noise(point_cloud)
    inlier_point_cloud_data = util.o3d_point_cloud_to_numpy_array(inlier_point_cloud)
    # endregion

    # region find_plane
    print("Trying to discover plane parallel to ground using linear regression")

    normal, point_on_plane = find_plane(inlier_point_cloud_data)
    # endregion

    # region remove_ground - max_distance
    print("Removing ground from point cloud")

    points_above_plane = remove_ground(inlier_point_cloud_data, normal, point_on_plane, max_distance=0.1)
    # endregion

    # region find_vineyard_angle_ransac - residual_threshold, max_trials
    print("Finding angle at which vineyard rows are angled")

    origin, direction = find_vineyard_angle_ransac(points_above_plane, max_trials=500)
    # endregion

    # region - rotate_point_cloud
    print("Leveling off the vineyard")

    rotated_points_2d = level_off_point_cloud(points_above_plane, origin, direction)
    # endregion

    # region group_vineyard_rows
    print(f"Grouping vineyard rows. Using predefined information of there being {nr_of_rows} rows")

    rotated_points_2d, y = group_vineyard_rows(rotated_points_2d, nr_of_rows)
    # endregion

    # region find_entrances_from_grouped_vineyard_rows - residual_threshold, max_trials
    print("Finding entrances in the rotated point cloud")

    rotated_entrances = find_entrances_from_grouped_vineyard_rows(rotated_points_2d, y)
    # endregion

    # region rotate_entrances_back
    print("Rotating back the entrances found in previous step")

    degrees = util.get_degrees_from_direction(direction)
    real_entrances = util.rotate_points(rotated_entrances, origin, degrees)
    # endregion

    show_entrances(points_above_plane, real_entrances)

    return real_entrances


if __name__ == '__main__':
    ply_file_path = util.get_ply_file_path(util.OG)
    point_cloud = util.read_ply_file_as_o3d_point_cloud(ply_file_path)

    entrances = find_entrances(point_cloud, 6)
    print("Entrances:\n", entrances)
