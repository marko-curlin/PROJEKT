from lib import util

from sklearn.linear_model import LinearRegression


def find_plane(point_cloud_data):
    xy, z = point_cloud_data[:, :2], point_cloud_data[:, 2]
    estimator = LinearRegression().fit(xy, z)

    plane_normal = util.get_normal(estimator.coef_)
    point_on_plane = util.get_point_on_cloud(estimator.coef_, estimator.intercept_)

    return plane_normal, point_on_plane


def main():
    ply_file_path = util.get_ply_file_path(util.SMALL_VINEYARD)
    data_points = util.read_ply_file_as_numpy_array(ply_file_path)

    xy, z = data_points[:, :2], data_points[:, 2]
    estimator = LinearRegression().fit(xy, z)

    print(f'regression coefficient: {estimator.coef_}')
    print(f'regression offset: {estimator.intercept_}')
    print()

    plane_normal = util.get_normal(estimator.coef_)
    point_on_plane = util.get_point_on_cloud(estimator.coef_, estimator.intercept_)

    print(f'plane normal: {plane_normal}')
    print(f'point on plane: {point_on_plane}')


if __name__ == '__main__':
    main()
