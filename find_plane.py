from lib import util

from sklearn.linear_model import LinearRegression


def main():
    ply_file_path = util.get_ply_file_path("inliers")
    data_points = util.read_ply_file_as_numpy_array(ply_file_path)

    xy, z = data_points[:, :2], data_points[:, 2]
    estimator = LinearRegression().fit(xy, z)

    print(f'regression coefficient: {estimator.coef_}')
    print(f'regression offset: {estimator.intercept_}')


if __name__ == '__main__':
    main()
