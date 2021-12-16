from lib import util

from sklearn.linear_model import LinearRegression


def main():
    ply_file_path = util.get_ply_file_path("OG")
    data_points = util.ply_file_to_numpy_array(ply_file_path)

    xy, z = data_points[:, :2], data_points[:, 2]
    estimator = LinearRegression().fit(xy, z)

    print(f'regression coefficient: {estimator.coef_}')
    print(f'regression offset: {estimator.intercept_}')


if __name__ == '__main__':
    main()
