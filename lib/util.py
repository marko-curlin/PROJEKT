from pathlib import Path

import numpy as np
import open3d as o3d

import lib.constants as c
from lib.constants import *


# noinspection PyProtectedMember
def get_ply_file_path(version) -> Path:
    if version == OG:
        return construct_path(OBJ_FOLDER, c._OG_PLY_FILE_NAME)
    elif version == INLIERS:
        return construct_path(OBJ_FOLDER, c._INLINE_PLY_FILE_NAME)
    elif version == SLICE_01:
        return construct_path(OBJ_FOLDER, c._SLICE01_PLY_FILE_NAME)
    elif version == ABOVE_GROUND_06:
        return construct_path(OBJ_FOLDER, c._ABOVE_GROUND_06_PLY_FILE_NAME)
    elif version == SLICE_01_INLIERS_K_20_DEV_3:
        return construct_path(OBJ_FOLDER, c._SLICE01_INLIERS_K_20_DEV_3_PLY_FILE_NAME)
    elif version == SMALL_VINEYARD:
        return construct_path(OBJ_FOLDER, c._SMALL_VINEYARD_FILE_NAME)
    elif version == SLICE_01_INLIERS_K_20_DEV_3_ROTATED:
        return construct_path(OBJ_FOLDER, c._SLICE01_INLIERS_K_20_DEV_3_PLY_FILE_NAME_ROTATED)

    raise Exception("No correct version chosen")


def construct_path(*paths) -> Path:
    path = Path()
    for _path in paths:
        path = path.joinpath(_path)
    return path


def read_ply_file_as_numpy_array(ply_file_path) -> np.ndarray:
    scene = read_ply_file_as_o3d_point_cloud(ply_file_path)
    return np.asarray(scene.points)


def read_ply_file_as_o3d_point_cloud(ply_file_path) -> o3d.geometry.PointCloud:
    return o3d.io.read_point_cloud(str(ply_file_path))


def write_point_cloud_to_file(point_cloud: o3d.geometry.PointCloud, ply_file_path, write_ascii=True):
    if not o3d.io.write_point_cloud(str(ply_file_path), point_cloud, write_ascii=write_ascii, print_progress=True):
        raise Exception("Point cloud write failed")


def write_numpy_array_to_file(np_array: np.ndarray, ply_file_path, write_ascii=True):
    point_cloud = numpy_array_to_point_cloud_object(np_array)
    write_point_cloud_to_file(point_cloud, ply_file_path, write_ascii)


def o3d_point_cloud_to_numpy_array(point_cloud) -> np.ndarray:
    return np.asarray(point_cloud.points)


def map_3d_to_plane(plane_normal, point_on_plane, point_cloud):
    mapped_points = np.empty((len(point_cloud), 3))

    for i, point in enumerate(point_cloud):
        p_ = point - point_on_plane

        p_normal = np.dot(p_, plane_normal) * plane_normal
        p_tangent = p_ - p_normal

        mapped_point = p_tangent + point_on_plane
        mapped_points[i] = mapped_point

    return mapped_points


def slice_3d_with_plane(plane_normal, point_on_plane, point_cloud, distance_to_slice=.1):
    plane_slice = np.empty((len(point_cloud), 3))

    points_in_slice_counter = 0
    for point in point_cloud:
        p_ = point - point_on_plane

        dist_to_plane = np.dot(p_, plane_normal)
        if abs(dist_to_plane) > distance_to_slice:
            continue

        p_normal = np.dot(p_, plane_normal) * plane_normal
        p_tangent = p_ - p_normal

        mapped_point = p_tangent + point_on_plane
        plane_slice[points_in_slice_counter] = mapped_point
        points_in_slice_counter += 1

    plane_slice = np.delete(plane_slice, slice(points_in_slice_counter, len(plane_slice)), axis=0)

    return plane_slice


def get_points_above_plane(plane_normal, point_on_plane, point_cloud, max_distance=.5):
    plane_slice = np.empty((len(point_cloud), 3))

    points_above_plane_counter = 0
    for point in point_cloud:
        p_ = point - point_on_plane

        dist_to_plane = np.dot(p_, plane_normal)
        # dist_to_plane is vector distance <-inf, +inf>
        if 0 < dist_to_plane < max_distance:
            plane_slice[points_above_plane_counter] = point
            points_above_plane_counter += 1

    plane_slice = np.delete(plane_slice, slice(points_above_plane_counter, len(plane_slice)), axis=0)

    return plane_slice


def numpy_array_to_point_cloud_object(np_array):
    point_cloud = o3d.geometry.PointCloud()
    if np_array.shape[-1] == 3:
        point_cloud.points = o3d.utility.Vector3dVector(np_array)
    elif np_array.shape[-1] == 2:
        # expand third dimension with zeros
        np_array = np.hstack((np_array, np.zeros((np_array.shape[0], 1), dtype=np_array.dtype)))
        point_cloud.points = o3d.utility.Vector3dVector(np_array)
    else:
        raise Exception(f"can't create pointCloud with shape: {np_array.shape}")
    return point_cloud


def draw_cloud(*point_clouds):
    o3d.visualization.draw_geometries(point_clouds)


def remove_z_axis(data):
    if data.ndim == 1:
        return data[:2]
    return data[:, 0:2]


def get_normal(coefficient):
    return -coefficient[0], -coefficient[1], 1


def get_point_on_cloud(coefficient, intercept):
    return 0, 0, intercept


# taken from https://stackoverflow.com/a/58781388/6673178
def rotate_points(points, origin=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    # noinspection PyPep8Naming
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    points = np.atleast_2d(points)
    return np.squeeze((R @ (points.T - o.T) + o.T).T)


def get_degrees_from_direction(direction):
    angle = np.math.atan2(direction[1], direction[0])
    if abs(angle) > np.pi / 2:
        angle = angle - np.pi
    angle_degrees = np.degrees(angle)

    return angle_degrees
