import numpy as np
from typing import Tuple


# =============================================== XYZ 2 POLAR ==========================================================
def xyz_2_polar(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a vector in 3D space (XYZ) to polar coordinates (phi/theta). Please note that the functions does not
    perform any checks and one can get unexpected results if the length of the vector != 1.

    Arguments:
        xyz: unit sphere coordinates in 3d (xyz)

    Returns:
        polar coordinates (phu, theta)

    References:
        - https://en.wikipedia.org/wiki/Spherical_coordinate_system
    """

    phi = np.arctan2(xyz[..., 0], xyz[..., 2])
    theta = np.arcsin(xyz[..., 1])

    return phi, theta


# =============================================== EQUI 2 POLAR =========================================================
def equi_2_polar(xy: np.ndarray, hw: Tuple) -> Tuple[np.ndarray, np.ndarray]:
    """
    Utility functions that converts equirectagular coordinates to polar coordinates.

    Args:
        xy: pixel cooridnates of shape [..., 2]
        hw: height and width of the equirectangular image

    Returns:
        phi, theta (or colloquially called lat,lon)

    References:
        - https://en.wikipedia.org/wiki/Spherical_coordinate_system
        - https://stackoverflow.com/questions/43741885/how-to-convert-spherical-coordinates-to-equirectangular-projection-coordinates
    """

    phi = ((xy[..., 0] + 0.5) / hw[0] - 0.5) * 2 * np.pi
    theta = ((xy[..., 1] + 0.5) / hw[1] - 0.5) * np.pi

    return phi, theta


# =============================================== POLAR 2 EQUI =========================================================
def polar_2_equi(phi: np.ndarray, theta: np.ndarray, hw: Tuple) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert from polar coordinates(radians) to equirectangular coordinates

    Args:
        phi: latitude (in radians)
        theta: longitude (in radians)
        hw: height and width of the target equirectagular image

    Returns:
        coordinates in pixel values (x - horizontal, y - vertical)

      References:
        - https://en.wikipedia.org/wiki/Spherical_coordinate_system
        - https://stackoverflow.com/questions/43741885/how-to-convert-spherical-coordinates-to-equirectangular-projection-coordinates
    """

    check_eq_shape(hw)

    x = (phi / (2 * np.pi) + 0.5) * hw[1] - 0.5
    y = (theta / np.pi + 0.5) * hw[0] - 0.5

    return x, y


# =============================================== ROTATE ===============================================================

def rotate_on_axis(coordinates_xyz, axis, angle):
    """
    Rotates a 3D point around a specified axis by a given angle.

    Args:
        coordinates_xyz: A NumPy array representing the 3D point (x, y, z).
        axis: A string representing the rotation axis ('x', 'y', or 'z').
        angle: The rotation angle in radians.

    Returns:
        A NumPy array representing the rotated 3D point.
    """
    # Convert the input point to a NumPy array if necessary
    coordinates_xyz = np.array(coordinates_xyz)

    # Create the rotation matrix based on the specified axis
    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
    elif axis == 'z':
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Invalid axis specified. Choose 'x', 'y', or 'z'.")

    # Apply the rotation matrix to the point
    rotated_coordinates = np.dot(rotation_matrix, coordinates_xyz)

    return rotated_coordinates

# =============================================== CHECKS ===============================================================
def check_eq_image_shape(eq_image: np.ndarray):
    if 2 * eq_image.shape[0] != eq_image.shape[1]:
        raise ValueError(f"Image of shape {eq_image.shape} doesn't have aspect ratio 2:1 (h:w)!")


def check_eq_shape(hw: Tuple):
    if 2 * hw[0] != hw[1]:
        raise ValueError(f"Shape [{hw}] should have the 2:1 aspect ratio. (h:w)!")
