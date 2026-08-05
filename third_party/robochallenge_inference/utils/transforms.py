import numpy as np
from scipy.spatial.transform import Rotation as R


def quat_to_euler(quat, degrees=False):
    """Convert quaternion to Euler angles (roll, pitch, yaw)"""
    euler = R.from_quat(quat).as_euler("xyz", degrees=degrees)
    return euler


def euler_to_quat(euler, degrees=False):
    """Convert Euler angles to quaternion"""
    if euler.ndim == 1:
        quat = R.from_euler("xyz", euler, degrees=degrees).as_quat()
    else:
        quat = R.from_euler("xyz", euler, degrees=degrees).as_quat()
    return quat


def unwrap_euler_sequence(euler_seq):
    """Unwrap Euler angle sequence to avoid discontinuities"""
    return np.unwrap(euler_seq, axis=0)
