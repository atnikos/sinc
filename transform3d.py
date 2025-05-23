from einops import rearrange
import numpy as np
import torch
from torch import Tensor
from roma import rotmat_to_rotvec, rotvec_to_rotmat
from torch.nn.functional import pad
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# Check PYTORCH3D_LICENCE before use

import functools
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
import torch
from einops import rearrange


def rotate_trajectory(traj, rotZ, inverse=False):
    '''
        Rotate the trajectory of a given body
    '''
    if inverse:
        # transpose
        rotZ = rearrange(rotZ, "... i j -> ... j i")

    vel = torch.diff(traj, dim=-2)
    # 0 for the first one => keep the dimentionality
    vel = torch.cat((0 * vel[..., [0], :], vel), dim=-2)
    vel_local = torch.einsum("...kj,...k->...j", rotZ[..., :2, :2], vel[..., :2])
    # Integrate the trajectory
    traj_local = torch.cumsum(vel_local, dim=-2)
    # First frame should be the same as before
    traj_local = traj_local - traj_local[..., [0], :] + traj[..., [0], :]
    return traj_local


def rotate_trans(trans, rotZ, inverse=False):
    '''
        Rotate the translation of a given body
    '''

    traj = trans[..., :2]
    transZ = trans[..., 2]
    traj_local = rotate_trajectory(traj, rotZ, inverse=inverse)
    trans_local = torch.cat((traj_local, transZ[..., None]), axis=-1)
    return trans_local


def rotate_body_degrees(rots, trans, offset=0.0):

    """
        Rotate the whole body
    """
    # rots, trans = data.rots.clone(), data.trans.clone()
    global_poses = rots[..., 0, :, :]
    global_euler = matrix_to_euler_angles(global_poses, "ZYX")
    anglesZ, anglesY, anglesX = torch.unbind(global_euler, -1)
    rotZ = _axis_angle_rotation("Z", anglesZ)

    diff_mat_rotZ = rotZ[..., 1:, :, :] @ rotZ.transpose(-1, -2)[..., :-1, :, :]
    vel_anglesZ = matrix_to_axis_angle(diff_mat_rotZ)[..., 2]
    # padding "same"
    vel_anglesZ = torch.cat((vel_anglesZ[..., [0]], vel_anglesZ), dim=-1)
    # canonicalizing here
    new_anglesZ = torch.cumsum(vel_anglesZ, -1) + offset
    new_rotZ = _axis_angle_rotation("Z", new_anglesZ)

    new_global_euler = torch.stack((new_anglesZ, anglesY, anglesX), -1)
    new_global_orient = euler_angles_to_matrix(new_global_euler, "ZYX")

    rots[:, 0] = new_global_orient
    trans = rotate_trans(trans, rotZ[0], inverse=False)
    trans = rotate_trans(trans, new_rotZ[0], inverse=True)
    # trans = rotate_trans(trans, rotZ[0], inverse=True)

    # from sinc.transforms.smpl import RotTransDatastruct
    # return RotTransDatastruct(rots=rots, trans=trans)
    return rots, trans

"""
The transformation matrices returned from the functions in this file assume
the points on which the transformation will be applied are column vectors.
i.e. the R matrix is structured as

    R = [
            [Rxx, Rxy, Rxz],
            [Ryx, Ryy, Ryz],
            [Rzx, Rzy, Rzz],
        ]  # (3, 3)

This matrix can be applied to column vectors by post multiplication
by the points e.g.

    points = [[0], [1], [2]]  # (3 x 1) xyz coordinates of a point
    transformed_points = R * points

To apply the same matrix to points which are row vectors, the R matrix
can be transposed and pre multiplied by the points:

e.g.
    points = [[0, 1, 2]]  # (1 x 3) xyz coordinates of a point
    transformed_points = points * R.transpose(1, 0)
"""


# Added
def matrix_of_angles(cos, sin, inv=False, dim=2):
    assert dim in [2, 3]
    sin = -sin if inv else sin
    if dim == 2:
        row1 = torch.stack((cos, -sin), axis=-1)
        row2 = torch.stack((sin, cos), axis=-1)
        return torch.stack((row1, row2), axis=-2)
    elif dim == 3:
        row1 = torch.stack((cos, -sin, 0*cos), axis=-1)
        row2 = torch.stack((sin, cos, 0*cos), axis=-1)
        row3 = torch.stack((0*sin, 0*cos, 1+0*cos), axis=-1)
        return torch.stack((row1, row2, row3),axis=-2)


def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _copysign(a, b):
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)


def _sqrt_positive_part(x):
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix):
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if isinstance(matrix, np.ndarray):
        matrix = torch.from_numpy(matrix)
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    o0 = 0.5 * _sqrt_positive_part(1 + m00 + m11 + m22)
    x = 0.5 * _sqrt_positive_part(1 + m00 - m11 - m22)
    y = 0.5 * _sqrt_positive_part(1 - m00 + m11 - m22)
    z = 0.5 * _sqrt_positive_part(1 - m00 - m11 + m22)
    o1 = _copysign(x, matrix[..., 2, 1] - matrix[..., 1, 2])
    o2 = _copysign(y, matrix[..., 0, 2] - matrix[..., 2, 0])
    o3 = _copysign(z, matrix[..., 1, 0] - matrix[..., 0, 1])
    return torch.stack((o0, o1, o2, o3), -1)


def _axis_angle_rotation(axis: str, angle):
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    if axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    if axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles, convention: str):
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = map(_axis_angle_rotation, convention, torch.unbind(euler_angles, -1))
    return functools.reduce(torch.matmul, matrices)


def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
):
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def _index_from_letter(letter: str):
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2


def matrix_to_euler_angles(matrix, convention: str):
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)


def random_quaternions(
    n: int, dtype: Optional[torch.dtype] = None, device=None, requires_grad=False
):
    """
    Generate random quaternions representing rotations,
    i.e. versors with nonnegative real part.

    Args:
        n: Number of quaternions in a batch to return.
        dtype: Type to return.
        device: Desired device of returned tensor. Default:
            uses the current device for the default tensor type.
        requires_grad: Whether the resulting tensor should have the gradient
            flag set.

    Returns:
        Quaternions as tensor of shape (N, 4).
    """
    o = torch.randn((n, 4), dtype=dtype, device=device, requires_grad=requires_grad)
    s = (o * o).sum(1)
    o = o / _copysign(torch.sqrt(s), o[:, 0])[:, None]
    return o


def random_rotations(
    n: int, dtype: Optional[torch.dtype] = None, device=None, requires_grad=False
):
    """
    Generate random rotations as 3x3 rotation matrices.

    Args:
        n: Number of rotation matrices in a batch to return.
        dtype: Type to return.
        device: Device of returned tensor. Default: if None,
            uses the current device for the default tensor type.
        requires_grad: Whether the resulting tensor should have the gradient
            flag set.

    Returns:
        Rotation matrices as tensor of shape (n, 3, 3).
    """
    quaternions = random_quaternions(
        n, dtype=dtype, device=device, requires_grad=requires_grad
    )
    return quaternion_to_matrix(quaternions)


def random_rotation(
    dtype: Optional[torch.dtype] = None, device=None, requires_grad=False
):
    """
    Generate a single random 3x3 rotation matrix.

    Args:
        dtype: Type to return
        device: Device of returned tensor. Default: if None,
            uses the current device for the default tensor type
        requires_grad: Whether the resulting tensor should have the gradient
            flag set

    Returns:
        Rotation matrix as tensor of shape (3, 3).
    """
    return random_rotations(1, dtype, device, requires_grad)[0]


def standardize_quaternion(quaternions):
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def quaternion_raw_multiply(a, b):
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def quaternion_multiply(a, b):
    """
    Multiply two quaternions representing rotations, returning the quaternion
    representing their composition, i.e. the versor with nonnegative real part.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions of shape (..., 4).
    """
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)


def quaternion_invert(quaternion):
    """
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """

    return quaternion * quaternion.new_tensor([1, -1, -1, -1])


def quaternion_apply(quaternion, point):
    """
    Apply the rotation given by a quaternion to a 3D point.
    Usual torch rules for broadcasting apply.

    Args:
        quaternion: Tensor of quaternions, real part first, of shape (..., 4).
        point: Tensor of 3D points of shape (..., 3).

    Returns:
        Tensor of rotated points of shape (..., 3).
    """
    if point.shape[-1] != 3:
        raise ValueError(f"Points are not in 3D, f{point.shape}.")
    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((real_parts, point), -1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion),
        quaternion_invert(quaternion),
    )
    return out[..., 1:]


def axis_angle_to_matrix(axis_angle):
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def matrix_to_axis_angle(matrix):
    """
    Convert rotations given as rotation matrices to axis/angle.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def axis_angle_to_quaternion(axis_angle):
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    try:
        sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    except:
        torch.save(axis_angle, f'before_convert_axis_angle.pt')
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


def quaternion_to_axis_angle(quaternions):
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalisation per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    return matrix[..., :2, :].clone().reshape(*matrix.shape[:-2], 6)


def rotate_trajectory(traj, rotZ, inverse=False):
    if inverse:
        # transpose
        rotZ = rearrange(rotZ, "... i j -> ... j i")

    vel = torch.diff(traj, dim=-2)
    # 0 for the first one => keep the dimentionality
    vel = torch.cat((0 * vel[..., [0], :], vel), dim=-2)
    vel_local = torch.einsum("...kj,...k->...j", rotZ[..., :2, :2], vel[..., :2])
    # Integrate the trajectory
    traj_local = torch.cumsum(vel_local, dim=-2)
    # First frame should be the same as before
    traj_local = traj_local - traj_local[..., [0], :] + traj[..., [0], :]
    return traj_local


def rotate_trans(trans, rotZ, inverse=False):
    traj = trans[..., :2]
    transZ = trans[..., 2]
    traj_local = rotate_trajectory(traj, rotZ, inverse=inverse)
    trans_local = torch.cat((traj_local, transZ[..., None]), axis=-1)
    return trans_local



def canonicalize_rotations(global_orient, trans, angle=torch.pi/4):
    global_euler = matrix_to_euler_angles(global_orient, "ZYX")
    anglesZ, anglesY, anglesX = torch.unbind(global_euler, -1)

    rotZ = _axis_angle_rotation("Z", anglesZ)

    # remove the current rotation
    # make it local
    local_trans = rotate_trans(trans, rotZ)

    # For information:
    # rotate_joints(joints, rotZ) == joints_local

    diff_mat_rotZ = rotZ[..., 1:, :, :] @ rotZ.transpose(-1, -2)[..., :-1, :, :]

    vel_anglesZ = matrix_to_axis_angle(diff_mat_rotZ)[..., 2]
    # padding "same"
    vel_anglesZ = torch.cat((vel_anglesZ[..., [0]], vel_anglesZ), dim=-1)

    # Compute new rotation:
    # canonicalized
    anglesZ = torch.cumsum(vel_anglesZ, -1)
    anglesZ += angle
    rotZ = _axis_angle_rotation("Z", anglesZ)

    new_trans = rotate_trans(local_trans, rotZ, inverse=True)

    new_global_euler = torch.stack((anglesZ, anglesY, anglesX), -1)
    new_global_orient = euler_angles_to_matrix(new_global_euler, "ZYX")

    return new_global_orient, new_trans


def rotate_motion_canonical(rotations, translation, transl_zero=True):
    """
    Must be of shape S x (Jx3)
    """
    rots_motion = rotations
    trans_motion = translation
    datum_len = rotations.shape[0]
    rots_motion_rotmat = transform_body_pose(rots_motion.reshape(datum_len,
                                                        -1, 3),
                                                        'aa->rot')
    orient_R_can, trans_can = canonicalize_rotations(rots_motion_rotmat[:,
                                                                            0],
                                                        trans_motion)            
    rots_motion_rotmat_can = rots_motion_rotmat
    rots_motion_rotmat_can[:, 0] = orient_R_can

    rots_motion_aa_can = transform_body_pose(rots_motion_rotmat_can,
                                                'rot->aa')
    rots_motion_aa_can = rearrange(rots_motion_aa_can, 'F J d -> F (J d)',
                                    d=3)
    if transl_zero:
        translation_can = trans_can - trans_can[0]
    else:
        translation_can = trans_can

    return rots_motion_aa_can, translation_can

def transform_body_pose(pose, formats):
    """
    various angle transformations, transforms input to torch.Tensor
    input:
        - pose: pose tensor
        - formats: string denoting the input-output angle format
    """
    if isinstance(pose, np.ndarray):
        pose = torch.from_numpy(pose)
    if formats == "6d->aa":
        j = pose.shape[-1] / 6
        pose = rearrange(pose, '... (j d) -> ... j d', d=6)
        pose = pose.squeeze(-2)  # in case of only one angle
        pose = rotation_6d_to_matrix(pose)
        pose = matrix_to_axis_angle(pose)
        if j > 1:
            pose = rearrange(pose, '... j d -> ... (j d)')
    elif formats == "aa->6d":
        j = pose.shape[-1] / 3
        pose = rearrange(pose, '... (j c) -> ... j c', c=3)
        pose = pose.squeeze(-2)  # in case of only one angle
        # axis-angle to rotation matrix & drop last row
        pose = matrix_to_rotation_6d(axis_angle_to_matrix(pose))
        if j > 1:
            pose = rearrange(pose, '... j d -> ... (j d)')
    elif formats == "aa->rot":
        j = pose.shape[-1] / 3
        pose = rearrange(pose, '... (j c) -> ... j c', c=3)
        pose = pose.squeeze(-2)  # in case of only one angle
        # axis-angle to rotation matrix & drop last row
        pose = torch.clamp(axis_angle_to_matrix(pose), min=-1.0, max=1.0)
    elif formats == "6d->rot":
        j = pose.shape[-1] / 6
        pose = rearrange(pose, '... (j d) -> ... j d', d=6)
        pose = pose.squeeze(-2)  # in case of only one angle
        pose = torch.clamp(rotation_6d_to_matrix(pose), min=-1.0, max=1.0)
    elif formats == "rot->aa":
        # pose = rearrange(pose, '... (j d1 d2) -> ... j d1 d2', d1=3, d2=3)
        pose = matrix_to_axis_angle(pose)
    elif formats == "rot->6d":
        # pose = rearrange(pose, '... (j d1 d2) -> ... j d1 d2', d1=3, d2=3)
        pose = matrix_to_rotation_6d(pose)
    else:
        raise ValueError(f"specified conversion format is invalid: {formats}")
    return pose

def apply_rot_delta(rots, deltas, in_format="6d", out_format="6d"):
    """
    rots needs to have same dimentionality as delta
    """
    assert rots.shape == deltas.shape
    if in_format == "aa":
        j = rots.shape[-1] / 3
    elif in_format == "6d":
        j = rots.shape[-1] / 6
    else:
        raise ValueError(f"specified conversion format is unsupported: {in_format}")
    rots = transform_body_pose(rots, f"{in_format}->rot")
    deltas = transform_body_pose(deltas, f"{in_format}->rot")
    new_rots = torch.einsum("...ij,...jk->...ik", rots, deltas)  # Ri+1=Ri@delta
    new_rots = transform_body_pose(new_rots, f"rot->{out_format}")
    if j > 1:
        new_rots = rearrange(new_rots, '... j d -> ... (j d)')
    return new_rots

def rot_diff(rots1, rots2=None, in_format="6d", out_format="6d"):
    """
    dim 0 is considered to be the time dimention, this is where the shift will happen
    """
    self_diff = False
    if in_format == "aa":
        j = rots1.shape[-1] / 3
    elif in_format == "6d":
        j = rots1.shape[-1] / 6
    else:
        raise ValueError(f"specified conversion format is unsupported: {in_format}")
    rots1 = transform_body_pose(rots1, f"{in_format}->rot")
    if rots2 is not None:
        rots2 = transform_body_pose(rots2, f"{in_format}->rot")
    else:
        self_diff = True
        rots2 = rots1
        rots1 = rots1.roll(1, 0)
        
    rots_diff = torch.einsum("...ij,...ik->...jk", rots1, rots2)  # Ri.T@R_i+1
    if self_diff:
        rots_diff[0, ..., :, :] = torch.eye(3, device=rots1.device)

    rots_diff = transform_body_pose(rots_diff, f"rot->{out_format}")
    if j > 1:
        rots_diff = rearrange(rots_diff, '... j d -> ... (j d)')
    return rots_diff

def change_for(p, R, T=0, forward=True):
    """
    Change frame of reference for vector p
    p: vector in original coordinate frame
    R: rotation matrix of new coordinate frame ([x, y, z] format)
    T: translation of new coordinate frame
    Let angle R by a.
    forward: rotates the coordinate frame by -a (True) or rotate the point
    by +a.
    """
    if forward:  # R.T @ (p_global - pelvis_translation)
        return torch.einsum('...di,...d->...i', R, p - T)
    else:  # R @ (p_global - pelvis_translation)
        return torch.einsum('...di,...i->...d', R, p) + T

def get_z_rot(rot_, in_format="6d"):
    rot = rot_.clone().detach()
    rot = transform_body_pose(rot, f"{in_format}->rot")
    euler_z = matrix_to_euler_angles(rot, "ZYX")
    euler_z[..., 1:] = 0.0
    z_rot = torch.clamp(
        euler_angles_to_matrix(euler_z, "ZYX"),
        min=-1.0, max=1.0)  # add zero XY euler angles
    return z_rot

def remove_z_rot(pose, in_format="6d", out_format="6d"):
    """
    zero-out the global orientation around Z axis
    """
    assert out_format == "6d"
    if isinstance(pose, np.ndarray):
        pose = torch.from_numpy(pose)
    # transform to matrix
    pose = transform_body_pose(pose, f"{in_format}->rot")
    pose = matrix_to_euler_angles(pose, "ZYX")
    pose[..., 0] = 0
    pose = matrix_to_rotation_6d(torch.clamp(
        euler_angles_to_matrix(pose, "ZYX"),
        min=-1.0, max=1.0))
    return pose

def local_to_global_orient(body_orient: Tensor, poses: Tensor, parents: list,
                           input_format='aa', output_format='aa'):
    """
    Modified from aitviewer
    Convert relative joint angles to global by unrolling the kinematic chain.
    This function is fully differentiable ;)
    :param poses: A tensor of shape (N, N_JOINTS*d) defining the relative poses in angle-axis format.
    :param parents: A list of parents for each joint j, i.e. parent[j] is the parent of joint j.
    :param output_format: 'aa' for axis-angle or 'rotmat' for rotation matrices.
    :param input_format: 'aa' or 'rotmat' ...
    :return: The global joint angles as a tensor of shape (N, N_JOINTS*DOF).
    """
    assert output_format in ['aa', 'rotmat']
    assert input_format in ['aa', 'rotmat']
    dof = 3 if input_format == 'aa' else 9
    n_joints = poses.shape[-1] // dof + 1
    if input_format == 'aa':
        body_orient = rotvec_to_rotmat(body_orient)
        local_oris = rotvec_to_rotmat(rearrange(poses, '... (j d) -> ... j d', d=3))
        local_oris = torch.cat((body_orient[..., None, :, :], local_oris), dim=-3)
    else:
        # this part has not been tested
        local_oris = torch.cat((body_orient[..., None, :, :], local_oris), dim=-3)
    global_oris_ = []

    # Apply the chain rule starting from the pelvis
    for j in range(n_joints):
        if parents[j] < 0:
            # root
            global_oris_.append(local_oris[..., j, :, :])
        else:
            parent_rot = global_oris_[parents[j]]
            local_rot = local_oris[..., j, :, :]
            global_oris_.append(torch.einsum('...ij,...jk->...ik', parent_rot, local_rot))
            # global_oris[..., j, :, :] = torch.bmm(parent_rot, local_rot)
    global_oris = torch.stack(global_oris_, dim=1)
    # global_oris: ... x J x 3 x 3
    # account for the body's root orientation
    # global_oris = torch.einsum('...ij,...jk->...ik', body_orient[..., None, :, :], global_oris)

    if output_format == 'aa':
        return rotmat_to_rotvec(global_oris)
        # res = global_oris.reshape((-1, n_joints * 3))
    else:
        return global_oris
    # return res