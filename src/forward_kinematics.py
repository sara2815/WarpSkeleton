import warp as wp
import numpy as np


@wp.kernel
def compose_local_transforms(
    bind_local_transforms: wp.array(dtype=wp.transform),
    euler_angles: wp.array(dtype=wp.vec3),
    out_local_transforms: wp.array(dtype=wp.transform),
):
    """Compose each bone's animated local transform by rotating the bind-pose
    transform's quaternion portion by the euler angles.

    The input bind_local_transforms already contain both translation and rotation.
    Euler angles are converted using XYZ rotation order (compose three axis-aligned
    rotations via wp.quat_from_axis_angle for X, Y, Z as qx * qy * qz) and then
    multiplied with the bind rotation.
    """
    # TODO: Implement compose_local_transforms kernel
    tid = wp.tid()
    angles = euler_angles[tid]
    bind_t = bind_local_transforms[tid]
    
    x_axis = wp.vec3(1.0, 0.0, 0.0)
    y_axis = wp.vec3(0.0, 1.0, 0.0)
    z_axis = wp.vec3(0.0, 0.0, 1.0)
    
    qx = wp.quat_from_axis_angle(x_axis, angles.x)
    qy = wp.quat_from_axis_angle(y_axis, angles.y)
    qz = wp.quat_from_axis_angle(z_axis, angles.z)
    
    
    # XYZ order
    q = qx * qy * qz  #apparently quaternion multiplcation does not have its own function
    
    ak = wp.transform(wp.vec3(0.0, 0.0, 0.0), q)
    
    out_local_transforms[tid] = wp.transform_multiply(bind_t, ak)



@wp.kernel
def compute_world_transforms(
    parent_indices: wp.array(dtype=wp.int32),
    local_transforms: wp.array(dtype=wp.transform),
    out_world_transforms: wp.array(dtype=wp.transform),
):
    """Quaternion forward kinematics using wp.transform: walk each bone's parent
    chain and accumulate world-space transform via transform_multiply."""
    # TODO: Implement compute_world_transforms kernel
    tid = wp.tid()
    #walk up the parent chain and accumulate transforms
    T = local_transforms[tid]

    parent = parent_indices[tid]

    # walk up the hierarchy
    while parent != -1:
        T = wp.transform_multiply(local_transforms[parent], T)
        parent = parent_indices[parent]

    out_world_transforms[tid] = T



def compose_local_transforms_func(bind_local_transforms: np.ndarray,
                                  euler_angles: np.ndarray, device) -> wp.array:
    """Compose local transforms from bind-pose transforms and Euler angles.

    Args:
        bind_local_transforms: Bind-pose transforms shape (num_bones,7)
        euler_angles: Euler angles in radians (num_bones, 3)
        device: Warp device

    Returns:
        Local transforms array (Warp, dtype=wp.transform)
    """
    num_bones = len(bind_local_transforms)
    wp_bind_xforms = wp.from_numpy(bind_local_transforms, dtype=wp.transform, device=device)
    wp_euler_angles = wp.from_numpy(euler_angles, dtype=wp.vec3, device=device)
    wp_local_transforms = wp.zeros(num_bones, dtype=wp.transform, device=device)

    wp.launch(compose_local_transforms, dim=num_bones,
              inputs=[wp_bind_xforms, wp_euler_angles],
              outputs=[wp_local_transforms], device=device)

    return wp_local_transforms

def compute_world_transforms_func(parent_indices: np.ndarray,
                                  local_transforms: wp.array, device) -> wp.array:
    """Compute world-space transforms from local transforms.

    Args:
        parent_indices: Parent indices for each bone (num_bones,)
        local_transforms: Local transforms (Warp array, dtype=wp.transform)
        device: Warp device

    Returns:
        World transforms array (Warp, dtype=wp.transform)
    """
    num_bones = len(parent_indices)
    wp_parent_indices = wp.from_numpy(parent_indices, dtype=wp.int32, device=device)
    wp_world_transforms = wp.zeros(num_bones, dtype=wp.transform, device=device)

    wp.launch(compute_world_transforms, dim=num_bones,
              inputs=[wp_parent_indices, local_transforms],
              outputs=[wp_world_transforms], device=device)

    return wp_world_transforms
