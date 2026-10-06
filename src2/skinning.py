from torch import device
import warp as wp
import numpy as np

MAX_BIND_BONES = 4  # maximum number of bone influences per vertex for skinning

wp.constant(MAX_BIND_BONES)

@wp.kernel
def compute_material_coords(
    vert_world: wp.array(dtype=wp.vec3),
    vert_bone_indices: wp.array(dtype=wp.int32, ndim=2),
    world_transforms: wp.array(dtype=wp.transform),
    vert_bone: wp.array(dtype=wp.vec3, ndim=2)
):
    """Transform world-space vertices into bone-local (material) coordinates
    using inverse bind transforms, for use in linear blend skinning."""
    # TODO: Implement material coordinate computation
    tid = wp.tid()
    pos = vert_world[tid]

    bone_count = vert_bone_indices.shape[1]

    for i in range(bone_count):
        bone_idx = vert_bone_indices[tid, i]
        # TODO: Your implementation here
        # invalid bone entry
        if bone_idx == -1:
            vert_bone[tid, i] = wp.vec3(0.0, 0.0, 0.0)

        else:
            # bind-pose world transform
            bw = world_transforms[bone_idx]

            # inverse bind transform
            bw_inv = wp.transform_inverse(bw)

            # convert world position -> bone local
            x_material = wp.transform_point(bw_inv, pos)

            vert_bone[tid, i] = x_material


@wp.kernel
def compute_vertex_positions(
    vert_bone_pos: wp.array(dtype=wp.vec3, ndim=2),
    vert_bone_indices: wp.array(dtype=wp.int32, ndim=2),
    vert_bone_weights: wp.array(dtype=wp.float32, ndim=2),
    world_transforms: wp.array(dtype=wp.transform),
    out_verts: wp.array(dtype=wp.vec3),
):
    """Linear blend skinning: blend bone-transformed material coordinates
    using wp.transform_point and per-vertex bone weights."""
    # TODO: Implement linear blend skinning
    tid = wp.tid()
    final_pos = wp.vec3(0.0, 0.0, 0.0)
    bone_count = vert_bone_weights.shape[1]

    for i in range(bone_count):
        w = vert_bone_weights[tid, i]
        bone_ind = vert_bone_indices[tid, i]
        # TODO: Your implementation here
        
        if bone_ind == -1:
          continue
        if w < 1e-5:
            continue
        
        # material coordinate
        x_material = vert_bone_pos[tid, i]

        # animated world transform
        T = world_transforms[bone_ind]

        # transform to world space
        x_world = wp.transform_point(T, x_material)

        # weighted contribution
        final_pos += w * x_world
    out_verts[tid] = final_pos

def compute_material_coordinates(verts: np.ndarray, bone_indices: np.ndarray,
                                world_bind_transforms: wp.array, device) -> wp.array:
    """Compute material coordinates for skinning.

    Args:
        verts: Vertex positions (num_verts, 3)
        bone_indices: Bone indices per vertex (num_verts, max_bones)
        world_bind_transforms: World bind transforms (Warp array, dtype=wp.transform)
        device: Warp device

    Returns:
        Material coordinates array (Warp, num_verts, max_bones, 3)
    """
    num_verts = len(verts)
    max_bones = bone_indices.shape[1]

    wp_verts = wp.from_numpy(verts, dtype=wp.vec3, device=device)
    wp_bone_indices = wp.from_numpy(bone_indices, dtype=wp.int32, device=device)
    wp_material_coords = wp.zeros((num_verts, max_bones), dtype=wp.vec3, device=device)

    wp.launch(
        kernel=compute_material_coords,
        dim=num_verts,
        inputs=[wp_verts, wp_bone_indices, world_bind_transforms],
        outputs=[wp_material_coords],
        device=device
    )

    return wp_material_coords


def linear_blend_skin(material_coords: wp.array, bone_indices: np.ndarray,
                     bone_weights: np.ndarray, world_transforms: wp.array,
                     device) -> wp.array:
    """Perform linear blend skinning.

    Args:
        material_coords: Material coordinates (Warp array, num_verts, max_bones, 3)
        bone_indices: Bone indices per vertex (num_verts, max_bones)
        bone_weights: Bone weights per vertex (num_verts, max_bones)
        world_transforms: World transforms (Warp array, dtype=wp.transform)
        device: Warp device

    Returns:
        Deformed vertex positions (Warp array)
    """
    num_verts = material_coords.shape[0]

    wp_bone_indices = wp.from_numpy(bone_indices, dtype=wp.int32, device=device)
    wp_bone_weights = wp.from_numpy(bone_weights, dtype=wp.float32, device=device)
    wp_deformed_verts = wp.zeros(num_verts, dtype=wp.vec3, device=device)

    wp.launch(
        kernel=compute_vertex_positions,
        dim=num_verts,
        inputs=[material_coords, wp_bone_indices, wp_bone_weights,
                world_transforms],
        outputs=[wp_deformed_verts],
        device=device
    )

    return wp_deformed_verts


####################################
### dual quaternion skinning   ####
####################################

@wp.func
def quat_conjugate(q: wp.quat):
    return wp.quat(-q[0], -q[1], -q[2], q[3])

@wp.kernel
def dqs_skin_kernel(
    rest_vertices: wp.array(dtype=wp.vec3),
    bone_indices: wp.array(dtype=wp.int32, ndim=2),
    bone_weights: wp.array(dtype=wp.float32, ndim=2),
    world_transforms: wp.array(dtype=wp.transform),
    bind_transforms: wp.array(dtype=wp.transform), # Added bind transforms
    out_vertices: wp.array(dtype=wp.vec3),
):
    vid = wp.tid()
    v = rest_vertices[vid]

    qr = wp.quat(0.0, 0.0, 0.0, 0.0)
    qd = wp.quat(0.0, 0.0, 0.0, 0.0)

    #reference quaternion to avoid "cancelling out" rotations.
    q0 = wp.quat(0.0, 0.0, 0.0, 1.0)
    is_first_bone = float(1.0)

    for k in range(MAX_BIND_BONES):
        w = bone_weights[vid, k]
        if w < 1e-6:
            continue

        bone = bone_indices[vid, k]
        if bone == -1:
            continue

        # Calculate the transformation relative to bind pose
        # M = World_Current * Inverse(World_Bind)
        T_curr = world_transforms[bone]
        T_bind_inv = wp.transform_inverse(bind_transforms[bone])
        M = T_curr * T_bind_inv

        t = wp.transform_get_translation(M)
        r = wp.transform_get_rotation(M)

        # Neighborhood check: Ensure the shortest path for blending
        if is_first_bone > 0.5:
            q0 = r
            is_first_bone = 0.0
        else: 
            if wp.dot(q0, r) < 0.0:
             w = -w

        t_quat = wp.quat(t[0], t[1], t[2], 0.0)
       
        qr += w * r
        qd += w * (0.5 * t_quat * r)

    # Normalize the dual quaternion
    norm = wp.length(qr)
    qr /= norm
    qd /= norm

    # Apply the Dual Quaternion transformation + Rotation
    v_rot = wp.quat_rotate(qr, v)

    # Translation: 2 * qd * conjugate(qr)
    # The real part of this result is 0, the imaginary part is the translation vector
    t_dual = 2.0 * (qd * wp.quat(-qr[0], -qr[1], -qr[2], qr[3]))
    translation = wp.vec3(t_dual[0], t_dual[1], t_dual[2])

    out_vertices[vid] = v_rot + translation

def dual_quaternion_skin(
    rest_vertices,        # This should be mesh_data.verts
    bone_indices,
    bone_weights,
    world_transforms,
    world_bind_transforms, # Added this argument
    device,
):
    num_verts = rest_vertices.shape[0]

    # Ensure inputs are Warp arrays
    if not isinstance(rest_vertices, wp.array):
        wp_verts = wp.from_numpy(rest_vertices, dtype=wp.vec3, device=device)
    else:
        wp_verts = rest_vertices

    # Launch kernel
    out_vertices = wp.zeros(num_verts, dtype=wp.vec3, device=device)
    wp.launch(
        kernel=dqs_skin_kernel,
        dim=num_verts,
        inputs=[
            wp_verts,
            bone_indices,
            bone_weights,
            world_transforms,
            world_bind_transforms
        ],
        outputs=[out_vertices],
        device=device
    )
    return out_vertices