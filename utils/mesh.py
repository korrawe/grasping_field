# Copyright 2004-present Facebook. All Rights Reserved.

import logging
import numpy as np
import plyfile
import skimage.measure
import time
import torch

import utils.utils
from utils.customized_export_ply import customized_export_ply


def create_mesh_combined_decoder(
    hand_branch, obj_branch,
    decoder, latent_vec, filename, N=256, max_batch=32 ** 3, offset=None, scale=None,
    device="cpu", label_out=False, viz=False
):
    print("...Marching cubes...")
    start = time.time()
    ply_filename_hand = filename + "_hand"
    ply_filename_obj = filename + "_obj"

    decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 6) #4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    end_sample = time.time()
    # print("Creating uniform points: %f" % (end_sample - start))

    samples.requires_grad = False

    head = 0

    # print(latent_vec, latent_vec.size())
    while head < num_samples:

        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()
        sdf_hand, sdf_obj, predicted_class = utils.utils.decode_sdf_multi_output(decoder, latent_vec, sample_subset)
        samples[head : min(head + max_batch, num_samples), 3] = sdf_hand.squeeze(1).detach().cpu()
        samples[head : min(head + max_batch, num_samples), 4] = sdf_obj.squeeze(1).detach().cpu()
        samples[head : min(head + max_batch, num_samples), 5] = predicted_class.argmax(dim=1).detach().cpu()
        head += max_batch

    if hand_branch:
        sdf_values_hand = samples[:, 3]
        sdf_values_hand = sdf_values_hand.reshape(N, N, N)
    else:
        sdf_values_hand = None
    
    if obj_branch:
        sdf_values_obj = samples[:, 4]
        sdf_values_obj = sdf_values_obj.reshape(N, N, N)
    else:
        sdf_values_obj = None

    ###### high resolution ######
    new_voxel_size, new_origin = get_higher_res_cube(
        hand_branch, obj_branch, sdf_values_hand, sdf_values_obj, N, voxel_origin, voxel_size
    )
    
    samples_hr = torch.zeros(N ** 3, 6)

    # transform first 3 columns
    # to be the x, y, z index
    samples_hr[:, 2] = overall_index % N
    samples_hr[:, 1] = (overall_index.long() / N) % N
    samples_hr[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples_hr[:, 0] = (samples_hr[:, 0] * new_voxel_size) + new_origin[0]
    samples_hr[:, 1] = (samples_hr[:, 1] * new_voxel_size) + new_origin[1]
    samples_hr[:, 2] = (samples_hr[:, 2] * new_voxel_size) + new_origin[2]

    samples_hr.requires_grad = False

    head = 0
    while head < num_samples:

        sample_subset = samples_hr[head : min(head + max_batch, num_samples), 0:3].cuda()
        sdf_hand, sdf_obj, predicted_class = utils.utils.decode_sdf_multi_output(decoder, latent_vec, sample_subset)
        samples_hr[head : min(head + max_batch, num_samples), 3] = sdf_hand.squeeze(1).detach().cpu()
        samples_hr[head : min(head + max_batch, num_samples), 4] = sdf_obj.squeeze(1).detach().cpu()
        samples_hr[head : min(head + max_batch, num_samples), 5] = predicted_class.argmax(dim=1).detach().cpu()
        head += max_batch

    sdf_values_hand = samples_hr[:, 3]
    sdf_values_hand = sdf_values_hand.reshape(N, N, N)
    sdf_values_obj = samples_hr[:, 4]
    sdf_values_obj = sdf_values_obj.reshape(N, N, N)

    voxel_size = new_voxel_size
    voxel_origin = new_origin.tolist()

    if hand_branch:
        vertices, mesh_faces = convert_sdf_samples_to_ply(
            sdf_values_hand.data.cpu(),
            voxel_origin,
            voxel_size,
            ply_filename_hand + ".ply",
            offset,
            scale,
        )

        if label_out and vertices is not None:
            vertices[:, 0] = voxel_origin[0] + vertices[:, 0]
            vertices[:, 1] = voxel_origin[1] + vertices[:, 1]
            vertices[:, 2] = voxel_origin[2] + vertices[:, 2]
        
            num_out_vertices = vertices.shape[0]
            # print("output vertices count:", num_out_vertices)
            vertices = torch.from_numpy(vertices)
            vertices.requires_grad = False
            out_labels = torch.zeros(num_out_vertices)

            head = 0
            while head < num_out_vertices:
                sample_subset = vertices[head: min(head + max_batch, num_out_vertices), 0:3].cuda()
                sdf_hand, sdf_obj, predicted_class = utils.utils.decode_sdf_multi_output(decoder, latent_vec, sample_subset)
                out_labels[head: min(head + max_batch, num_out_vertices)] = predicted_class.argmax(dim=1).detach().cpu()
                head += max_batch
            
            if viz:
                write_verts_label_to_obj(
                    vertices,
                    out_labels,
                    ply_filename_hand + "_label.obj",
                    offset,
                    scale
                )

            write_verts_label_to_npz(
                vertices,
                out_labels,
                ply_filename_hand + "_label.npz",
                offset,
                scale
            )

            write_color_labeled_ply(
                vertices,
                mesh_faces,
                out_labels,
                ply_filename_hand+".ply",
                offset,
                scale
            )

    if obj_branch:
        convert_sdf_samples_to_ply(
            sdf_values_obj.data.cpu(),
            voxel_origin,
            voxel_size,
            ply_filename_obj + ".ply",
            offset,
            scale,
        )


def get_higher_res_cube(
    hand_branch,
    obj_branch,
    sdf_values_hand,
    sdf_values_obj,
    N,
    voxel_origin,
    voxel_size
):
    # print("voxel_size", voxel_size)
    if hand_branch:
        indices = torch.nonzero(sdf_values_hand < 0).float()
        if indices.shape[0] == 0:
            min_hand = torch.Tensor([0., 0., 0.])
            max_hand = torch.Tensor([0., 0., 0.])
        else:
            x_min_hand = torch.min(indices[:,0])
            y_min_hand = torch.min(indices[:,1])
            z_min_hand = torch.min(indices[:,2])
            min_hand = torch.Tensor([x_min_hand, y_min_hand, z_min_hand])

            x_max_hand = torch.max(indices[:,0])
            y_max_hand = torch.max(indices[:,1])
            z_max_hand = torch.max(indices[:,2])
            max_hand = torch.Tensor([x_max_hand, y_max_hand, z_max_hand])

    if obj_branch:
        indices = torch.nonzero(sdf_values_obj < 0).float()
        if indices.shape[0] == 0:
            min_obj = torch.Tensor([0., 0., 0.])
            max_obj = torch.Tensor([0., 0., 0.])
        else:
            x_min_obj = torch.min(indices[:,0])
            y_min_obj = torch.min(indices[:,1])
            z_min_obj = torch.min(indices[:,2])
            min_obj = torch.Tensor([x_min_obj, y_min_obj, z_min_obj])

            x_max_obj = torch.max(indices[:,0])
            y_max_obj = torch.max(indices[:,1])
            z_max_obj = torch.max(indices[:,2])
            max_obj = torch.Tensor([x_max_obj, y_max_obj, z_max_obj])

    if not obj_branch:
        min_index = min_hand
        max_index = max_hand
    elif not hand_branch:
        min_index = min_obj
        max_index = max_obj
    else:
        min_index = torch.min(min_hand, min_obj)
        max_index = torch.max(max_hand, max_obj)

    # Buffer 2 voxels each side
    new_cube_size = (torch.max(max_index - min_index) + 4) * voxel_size
    # print("New cube size", new_cube_size)

    new_voxel_size = new_cube_size / (N-1)
    # print("new voxel size", new_voxel_size)
    # [z,y,x]
    new_origin = (min_index - 2 ) * voxel_size - 1.0  # (-1,-1,-1) origin
    # print("new origin", new_origin)
    return new_voxel_size, new_origin


def write_verts_label_to_obj(
    pytorch_3d_xyz_tensor,
    pytorch_label_tensor,
    obj_filename_out,
    offset=None,
    scale=None,
):
    mesh_points = pytorch_3d_xyz_tensor.data.numpy()
    numpy_label_tensor = pytorch_label_tensor.numpy()

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset
   
    with open(obj_filename_out, 'w') as fp:
        for idx, v in enumerate(mesh_points):
            clr = numpy_label_tensor[idx] * 45.0
            fp.write('v %.4f %.4f %.4f %.2f %.2f %.2f\n' % ( v[0], v[1], v[2], clr, clr, clr) )


def write_verts_label_to_npz(
    pytorch_3d_xyz_tensor,
    pytorch_label_tensor,
    npz_filename_out,
    offset=None,
    scale=None,
):
    mesh_points = pytorch_3d_xyz_tensor.data.numpy()
    numpy_label_tensor = pytorch_label_tensor.numpy()

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset
    
    # print(numpy_label_tensor)
    # print(numpy_label_tensor.sum())
   
    # with open(npz_filename_out, 'w') as fp:
    np.savez(npz_filename_out, points=mesh_points, labels=numpy_label_tensor)


def write_color_labeled_ply(
    pytorch_3d_xyz_tensor,
    numpy_faces,
    pytorch_label_tensor,
    ply_filename_out,
    offset=None,
    scale=None,
):
    mesh_points = pytorch_3d_xyz_tensor.data.numpy()
    numpy_label_tensor = pytorch_label_tensor.numpy()

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset
    
    part_color = np.array([[ 13, 212, 128],
       [250,  70,  42],
       [131,  66,  37],
       [ 78, 137,  54],
       [187, 246, 163],
       [ 67, 220,  74]]).astype(np.uint8)

    vertex_color = np.ones((mesh_points.shape[0],3), dtype=np.uint8)
    vertex_color[:,0:3] = part_color[numpy_label_tensor.astype(np.int32), :]

    customized_export_ply(outfile_name = ply_filename_out, v = mesh_points, f = numpy_faces, v_c = vertex_color)

    return

def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply
    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to
    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    try:
        verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
            numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
            # numpy_3d_sdf_tensor, level=-0.005, spacing=[voxel_size] * 3
        )
    except Exception as e:
        logging.warning(
            "Cannot reconstruct mesh from '{}'".format(ply_filename_out)
        )
        print(e)
        return None
    # verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
    #     numpy_3d_sdf_tensor, level=None, spacing=[voxel_size] * 3
    # )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )

    return verts, faces