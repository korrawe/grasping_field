import json
import os
import torch

import pickle
import numpy as np

sdf_samples_subdir = "SdfSamples"
surface_samples_subdir = "SurfaceSamples"
model_params_subdir = "ModelParameters"
normalization_param_subdir = "NormalizationParameters"
optimizer_params_subdir = "OptimizerParameters"
reconstructions_subdir = "Reconstructions"
reconstruction_meshes_subdir = "Meshes"
evaluation_subdir = "Evaluation"
specifications_filename = "specs.json"
logs_filename = "Logs.pth"


def load_experiment_specifications(experiment_directory):

    filename = os.path.join(experiment_directory, specifications_filename)

    if not os.path.isfile(filename):
        raise Exception(
            "The experiment directory ({}) does not include specifications file "
            + '"specs.json"'.format(experiment_directory)
        )

    return json.load(open(filename))


def is_checkpoint_exist(experiment_directory, checkpoint):

    filename = os.path.join(
        experiment_directory, model_params_subdir, checkpoint + ".pth"
    )

    return os.path.isfile(filename)


def load_model_parameters(experiment_directory, checkpoint, model):

    filename = os.path.join(
        experiment_directory, model_params_subdir, checkpoint + ".pth"
    )

    if not os.path.isfile(filename):
        raise Exception('model state dict "{}" does not exist'.format(filename))

    data = torch.load(filename)

    model.load_state_dict(data["model_state_dict"], strict=False)

    return data["epoch"]


def get_reconstructed_mesh_filename(
    experiment_dir, epoch, dataset, class_name, instance_name
):

    return os.path.join(
        experiment_dir,
        reconstructions_subdir,
        str(epoch),
        reconstruction_meshes_subdir,
        dataset,
        class_name + "_" + instance_name + ".ply",
    )


def get_reconstructed_mesh_label_filename(
    experiment_dir, epoch, dataset, class_name, instance_name
):
    return os.path.join(
        experiment_dir,
        reconstructions_subdir,
        str(epoch),
        reconstruction_meshes_subdir,
        dataset,
        class_name + "_" + instance_name + "_label.npz",
    )


def get_pred_joints_out_filename(
    experiment_dir, epoch, dataset, class_name, instance_name
):
    return os.path.join(
        experiment_dir,
        reconstructions_subdir,
        str(epoch),
        reconstruction_meshes_subdir,
        dataset,
        class_name + "_" + instance_name + "_joint.npy",
    )


def get_evaluation_dir(experiment_dir, checkpoint, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, evaluation_subdir, checkpoint)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_obman_evaluation_dir(experiment_dir, checkpoint, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, evaluation_subdir, checkpoint)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_model_params_dir(experiment_dir, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, model_params_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_optimizer_params_dir(experiment_dir, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, optimizer_params_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_normalization_params_filename(
    data_dir, dataset_name, class_name, instance_name
):
    return os.path.join(
        data_dir,
        normalization_param_subdir,
        dataset_name,
        class_name,
        # instance_name + ".npz",
        "obj.npz",
    )