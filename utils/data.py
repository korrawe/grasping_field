import logging
import numpy as np
import os
import torch
import torch.utils.data
import trimesh
from torchvision import transforms
from skimage import io, transform, color

import utils.misc as misc_utils


class PointCloudInput(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        data_list,
        sample_surface=False,
        pc_sample=500,
        verbose=0,
        model_type="1encoder1decoder",
    ):
        self.data_source = data_source
        # self.imagefiles = get_images_filenames(image_source, split, fhb=fhb)

        self.sample_surface = sample_surface
        self.pc_sample = pc_sample

        self.input_files = data_list

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):

        filename = os.path.join(self.data_source, self.input_files[idx])
        # print("load mesh", filename)

        if self.sample_surface:
            global_scale = 5.0
            input_mesh = trimesh.load(filename, process=False)
            surface_points = trimesh.sample.sample_surface(input_mesh, self.pc_sample)[0]
            surface_points = torch.from_numpy(surface_points * global_scale).float()
        else:
            surface_points = torch.from_numpy(load_points(filename)).float()
        # print(surface_points)

        return surface_points, idx, self.input_files[idx]


def load_points(filename):
    points = []
    # print(filename)
    with open(filename, 'r') as fp:
        for line in fp:
            point = line.strip().split(" ")[1:]
            point = np.asarray(point)
            point = point.astype(float)
            points.append(point)
    return np.asarray(points)


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


def romove_higher_dist(tensor, dist, both_col=False):
    keep = torch.abs(tensor[:, 3]) < abs(dist)
    return tensor[keep, :3]


def filter_invalid_sdf(tensor, lab_tensor, dist):
    keep = (torch.abs(tensor[:, 3]) < abs(dist)) & (torch.abs(tensor[:, 4]) < abs(dist))
    # print(keep[:15])
    return tensor[keep, :], lab_tensor[keep, :]


def unpack_normal_params(filename):
    npz = np.load(filename)
    scale = torch.from_numpy(npz["scale"])
    offset = torch.from_numpy(npz["offset"])
    return scale, offset


def unpack_sdf_samples(filename, subsample=None, hand=True, clamp=None, filter_dist=False):
    npz = np.load(filename)
    if subsample is None:
        return npz
    try:
        pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
        neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
        pos_sdf_other = torch.from_numpy(npz["pos_other"])
        neg_sdf_other = torch.from_numpy(npz["neg_other"])
        if hand:
            lab_pos_tensor = torch.from_numpy(npz["lab_pos"])
            lab_neg_tensor = torch.from_numpy(npz["lab_neg"])
        else:
            lab_pos_tensor = torch.from_numpy(npz["lab_pos_other"])
            lab_neg_tensor = torch.from_numpy(npz["lab_neg_other"])
    except Exception as e:
        print("fail to load {}, {}".format(filename, e))
    ### make it (x,y,z,sdf_to_hand,sdf_to_obj)
    if hand:
        pos_tensor = torch.cat([pos_tensor, pos_sdf_other], 1)
        neg_tensor = torch.cat([neg_tensor, neg_sdf_other], 1)
    else:
        xyz_pos = pos_tensor[:, :3]
        sdf_pos = pos_tensor[:, 3].unsqueeze(1)
        pos_tensor = torch.cat([xyz_pos, pos_sdf_other, sdf_pos], 1)

        xyz_neg = neg_tensor[:, :3]
        sdf_neg = neg_tensor[:, 3].unsqueeze(1)
        neg_tensor = torch.cat([xyz_neg, neg_sdf_other, sdf_neg], 1)

    # split the sample into half
    half = int(subsample / 2)

    if filter_dist:
        pos_tensor, lab_pos_tensor = filter_invalid_sdf(pos_tensor, lab_pos_tensor, 2.0)
        neg_tensor, lab_neg_tensor = filter_invalid_sdf(neg_tensor, lab_neg_tensor, 2.0)

    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    # label
    sample_pos_lab = torch.index_select(lab_pos_tensor, 0, random_pos)
    sample_neg_lab = torch.index_select(lab_neg_tensor, 0, random_neg)

    # hand part label
    # 0-finger corase, 1-finger fine, 2-contact, 3-sealed wrist
    hand_part_pos = sample_pos_lab[:, 0]
    hand_part_neg = sample_neg_lab[:, 0]
    samples = torch.cat([sample_pos, sample_neg], 0)
    labels = torch.cat([hand_part_pos, hand_part_neg], 0)

    if clamp:
        labels[samples[:, 3] < -clamp] = -1
        labels[samples[:, 3] > clamp] = -1
        # print(labels)

    if not hand:
        labels[:] = -1
    return samples, labels


def get_instance_filenames(data_source, input_type, encoder_input_source, split, check_file=True, fhb=False, dataset_name="obman"):
    unusable_count = 0
    npzfiles_hand = []
    npzfiles_obj = []
    normalization_params = []
    # imagefiles = []
    encoder_input_files = []
    for dataset in split:
        for class_name in split[dataset]:
            # Hand
            hand_instance_filename = os.path.join(dataset, class_name, "hand.npz")
            if check_file and not os.path.isfile(
                os.path.join(data_source, misc_utils.sdf_samples_subdir, hand_instance_filename)
            ):
                logging.warning(
                    "Requested non-existent hand file '{}'".format(hand_instance_filename)
                )
                unusable_count += 1
                continue

            # Object
            obj_instance_filename = os.path.join(dataset, class_name, "obj.npz")
            if check_file and not os.path.isfile(
                os.path.join(data_source, misc_utils.sdf_samples_subdir, obj_instance_filename)
            ):
                logging.warning(
                    "Requested non-existent object file '{}'".format(obj_instance_filename)
                )
                unusable_count += 1
                continue

            # Offset and scale
            normalization_params_filename = os.path.join(dataset, class_name, "obj.npz")
            if check_file and not os.path.isfile(
                os.path.join(data_source, misc_utils.normalization_param_subdir, normalization_params_filename)
            ):
                logging.warning(
                    "Requested non-existent normalization params file '{}'".format(
                        normalization_params_filename)
                )
                unusable_count += 1
                continue

            if input_type == 'point_cloud':
                encoder_input_files = []
                pass

            npzfiles_hand += [hand_instance_filename]
            npzfiles_obj += [obj_instance_filename]
            normalization_params += [normalization_params_filename]

    logging.warning(
        "Non-existent file count: {} out of {}".format(unusable_count, len(npzfiles_hand))
        )
    return npzfiles_hand, npzfiles_obj, normalization_params, encoder_input_files


def get_negative_surface_points(filename=None, pc_sample=500, surface_dist=0.005, data=None):

    if filename:
        npz = np.load(filename)
    else:
        npz = data
    try:
        neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
    except Exception as e:
        print("fail to load {}, {}".format(filename, e))

    neg_tensor = romove_higher_dist(neg_tensor, surface_dist)

    random_neg = (torch.rand(pc_sample) * neg_tensor.shape[0]).long()
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    return sample_neg


class SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        input_type,
        data_source,
        split,
        subsample,
        dataset_name="obman",
        image_source=None,
        hand_branch=True,
        obj_branch=True,
        indep_obj_scale=False,
        same_point=True,
        filter_dist=False,
        image_size=224,
        load_ram=False,
        print_filename=False,
        num_files=1000000,
        clamp=None,
        pc_sample=500,
        check_file=True,
        fhb=False,
        model_type="1encoder1decoder",
        obj_center=False
    ):
        self.input_type = input_type
        self.subsample = subsample

        self.dataset_name = dataset_name
        self.data_source = data_source
        self.image_source = image_source

        self.hand_branch = hand_branch
        self.obj_branch = obj_branch

        self.pc_sample = pc_sample

        self.filter_dist = filter_dist
        self.model_type = model_type
        self.obj_center = obj_center

        if image_source:
            self.encoder_input_source = image_source
        else: 
            self.encoder_input_source = None
        (self.npyfiles_hand, 
         self.npyfiles_obj,
         self.normalization_params, 
         self.encoder_input_files) = get_instance_filenames(data_source, input_type, self.encoder_input_source,
                                                            split, check_file=check_file, fhb=fhb, dataset_name=dataset_name)

        self.indep_obj_scale = indep_obj_scale
        self.same_point = same_point
        self.clamp = clamp

        logging.debug(
            "using "
            + str(len(self.npyfiles_hand))
            + " shapes from data source "
            + data_source
        )

        self.load_ram = load_ram
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            transforms.RandomAffine(0, translate=(0.1,0.1)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.npyfiles_hand)

    def __getitem__(self, idx):
        filename_hand = os.path.join(
            self.data_source, misc_utils.sdf_samples_subdir, self.npyfiles_hand[idx]
        )
        filename_obj = os.path.join(
            self.data_source, misc_utils.sdf_samples_subdir, self.npyfiles_obj[idx]
        )

        norm_params_filename = os.path.join(
            self.data_source, misc_utils.normalization_param_subdir, self.normalization_params[idx]
        )

        if not self.load_ram:
            if 'image' in self.input_type:
                pass
            else:
                encoder_input_hand = get_negative_surface_points(filename_hand, self.pc_sample)
                encoder_input_obj =  get_negative_surface_points(filename_obj, self.pc_sample)
            scale, offset = unpack_normal_params(norm_params_filename)
            
            # If only hand branch or obj branch is used, subsample is not reduced by half
            # to maintain the same number of samples used when trained with two branches.
            if not self.same_point or not (self.hand_branch and self.obj_branch):
                num_sample = self.subsample
            else:
                num_sample = int(self.subsample / 2)

            hand_samples, hand_labels = unpack_sdf_samples(filename_hand, num_sample, hand=True, clamp=self.clamp, filter_dist=self.filter_dist)
            obj_samples, obj_labels = unpack_sdf_samples(filename_obj, num_sample, hand=False, clamp=self.clamp, filter_dist=self.filter_dist)

            if not self.indep_obj_scale:
                # Scale object back to the hand coordinate 
                obj_samples[:, 0:3] = obj_samples[:, 0:3] / scale - offset
                # Scale sdf back to original scale
                obj_samples[:, 3] = obj_samples[:, 3] / scale
                hand_samples[:, 4] = hand_samples[:, 4] / scale

                #### scale to fit unit sphere -> rescale when reconstruction
                obj_samples[:, 0:5] = obj_samples[:, 0:5] / 2.0
                hand_samples[:, 0:5] = hand_samples[:, 0:5] / 2.0

                if self.input_type == 'point_cloud':
                    encoder_input_obj = encoder_input_obj / scale - offset
            
            if 'VAE' in self.model_type and self.obj_center:
                # normalize point cloud
                (encoder_input_hand, encoder_input_obj,
                hand_samples, obj_samples) = normalize_obj_center(encoder_input_hand, encoder_input_obj,
                                                hand_samples, obj_samples, scale=2.0)

            return (hand_samples, hand_labels, obj_samples, obj_labels,
                    scale, offset, encoder_input_hand, encoder_input_obj, self.npyfiles_hand[idx])


def normalize_obj_center(encoder_input_hand, encoder_input_obj, hand_samples=None, obj_samples=None, scale=1.0):
    object_center = encoder_input_obj.mean(dim=0)
    encoder_input_hand = encoder_input_hand - object_center
    encoder_input_obj = encoder_input_obj - object_center
    if (hand_samples is not None) and (obj_samples is not None):
        hand_samples[:,:3] = hand_samples[:,:3] - object_center / scale
        obj_samples[:,:3] = obj_samples[:,:3] - object_center / scale
        return encoder_input_hand, encoder_input_obj, hand_samples, obj_samples
    return encoder_input_hand, encoder_input_obj


class PointCloudsSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        load_ram=False,
        indep_obj_scale=False,
        pc_sample=500,
        print_filename=False,
        fhb=False,
        model_type="1encoder1decoder",
        obj_center=False,
        dataset_name='obman'
    ):
        self.data_source = data_source
        # self.imagefiles = get_images_filenames(image_source, split, fhb=fhb)

        self.pc_sample = pc_sample
        self.model_type = model_type
        self.obj_center = obj_center

        (self.npyfiles_hand, 
         self.npyfiles_obj,
         self.normalization_params, 
         _) = get_instance_filenames(data_source, 'point_cloud', None, split, check_file=False, fhb=fhb)

        self.indep_obj_scale = indep_obj_scale

        logging.debug(
            "using "
            + str(len(self.npyfiles_hand))
            + " shapes from data source "
            + data_source
        )

        self.load_ram = load_ram

    def __len__(self):
        return len(self.npyfiles_hand)

    def __getitem__(self, idx):
        # image_filename = os.path.join(self.image_source, self.imagefiles[idx])
        filename_hand = os.path.join(
            self.data_source, misc_utils.sdf_samples_subdir, self.npyfiles_hand[idx]
        )
        filename_obj = os.path.join(
            self.data_source, misc_utils.sdf_samples_subdir, self.npyfiles_obj[idx]
        )
        norm_params_filename = os.path.join(
            self.data_source, misc_utils.normalization_param_subdir, self.normalization_params[idx]
        )

        if self.load_ram:
            return 0
            # return (
            #     unpack_sdf_samples_from_ram(self.loaded_data[idx], self.subsample),
            #     idx,
            # )
        else:
            # pc_sample = 500
            encoder_input_hand = get_negative_surface_points(filename_hand, self.pc_sample)
            encoder_input_obj =  get_negative_surface_points(filename_obj, self.pc_sample)

            scale, offset = unpack_normal_params(norm_params_filename)

            # print(obj_samples[0:2])
            if not self.indep_obj_scale:
                encoder_input_obj = encoder_input_obj / scale - offset
            
            if 'VAE' in self.model_type and self.obj_center:
                encoder_input_hand, encoder_input_obj = normalize_obj_center(encoder_input_hand, encoder_input_obj)
                # print("object center!!!")
            return encoder_input_hand, encoder_input_obj, idx, self.npyfiles_hand[idx]

            # image = io.imread(image_filename)
            # image = self.transform(image)
            # return image , idx, self.imagefiles[idx]


class ImagesInput(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        data_list,
        image_size=224
    ):
        self.data_source = data_source
        self.input_files = data_list

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        filename = os.path.join(self.data_source, self.input_files[idx])
        image = io.imread(filename)
        if image.shape[2] == 4:
            image = image[:, :, :3]
        image = self.transform(image)
        return image, idx, self.input_files[idx]
