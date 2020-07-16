import logging
import numpy as np
import os
import torch
import torch.utils.data
import trimesh


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
            input_mesh = trimesh.load(filename, process = False)
            surface_points = trimesh.sample.sample_surface(input_mesh, self.pc_sample)[0]
            surface_points = torch.from_numpy(surface_points * global_scale).float()
        else:
            surface_points = torch.from_numpy(load_points(filename)).float()
        # print(surface_points)
        
        return surface_points, idx, self.input_files[idx]


def load_points(filename):
    points = []
    # print(filename)
    with open( filename, 'r') as fp:
        for line in fp:
            point = line.strip().split(" ")[1:]
            point = np.asarray(point)
            point = point.astype(float)
            points.append(point)
    return np.asarray(points)
