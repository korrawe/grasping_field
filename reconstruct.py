import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
import argparse
import torch
import torch.utils.data as data_utils
import signal
import sys
import os
import logging
import numpy as np
import json
import time

import utils
import networks.model as arch
import pcl2mano.pcl2mano as mano_helper
import utils.misc as misc_utils
from networks.model_utils import get_model


def get_input_type(task):
    if task == "sample_grasp":
        return "point_cloud"
    else:
        return "image"

def reconstruct(loaded_model, 
                split_filename,
                input_source,
                output_dir,
                specs,
                model_type='1encoder2decoder',
                task="sample_grasp",
                sample=False,
                n_sample=5,
                mesh_input=False,
                device="cpu",
                scale=None,
                cube_dim=128,  # 256,
                label_out=False,
                viz=False,
                verbose=0):

    # For testing image input
    # task = "rgb_recon_obman"
    
    # output_dir = "./output/reconstruction/"
    # split_filename = "./input/image/obman_test.json"
    # input_source = "./input/image/obman_test"
    # End testing image input
    # input_type = "point_cloud"

    print("Task:", task)
    if task == "sample_grasp" and sample:
        print("Sample hand")

    output_mesh_dir = os.path.join(output_dir, "meshes")
    if not os.path.isdir(output_mesh_dir):
        os.makedirs(output_mesh_dir)

    output_mano_dir = os.path.join(output_dir, "mano")
    if not os.path.isdir(output_mano_dir):
        os.makedirs(output_mano_dir)

    with open(split_filename, "r") as f:
        input_list = json.load(f)["filenames"]

    input_type = get_input_type(task)
    print("input type", input_type)
    if input_type == 'point_cloud':
        dataset = utils.data.PointCloudInput(
            input_source, input_list, sample_surface=mesh_input
        )
    elif input_type == 'image':
        dataset = utils.data.ImagesInput(
            input_source, input_list
        )

    # load data
    num_data_loader_threads = 1
    logging.debug("loading data with {} threads".format(num_data_loader_threads))

    data_loader = data_utils.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_data_loader_threads,
        drop_last=False,
    )

    loaded_model.eval()

    for encoder_input, idx, filename in data_loader:
        print(filename)
        encoder_input = encoder_input.to(device)
        # print(encoder_input_obj)

        # Reconstruction from img
        if task == "rgb_recon_obman":
            with torch.no_grad():
                latent = loaded_model.module.encoder(encoder_input)
                # print(latent)
            out_filename = os.path.splitext(os.path.basename(filename[0]))[0]
            print("* Processing:", out_filename)

            mesh_filename = os.path.join(output_mesh_dir, out_filename)
            if not os.path.exists(os.path.dirname(mesh_filename)):
                os.makedirs(os.path.dirname(mesh_filename))
            # print(mesh_filename)

            obj_branch = True
            hand_branch = True
            with torch.no_grad():
                utils.mesh.create_mesh_combined_decoder(
                    hand_branch, obj_branch,
                    loaded_model.module.decoder, latent, mesh_filename, N=cube_dim, max_batch=int(2 ** 18),  # N = 256
                    scale=scale, device=device, label_out=label_out, viz=viz
                )
            del latent
            # Fit MANO to the label
            if label_out:
                print("...Fitting MANO to mesh...")
                mesh_out_name = os.path.join(output_mano_dir, out_filename + "_hand_mano.ply")
                mano_helper.fit_mano(mesh_filename + "_hand_label.npz", mesh_out_name)
            print("-------------")

        # VAE grasp sampling
        elif task == "sample_grasp":
            start = time.time()
            encoder_input_hand = None
            encoder_input_obj = encoder_input

            # n_sample = 5
            for i in range(n_sample):
                with torch.no_grad():
                    latent = loaded_model.module.compute_latent(encoder_input_hand, encoder_input_obj, sample=sample)

                out_filename = os.path.splitext(os.path.basename(filename[0]))[0] + "_" + str(i)
                print("* Processing:", out_filename)

                mesh_filename = os.path.join(output_mesh_dir, out_filename)
                if not os.path.exists(os.path.dirname(mesh_filename)):
                    os.makedirs(os.path.dirname(mesh_filename))
                # print(mesh_filename)

                obj_branch_tmp = False  # True if i == 0 else
                hand_branch = True
                with torch.no_grad():
                    utils.mesh.create_mesh_combined_decoder(
                        hand_branch, obj_branch_tmp,
                        loaded_model.module.decoder, latent, mesh_filename, N=cube_dim, max_batch=int(2 ** 18),  # N = 256
                        scale=scale, device=device, label_out=label_out, viz=viz
                    )
                del latent
                # Fit MANO to the label
                if label_out:
                    print("...Fitting MANO to mesh...")
                    mesh_out_name = os.path.join(output_mano_dir, out_filename + "_hand_mano.ply")
                    mano_helper.fit_mano(mesh_filename + "_hand_label.npz", mesh_out_name)
                print("-------------")

            end = time.time()
            if verbose:
                print("Overall: {}".format(end - start))
            print("-------------")


def reconstruct_training(experiment_directory, 
                split_filename,
                input_type,
                input_source,
                encoderDecoder,
                saved_model_epoch,
                specs,
                hand_branch,
                obj_branch,
                model_type='1encoder2decoder',
                scale=None,
                cube_dim=128, # 256,
                verbose=0,
                fhb=False,
                dataset_name='ho3d',
                obj_center=False,
                label_out=False,
                sample=False,
                viz=False):
    
    reconstruction_dir = os.path.join(
        experiment_directory, misc_utils.reconstructions_subdir, str(saved_model_epoch)
    )

    if not os.path.isdir(reconstruction_dir):
        os.makedirs(reconstruction_dir)

    reconstruction_meshes_dir = os.path.join(
        reconstruction_dir, misc_utils.reconstruction_meshes_subdir
    )
    if not os.path.isdir(reconstruction_meshes_dir):
        os.makedirs(reconstruction_meshes_dir)
    
    with open(split_filename, "r") as f:
        train_split = json.load(f)
    for name in train_split:
        split_name = name
        break
    
    if verbose:
        print("Split:", split_name)
        print(input_source)
    
    if input_type == 'point_cloud':
        dataset = utils.data.PointCloudsSamples(
            input_source, train_split, load_ram=False, fhb=fhb, model_type=model_type, obj_center=obj_center
        )

    # load data
    num_data_loader_threads = 1 
    logging.debug("loading data with {} threads".format(num_data_loader_threads))

    data_loader = data_utils.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_data_loader_threads,
        drop_last=False,
    )

    reconstruct_limit = 10000

    for encoder_input_hand, encoder_input_obj, idx, image_filename in data_loader:
        if (reconstruct_limit is not None) and idx > reconstruct_limit:
            break

        print(image_filename)

        start = time.time()
        encoderDecoder.eval()
        # encoderDecoder.encoder.eval()
        # print(image.shape)
        encoder_input_hand = encoder_input_hand.cuda()
        if input_type == 'image':
            encoder_input_obj = encoder_input_hand
        else:
            encoder_input_obj = encoder_input_obj.cuda()
        # print(image[0,0,:5,:5])
        # print(image[:3])
        # print(image.size())
        
        # latent = encoderDecoder.encoder(image)
        # npimg = image[0].cpu().numpy()
        # plt.imshow(np.transpose(npimg, (1, 2, 0)))

        if 'VAE' in model_type:
            n_sample = 5
            print("VAE n=", n_sample)
            for i in range(n_sample):
                with torch.no_grad():
                    latent = encoderDecoder.module.compute_latent(encoder_input_hand, encoder_input_obj, sample=sample)
                
                base = os.path.splitext(image_filename[0])
                out_filename = image_filename[0].split('/')[1] + "_" + str(i)
                if dataset_name == 'ho3d':
                    inst = image_filename[0].split('/')
                    seq = inst[1]
                    frame_num = inst[2]
                    out_filename = "_".join([seq, frame_num]) + "_" + str(i)
                print(out_filename)

                mesh_filename = os.path.join(reconstruction_meshes_dir, split_name, out_filename)
                if not os.path.exists(os.path.dirname(mesh_filename)):
                    os.makedirs(os.path.dirname(mesh_filename))
                print(mesh_filename)
                
                obj_branch_tmp = True if i == 0 else False
                with torch.no_grad():
                    utils.mesh.create_mesh_combined_decoder(
                        hand_branch, obj_branch_tmp,
                        encoderDecoder.module.decoder, latent, mesh_filename, N=cube_dim, max_batch=int(2 ** 18), # N= 256
                        scale=scale, 
                        label_out=label_out,
                        viz=viz
                    )
                del latent

            continue
        
        end = time.time()
        if verbose:
            print("Overall: {}".format(end - start))
            print("-------------")


def get_default_args(args):
    if args.task == "rgb_recon_obman":
        args.model_directory = "./pretrained_model/reconstruction/obman"
        args.split_filename = "./input/image/obman_test.json"
        args.input_source = "./input/image/obman_test"
        args.output_dest = "./output/reconstruction/"

    elif args.task == "rgb_recon_fhb":
        pass
    return args


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Use a trained GraspingField model to reconstruct hand shapes given "
        + " object shapes."
    )
    arg_parser.add_argument(
        "--model",
        "-e",
        dest="model_directory",
        default="./pretrained_model",
        help="The experiment directory which includes specifications and pretrained model",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint weights to use. This can be a number indicated an epoch "
        + "or 'latest' for the latest weights (this is the default)",
    )
    arg_parser.add_argument(
        "--input",
        "-i",
        dest="input_source",
        default="./input",
        help="The input source directory.",
    )
    arg_parser.add_argument(
        "--output",
        "-o",
        dest="output_dest",
        default="./output",
        help="The output directory.",
    )
    arg_parser.add_argument(
        "--split",
        "-s",
        dest="split_filename",
        default="input.json",
        help="The json file containin a list of inputs.",
    )
    arg_parser.add_argument(
        '--sample', dest='sample',
        # action='store_true',
        default=True,
        help="Sample hands conditioned on object shape"
    )
    arg_parser.add_argument(
        '--mesh_input', dest='mesh_input',
        # action='store_true',
        default=True,
        help="If true, load the input and do surface sampling. Otherwise, use input as point cloud"
    )
    arg_parser.add_argument(
        '--label', dest='label_out',
        # action='store_true',
        default=True,
        help="If true, output npy files containing hand-part label for each points "
        + "in the output meshes. Required for MANO fitting."
    )
    arg_parser.add_argument(
        '--viz', dest='viz', 
        action='store_true',
        help="If true, output easy-to-visualized obj files containing hand-part labels"
    )
    arg_parser.add_argument(
        "--task",
        "-t",
        dest="task",
        default="sample_grasp",
        choices=["sample_grasp", "rgb_recon_obman", "rgb_recon_fhb"],
        help="task to perform"
    )

    # utils.add_common_args(arg_parser)
    args = arg_parser.parse_args()
    # utils.configure_logging(args)

    # Change defualt args based on task
    args = get_default_args(args)

    specs_filename = os.path.join(args.model_directory, "specs.json")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))

    loaded_model = get_model(args.model_directory, specs, device)

    global_scale = 5.0
    recon_scale = 0.5 * global_scale

    # logging.debug(encoderDecoder)

    reconstruct(loaded_model,
                args.split_filename,
                args.input_source,
                args.output_dest,
                specs,
                model_type=specs["ModelType"],
                task=args.task,
                sample=args.sample,
                mesh_input=args.mesh_input,
                device=device,
                scale=recon_scale,
                cube_dim=128,
                label_out=args.label_out,
                viz=args.viz,)