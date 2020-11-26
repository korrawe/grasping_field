#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import concurrent.futures
import json
import logging
import os
import subprocess

import deep_sdf
import deep_sdf.workspace as ws

##
import shutil
##


def filter_classes_glob(patterns, classes):
    import fnmatch

    passed_classes = set()
    for pattern in patterns:

        passed_classes = passed_classes.union(
            set(filter(lambda x: fnmatch.fnmatch(x, pattern), classes))
        )

    return list(passed_classes)


def filter_classes_regex(patterns, classes):
    import re

    passed_classes = set()
    for pattern in patterns:
        regex = re.compile(pattern)
        passed_classes = passed_classes.union(
            set(filter(regex.match, classes))
        )

    return list(passed_classes)


def filter_classes(patterns, classes):
    if patterns[0] == "glob":
        return filter_classes_glob(patterns, classes[1:])
    elif patterns[0] == "regex":
        return filter_classes_regex(patterns, classes[1:])
    else:
        return filter_classes_glob(patterns, classes)


def process_mesh(mesh_filepath_hand,
                 mesh_filepath_obj, 
                 target_filepath_hand,
                 target_filepath_obj, 
                 executable,
                 additional_args):
    logging.info(mesh_filepath_hand + " --> " + target_filepath_hand + " and obj")
    command = [
        executable,
        "--hand",
        mesh_filepath_hand,
        "--obj",
        mesh_filepath_obj,
        "--outhand",
        target_filepath_hand,
        "--outobj",
        target_filepath_obj,
    ] + additional_args

    # subproc = subprocess.Popen(command, stdout=subprocess.DEVNULL) subprocess.PIPE
    ####
    subproc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = subproc.communicate()
    ### for debugging ### 
    # print(target_filepath_hand)
    # # print("out:", out)
    # print("out:")
    # print(out.decode("utf-8").replace('\\n', '\n'))
    # if err is not None:
    #     print("err:")
    #     print(err.decode("utf-8").replace('\\n', '\n'))
    ### end for debugging ###
    

    mesh_out_dir = os.path.dirname(target_filepath_hand)
    shape_num = os.path.basename(os.path.normpath(mesh_out_dir))


    if "mesh rejected object" in out.decode("utf-8"):
        with open(os.path.join(os.path.dirname(mesh_out_dir) , "invalid", shape_num + '.reject'), 'w') as f:
            f.write(out.decode("utf-8").replace('\\n', '\n'))
    elif "success object" not in out.decode("utf-8"):
        with open(os.path.join(os.path.dirname(mesh_out_dir) , "invalid", shape_num + '.fail'), 'w') as f:
            f.write(out.decode("utf-8").replace('\\n', '\n'))
    #####
    # subproc.wait()


def process_mesh_surface(mesh_filepath,
                         target_filepath,
                         executable,):
    logging.info(mesh_filepath + " --> " + target_filepath)
    command = [
        executable,
        "-m",
        mesh_filepath,
        "-o",
        target_filepath,
    ]

    # subproc = subprocess.Popen(command, stdout=subprocess.DEVNULL) subprocess.PIPE
    ####
    subproc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = subproc.communicate()
    # ### for debugging ### 
    # print(target_filepath)
    # # # print("out:", out)
    # print("out:")
    # print(out.decode("utf-8").replace('\\n', '\n'))
    # if err is not None:
    #     print("err:")
    #     print(err.decode("utf-8").replace('\\n', '\n'))
    # ### end for debugging ###
    

    # mesh_out_dir = os.path.dirname(target_filepath_hand)
    # shape_num = os.path.basename(os.path.normpath(mesh_out_dir))


    # if "mesh rejected object" in out.decode("utf-8"):
    #     with open(os.path.join(os.path.dirname(mesh_out_dir) , "invalid", shape_num + '.reject'), 'w') as f:
    #         f.write(out.decode("utf-8").replace('\\n', '\n'))
    # elif "success object" not in out.decode("utf-8"):
    #     with open(os.path.join(os.path.dirname(mesh_out_dir) , "invalid", shape_num + '.fail'), 'w') as f:
    #         f.write(out.decode("utf-8").replace('\\n', '\n'))
    #####
    # subproc.wait()

def append_data_source_map(data_dir, name, source):

    data_source_map_filename = ws.get_data_source_map_filename(data_dir)

    print("data sources stored to " + data_source_map_filename)

    data_source_map = {}

    if os.path.isfile(data_source_map_filename):
        with open(data_source_map_filename, "r") as f:
            print(data_source_map_filename)
            data_source_map = json.load(f)

    if name in data_source_map:
        # print(data_source_map[name], os.path.abspath(source))
        if not data_source_map[name] == os.path.abspath(source):
            raise RuntimeError(
                "Cannot add data with the same name and a different source."
            )

    else:
        data_source_map[name] = os.path.abspath(source)

        with open(data_source_map_filename, "w") as f:
            json.dump(data_source_map, f, indent=2)


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Pre-processes data from a data source and append the results to a dataset.",
    )
    arg_parser.add_argument(
        "--data_dir",
        "-d",
        dest="data_dir",
        required=True,
        help="The directory which holds all preprocessed data.",
    )
    arg_parser.add_argument(
        "--source",
        "-s",
        dest="source_dir",
        required=True,
        help="The directory which holds the data to preprocess and append.",
    )
    arg_parser.add_argument(
        "--name",
        "-n",
        dest="source_name",
        default=None,
        help="The name to use for the data source. If unspecified, it defaults to the directory name.",
    )
    arg_parser.add_argument(
        "--split",
        dest="split_filename",
        required=True,
        help="A split filename defining the shapes to be processed.",
    )
    arg_parser.add_argument(
        "--skip",
        dest="skip",
        default=False,
        action="store_true",
        help="If set, previously-processed shapes will be skipped",
    )
    arg_parser.add_argument(
        "--threads",
        dest="num_threads",
        default=8,
        help="The number of threads to use to process the data.",
    )
    arg_parser.add_argument(
        "--test",
        "-t",
        dest="test_sampling",
        default=False,
        action="store_true",
        help="If set, the script will produce SDF samplies for testing",
    )
    arg_parser.add_argument(
        "--surface",
        dest="surface_sampling",
        default=False,
        action="store_true",
        help="If set, the script will produce mesh surface samples for evaluation. "
        + "Otherwise, the script will produce SDF samples for training.",
    )

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    additional_general_args = []

    if args.surface_sampling:
        executable = "bin/SampleVisibleMeshSurface"
        subdir = ws.surface_samples_subdir
        extension = ".ply"
    else:
        executable = "bin/PreprocessMesh"
        subdir = ws.sdf_samples_subdir
        extension = ".npz"

        if args.test_sampling:
            additional_general_args += ["-t"]

    with open(args.split_filename, "r") as f:
        split = json.load(f)

    if args.source_name is None:
        args.source_name = os.path.basename(os.path.normpath(args.source_dir))

    dest_dir = os.path.join(args.data_dir, subdir, args.source_name)

    logging.info(
        "Preprocessing data from "
        + args.source_dir
        + " and placing the results in "
        + dest_dir
    )

    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)

    ### 
    if not os.path.isdir(os.path.join(dest_dir, "invalid")):
        os.makedirs(os.path.join(dest_dir, "invalid"))
    ### 
    print("dest:", dest_dir)
    print("dest invalid:", os.path.join(dest_dir, "invalid"))

    # if args.surface_sampling: # tap <-1
    normalization_param_dir = os.path.join(
        args.data_dir, ws.normalization_param_subdir, args.source_name
    )
    if not os.path.isdir(normalization_param_dir):
        os.makedirs(normalization_param_dir)
    print("-Normalized param", normalization_param_dir)

    append_data_source_map(args.data_dir, args.source_name, args.source_dir)

    print(">>", args.source_name)
    class_directories = split[args.source_name]

    meshes_targets_and_specific_args = []

    count = 0
    for class_dir in class_directories:
        class_path = os.path.join(args.source_dir, class_dir)
        # print(class_dir)
        # print(class_directories)
        # instance_dirs = class_directories[class_dir]

        logging.debug(
            "Processing "
            + "hand and object"
            + " instances of class "
            + class_dir
        )

        target_dir = os.path.join(dest_dir, class_dir)

        # create directory
        if not os.path.isdir(target_dir):
            os.mkdir(target_dir)

        # for instance_dir in instance_dirs: # <- 1 tab
        instance_dir_hand = "hand"
        instance_dir_obj = "obj"
        # print("iii", instance_dir)
        shape_dir_hand = os.path.join(class_path, instance_dir_hand)
        shape_dir_obj = os.path.join(class_path, instance_dir_obj)

        # print(class_path)
        # print(shape_dir_hand)
        # print(shape_dir_obj)

        processed_filepath_hand = os.path.join(
            target_dir, instance_dir_hand + extension
        )
        processed_filepath_obj = os.path.join(
            target_dir, instance_dir_obj + extension
        )

        # if args.skip and os.path.isfile(processed_filepath_hand):
        #     logging.debug("skipping " + processed_filepath_hand) # + " and " + processed_filepath_obj)
        #     continue

        try:
            # mesh_filename_hand = deep_sdf.data.find_mesh_in_directory(shape_dir_hand)
            # mesh_filename_obj = deep_sdf.data.find_mesh_in_directory(shape_dir_obj)
            # print(shape_dir_hand)
            # print(mesh_filename_hand)
            mesh_filename_hand = os.path.join(shape_dir_hand, "models","model_normalized_sealed.obj") ### 
            mesh_filename_obj = os.path.join(shape_dir_obj, "models","model_normalized.obj")
            # print(mesh_filename_hand)
            # print(mesh_filename_obj)

            specific_args = []

            # if args.surface_sampling: # tab <-1
            # create normalization directory
            normalization_param_target_dir = os.path.join(
                normalization_param_dir, class_dir
            )
            # print(normalization_param_target_dir)

            if not os.path.isdir(normalization_param_target_dir):
                os.mkdir(normalization_param_target_dir)

            normalization_param_filename = os.path.join(
                normalization_param_target_dir, instance_dir_obj + ".npz"
            )
            # specific_args = ["-n", normalization_param_filename] 
            specific_args = ["--normalize", normalization_param_filename]

            # print("dddd", os.path.join(shape_dir_hand, mesh_filename_hand))
            # print("ffff", os.path.join(shape_dir_obj, mesh_filename_obj))
            # print("jjj", os.path.join(shape_dir, mesh_filename))
            
            if args.surface_sampling:
                meshes_targets_and_specific_args.append(
                    (
                        os.path.join(shape_dir_obj, mesh_filename_obj), # obj
                        processed_filepath_obj,
                        # specific_args,
                    )
                )
                meshes_targets_and_specific_args.append(
                    (
                        os.path.join(shape_dir_hand, mesh_filename_hand), # hand
                        processed_filepath_hand,
                        # specific_args,
                    )
                )
            else:
                meshes_targets_and_specific_args.append(
                    (
                        os.path.join(shape_dir_hand, mesh_filename_hand), # hand 
                        os.path.join(shape_dir_obj, mesh_filename_obj), # obj
                        processed_filepath_hand,
                        processed_filepath_obj,
                        specific_args,
                    )
                )

            # for copy
            # shutil.copyfile(processed_filepath_hand, dest)
            # copy_prefix = "/is/cluster/work/kkarunratanakul/obman/data_new/SdfSamples/train/"
            # hand_dest = os.path.join(copy_prefix, class_dir, "hand.npz")
            # obj_dest = os.path.join(copy_prefix, class_dir, "obj.npz")
            # # print(processed_filepath_hand)
            # # # print(hand_dest)
            # # print(processed_filepath_obj)
            # # print(obj_dest)
            # shutil.copyfile(processed_filepath_hand, hand_dest)
            # shutil.copyfile(processed_filepath_obj, obj_dest)
            # break

        except deep_sdf.data.NoMeshFileError:
            logging.warning("No mesh found for instance " + instance_dir_hand)
        except deep_sdf.data.MultipleMeshFileError:
            logging.warning(
                "Multiple meshes found for instance " + instance_dir_hand
            )
        # except Exception as e:
        #     print(e)

        # if count % 100 == 0:
        #     print(count)
        # print(count)
        # count += 1
        

    print(" Start sampling")
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=int(args.num_threads)
    ) as executor:
        if args.surface_sampling:
            for (
                mesh_filepath,
                target_filepath,
            ) in meshes_targets_and_specific_args:
                # print(mesh_filepath)
                executor.submit(
                    process_mesh_surface,
                    mesh_filepath,
                    target_filepath,
                    executable,
                )
        else:
            for (
                # mesh_filepath,
                mesh_filepath_hand, ###
                mesh_filepath_obj,
                target_filepath_hand,
                target_filepath_obj,
                specific_args,
            ) in meshes_targets_and_specific_args:
                break
                # process_mesh(
                #     mesh_filepath_hand, ###
                #     mesh_filepath_obj, ###
                #     target_filepath_hand, ###
                #     target_filepath_obj, ###
                #     executable,
                #     specific_args + additional_general_args)
                executor.submit(
                    process_mesh,
                    mesh_filepath_hand, ###
                    mesh_filepath_obj, ###
                    target_filepath_hand, ###
                    target_filepath_obj, ###
                    executable,
                    specific_args + additional_general_args,
                )

        executor.shutdown()
