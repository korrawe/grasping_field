import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import torch
import torch.utils.data as data_utils
import signal
import sys
import os
import logging
import numpy as np
import json
import time

from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms
from torch import distributions as dist

import utils
import networks.model as arch

import reconstruct
# import evaluate

class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):

        return self.initial * (self.factor ** (epoch // self.interval))


class LinearWeightSchedule:
    def __init__(self, start_ep, interval, initial=0.0, target=1.0):
        self.start_ep = start_ep
        self.interval = interval
        self.initial = initial
        self.target = target

    def get_weight(self, epoch):
        if epoch < self.start_ep:
            return self.initial
        return min(self.target, self.initial + (self.target - self.initial) * (epoch - self.start_ep) / self.interval)
    

def get_kl_weight_schedules(specs):

    kl_schedules_specs = specs["KLSchedule"]

    return LinearWeightSchedule(
        kl_schedules_specs["Start"],
        kl_schedules_specs["Interval"],
        0.0,
        get_spec_with_default(kl_schedules_specs, "Target", 0.1)
    )


def get_learning_rate_schedules(specs):

    schedule_specs_list = specs["LearningRateSchedule"]
    schedules = []

    for schedule_specs in schedule_specs_list:

        if schedule_specs["Type"] == "Step":
            schedules.append(
                StepLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Interval"],
                    schedule_specs["Factor"],
                )
            )

        else:
            raise Exception(
                'no known learning rate schedule of type "{}"'.format(
                    schedule_specs["Type"]
                )
            )

    return schedules


def save_model(experiment_directory, filename, model, epoch):

    model_params_dir = utils.misc.get_model_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "model_state_dict": model.state_dict()},
        os.path.join(model_params_dir, filename),
    )


def save_optimizer(experiment_directory, filename, optimizer, epoch):

    optimizer_params_dir = utils.misc.get_optimizer_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(optimizer_params_dir, filename),
    )


def load_optimizer(experiment_directory, filename, optimizer):

    full_filename = os.path.join(
        utils.misc.get_optimizer_params_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception(
            'optimizer state dict "{}" does not exist'.format(full_filename)
        )

    data = torch.load(full_filename)

    optimizer.load_state_dict(data["optimizer_state_dict"])

    return data["epoch"]


def load_logs(experiment_directory):

    full_filename = os.path.join(experiment_directory, utils.misc.logs_filename)

    if not os.path.isfile(full_filename):
        raise Exception('log file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    return (
        data["loss"],
        data["learning_rate"],
        data["timing"],
        data["epoch"],
    )


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


def get_mean_latent_vector_magnitude(latent_vectors):
    host_vectors = np.array(
        [vec.detach().cpu().numpy().squeeze() for vec in latent_vectors]
    )
    return np.mean(np.linalg.norm(host_vectors, axis=1))


def append_parameter_magnitudes(param_mag_log, model, writer, step):
    for name, param in model.named_parameters():
        if len(name) > 7 and name[:7] == "module.":
            name = name[7:]
        if name not in param_mag_log.keys():
            param_mag_log[name] = []
        param_mag_log[name].append(param.data.norm().item())
        writer.add_scalar(name + 'mag', param.data.norm().item(), step)


def main_function(experiment_directory, continue_from, batch_split):

    logging.debug("running " + experiment_directory)

    print(experiment_directory)

    specs = utils.misc.load_experiment_specifications(experiment_directory)

    logging.info("Experiment description: \n" + specs["Description"])

    data_source = specs["DataSource"]
    image_source = specs["ImageSource"]
    train_split_file = specs["TrainSplit"]
    val_split_file = get_spec_with_default(specs, "ValSplit", None)

    is_fhb = get_spec_with_default(specs, "FHB", False)
    if is_fhb:
        print("FHB dataset")

    check_file = get_spec_with_default(specs,"CheckFile", True)

    logging.debug(specs["NetworkSpecs"])

    dataset_name = get_spec_with_default(specs, "Dataset", "obman")

    ### Model Type
    model_type = get_spec_with_default(specs, "ModelType", "1encoder2decoder")
    obj_center = get_spec_with_default(specs, "ObjectCenter", False)

    hand_branch = get_spec_with_default(specs, "HandBranch", True)
    obj_branch = get_spec_with_default(specs, "ObjectBranch", True)

    print("Hand branch:", hand_branch)
    print("Object branch:", obj_branch)
    assert hand_branch or obj_branch

    classifier_branch = get_spec_with_default(specs, "ClassifierBranch", False)
    classifier_weight = get_spec_with_default(specs, "ClassifierWeight", 0.1)
    print("Classifier Weight:", classifier_weight)

    use_gaussian_reconstruction_weight = get_spec_with_default(specs, "GaussianWeightLoss", False)


    do_penetration_loss = get_spec_with_default(specs, "PenetrationLoss", False)
    penetration_loss_weight = get_spec_with_default(specs, "PenetrationLossWeight", 15.0) # 1000.0)
    start_additional_loss = get_spec_with_default(specs, "AdditionalLossStart", 200000) # 500)
    do_contact_loss = get_spec_with_default(specs, "ContactLoss", False)
    contact_loss_weight = get_spec_with_default(specs, "ContactLossWeight", 0.005)
    contact_loss_sigma = get_spec_with_default(specs, "ContactLossSigma", 0.005)
    print("Penetration Loss:", do_penetration_loss)
    print("Penetration Loss Weight:", penetration_loss_weight)
    print("Additional Loss start at epoch:", start_additional_loss)
    print("Contact Loss:", do_contact_loss)
    print("Contact Loss Weight:", contact_loss_weight)
    print("Contact Loss Sigma (m):", contact_loss_sigma)

    latent_size = specs["LatentSize"]

    checkpoints = list(
        range(
            specs["SnapshotFrequency"],
            specs["NumEpochs"] + 1,
            specs["SnapshotFrequency"],
        )
    )
    for checkpoint in specs["AdditionalSnapshots"]:
        checkpoints.append(checkpoint)
    checkpoints.sort()

    lr_schedules = get_learning_rate_schedules(specs)


    grad_clip = get_spec_with_default(specs, "GradientClipNorm", None)
    if grad_clip is not None:
        logging.debug("clipping gradients to max norm {}".format(grad_clip))

    def save_latest(epoch):
        save_model(experiment_directory, "latest.pth", encoderDecoder, epoch)
        save_optimizer(experiment_directory, "latest.pth", optimizer_all, epoch)
        # save_latent_vectors(experiment_directory, "latest.pth", lat_vecs, epoch)

    def save_checkpoints(epoch):
        save_model(experiment_directory, str(epoch) + ".pth", encoderDecoder, epoch)
        save_optimizer(experiment_directory, str(epoch) + ".pth", optimizer_all, epoch)
        # save_latent_vectors(experiment_directory, str(epoch) + ".pth", lat_vecs, epoch)

    def signal_handler(sig, frame):
        logging.info("Stopping early...")
        sys.exit(0)

    def adjust_learning_rate(lr_schedules, optimizer, epoch):

        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)

    signal.signal(signal.SIGINT, signal_handler)

    # If true, use the data as-is. If false, multiply and offset obj location with normalized params
    indep_obj_scale = get_spec_with_default(specs, "IndependentObjScale", False)
    print("Independent Obj Scale:", indep_obj_scale)
    # Ignore points from other mesh in the begining when train 1 decoder
    ignore_other = get_spec_with_default(specs, "IgnorePointFromOtherMesh", False)
    print("Ignore other:", ignore_other)

    num_samp_per_scene = specs["SamplesPerScene"]
    scene_per_batch = specs["ScenesPerBatch"]
    scene_per_subbatch= scene_per_batch
    clamp_dist = specs["ClampingDistance"]
    minT = -clamp_dist
    maxT = clamp_dist
    enforce_minmax = True

    nb_classes = get_spec_with_default(specs["NetworkSpecs"], "num_class", 6)
    print("nb_label_class: ", nb_classes)

    ## Define Model
    if model_type == "PC_2encoder1decoder_VAE":
        kl_schedules = get_kl_weight_schedules(specs)

        input_type = 'point_cloud'
        same_point = True
        # If use 2 encoders, each encoder produces latent vector with half of the total size.
        half_latent_size = int(latent_size/2)
        print("Point cloud encoder, each branch has latent size", half_latent_size)
        
        encoder_obj = arch.ResnetPointnet(c_dim=half_latent_size, hidden_dim=256)
        use_sampling_trick = False
        if use_sampling_trick:
            encoder_hand = arch.ResnetPointnet(c_dim=latent_size, hidden_dim=256)
        else:
            encoder_hand = arch.ResnetPointnet(c_dim=latent_size, hidden_dim=256, cond_dim=latent_size)

        encoder_hand = encoder_hand.cuda()
        encoder_obj = encoder_obj.cuda()
        combined_decoder = arch.CombinedDecoder(latent_size, **specs["NetworkSpecs"], 
                                                use_classifier=classifier_branch).cuda()

        encoderDecoder = arch.ModelTwoEncodersOneDecoderVAE(
            encoder_hand, encoder_obj, combined_decoder, 
            nb_classes, num_samp_per_scene, 
            classifier_branch
        )
        encoderDecoder = encoderDecoder.cuda()

    encoder_input_source = data_source if input_type == 'point_cloud' else image_source

    logging.info("training with {} GPU(s)".format(torch.cuda.device_count()))

    encoderDecoder = torch.nn.DataParallel(encoderDecoder)

    num_epochs = specs["NumEpochs"]
    log_frequency = get_spec_with_default(specs, "LogFrequency", 5)
    log_frequency_step = get_spec_with_default(specs, "LogFrequencyStep", 100)

    logging.debug("torch num_threads: {}".format(torch.get_num_threads()))

    logging.debug(encoderDecoder)

    if "1decoder" in model_type and ignore_other:
        loss_l1 = torch.nn.L1Loss(reduction='sum')
    elif use_gaussian_reconstruction_weight:
        loss_l1 = torch.nn.L1Loss(reduction='none')
    else:
        loss_l1 = torch.nn.L1Loss()
    criterion_ce = torch.nn.CrossEntropyLoss(ignore_index=-1)

    if "VAE" in model_type:
        hand_latent_reg_l2 = torch.nn.MSELoss()

    optimizer_all = torch.optim.Adam(
        [
            {
                "params": encoderDecoder.parameters(),
            }
        ]
    )

    # Tensorboard summary
    writer = SummaryWriter(os.path.join(experiment_directory, 'log'))
    # writer.add_graph(encoderDecoder)

    start_epoch = 1
    # global_step = 0

    # continue from latest checkpoint if exists
    if (continue_from is None and 
            utils.misc.is_checkpoint_exist(experiment_directory, 'latest')):
        continue_from = 'latest'

    if continue_from is not None:
        logging.info('continuing from "{}"'.format(continue_from))

        model_epoch = utils.misc.load_model_parameters(
            experiment_directory, continue_from, encoderDecoder
        )

        optimizer_epoch = load_optimizer(
            experiment_directory, continue_from + ".pth", optimizer_all
        )
        start_epoch = model_epoch + 1
        logging.debug("loaded")

    # Data loader
    filter_dist = False
    if start_epoch >= start_additional_loss:
        same_point = True
        filter_dist = True

    with open(train_split_file, "r") as f:
        train_split = json.load(f)

    sdf_dataset = utils.data.SDFSamples(
        input_type,
        data_source, train_split, num_samp_per_scene,
        dataset_name=dataset_name,
        image_source=image_source,
        hand_branch=hand_branch, obj_branch=obj_branch,
        indep_obj_scale=indep_obj_scale,
        same_point=same_point,
        filter_dist=filter_dist,
        clamp=clamp_dist,
        load_ram=False,
        check_file=check_file, fhb=is_fhb,
        model_type=model_type,
        obj_center=obj_center
    )

    num_data_loader_threads = get_spec_with_default(specs, "DataLoaderThreads", 8)
    logging.debug("loading data with {} threads".format(num_data_loader_threads))

    sdf_loader = data_utils.DataLoader(
        sdf_dataset,
        batch_size=scene_per_subbatch,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=True
    )

    # training loop
    logging.info("starting from epoch {}".format(start_epoch))

    for epoch in range(start_epoch, num_epochs + 1):
        start = time.time()

        logging.info("epoch {}...".format(epoch))

        encoderDecoder.train()

        adjust_learning_rate(lr_schedules, optimizer_all, epoch)

        if 'VAE' in model_type:
            kl_weight = kl_schedules.get_weight(epoch)

        # Change sdf_loader to get sdf to both hand and object from the same points
        # print("same_point", same_point)
        if epoch == start_additional_loss : # and not same_point:
            same_point = True
            filter_dist = True
            sdf_dataset = utils.data.SDFSamples(
                input_type,
                data_source, train_split, num_samp_per_scene,
                dataset_name=dataset_name,
                image_source=image_source,
                hand_branch=hand_branch, obj_branch=obj_branch,
                indep_obj_scale=indep_obj_scale,
                same_point=same_point,
                filter_dist=filter_dist,
                clamp=clamp_dist,
                load_ram=False, # True
                check_file=check_file, fhb=is_fhb,
                model_type=model_type,
                obj_center=obj_center
            )
            sdf_loader = data_utils.DataLoader(
                sdf_dataset,
                batch_size=scene_per_subbatch,
                shuffle=True,
                num_workers=num_data_loader_threads,
                drop_last=True
            )
        
        for i, (hand_samples, hand_labels, obj_samples, obj_labels,
                scale, offset, encoder_input_hand, encoder_input_obj, idx) in enumerate(sdf_loader):

            batch_loss = 0.0
            optimizer_all.zero_grad()

            for _subbatch in range(batch_split):
                if input_type == 'image':
                    encoder_input_hand = encoder_input_hand.cuda()
                elif input_type == 'point_cloud':
                    encoder_input_hand = encoder_input_hand.cuda()
                    encoder_input_obj = encoder_input_obj.cuda()
                elif input_type == 'image+point_cloud':
                    encoder_input_hand = encoder_input_hand.cuda()
                    encoder_input_obj = encoder_input_obj.cuda()

                if '1decoder' in model_type:
                    # Using same point
                    if hand_branch and obj_branch:
                        samples = torch.cat([hand_samples, obj_samples], 1)
                        labels = torch.cat([hand_labels, obj_labels], 1)

                        # Ignore points from other shape in the begining of the training
                        if ignore_other or epoch < start_additional_loss:
                            mask_hand = torch.cat([torch.ones(hand_samples.size()[:2]), torch.zeros(obj_samples.size()[:2])], 1)
                            mask_hand = (mask_hand.cuda()).reshape(num_samp_per_scene * scene_per_subbatch).unsqueeze(1)
                            mask_obj = torch.cat([torch.zeros(hand_samples.size()[:2]), torch.ones(obj_samples.size()[:2])], 1)
                            mask_obj = (mask_obj.cuda()).reshape(num_samp_per_scene * scene_per_subbatch).unsqueeze(1)
                        else:
                            mask_hand = torch.ones(num_samp_per_scene * scene_per_subbatch).unsqueeze(1).cuda()
                            mask_obj = torch.ones(num_samp_per_scene * scene_per_subbatch).unsqueeze(1).cuda()
                    elif hand_branch:
                        samples = hand_samples
                        labels = hand_labels
                    elif obj_branch:
                        samples = obj_samples
                        labels = obj_labels
                    
                    samples.requires_grad = False
                    labels.requires_grad = False
                    
                    sdf_data = (samples.cuda()).reshape(
                        num_samp_per_scene * scene_per_subbatch, 5 
                    )
                    labels = (labels.cuda().to(torch.long)).reshape(
                        num_samp_per_scene * scene_per_subbatch)
                    xyz_hand = sdf_data[:, 0:3]
                    xyz_obj = xyz_hand
                    sdf_gt_hand = sdf_data[:, 3].unsqueeze(1)
                    sdf_gt_obj = sdf_data[:, 4].unsqueeze(1)
                else:
                    hand_samples.requires_grad = False
                    hand_labels.requires_grad = False
                    obj_samples.requires_grad = False
                    obj_labels.requires_grad = False
                    # Seperated points - Hand
                    if same_point:
                        samples = torch.cat([hand_samples, obj_samples], 1)
                        labels = torch.cat([hand_labels, obj_labels], 1)
                        sdf_data = (samples.cuda()).reshape(
                            num_samp_per_scene * scene_per_subbatch, 5
                        )
                        labels = (labels.cuda().to(torch.long)).reshape(
                            num_samp_per_scene * scene_per_subbatch)
                        hand_labels = labels
                        obj_labels = labels
                        xyz_hand = sdf_data[:, 0:3]
                        xyz_obj = xyz_hand
                        sdf_gt_hand = sdf_data[:, 3].unsqueeze(1)
                        sdf_gt_obj = sdf_data[:, 4].unsqueeze(1)

                    else:
                        sdf_data_hand = (hand_samples.cuda()).reshape(
                            num_samp_per_scene * scene_per_subbatch, 5
                        )
                        hand_labels = (hand_labels.cuda().to(torch.long)).reshape(
                            num_samp_per_scene * scene_per_subbatch)
                        xyz_hand = sdf_data_hand[:, 0:3]
                        sdf_gt_hand = sdf_data_hand[:, 3].unsqueeze(1)

                        # Object
                        sdf_data_obj = (obj_samples.cuda()).reshape(
                            num_samp_per_scene * scene_per_subbatch, 5
                        )
                        obj_labels = (obj_labels.cuda().to(torch.long)).reshape(
                            num_samp_per_scene * scene_per_subbatch)
                        xyz_obj = sdf_data_obj[:, 0:3]
                        sdf_gt_obj = sdf_data_obj[:, 4].unsqueeze(1)
                
                # scale
                scale = scale.cuda().repeat_interleave(num_samp_per_scene, dim=0)

                if enforce_minmax:
                    if hand_branch:
                        sdf_gt_hand = torch.clamp(sdf_gt_hand, minT, maxT)
                    if obj_branch:
                        sdf_gt_obj = torch.clamp(sdf_gt_obj, minT, maxT)
                
                if model_type == 'PC_2encoder1decoder_VAE':
                    pred_sdf_hand, pred_sdf_obj, pred_class, kl_loss, z_hand = encoderDecoder(encoder_input_hand, encoder_input_obj, xyz_hand)
                elif model_type == 'pc+1encoder1decoder':
                    pred_sdf_hand, pred_sdf_obj, pred_class = encoderDecoder(encoder_input_hand, encoder_input_obj, xyz_hand)
                elif '2encoder' in model_type and '1decoder' in model_type:
                    pred_sdf_hand, pred_sdf_obj, pred_class = encoderDecoder(encoder_input_hand, encoder_input_obj, xyz_hand)
                # same points
                elif '1decoder' in model_type: 
                    pred_sdf_hand, pred_sdf_obj, pred_class = encoderDecoder(encoder_input_hand, xyz_hand)
                else:
                    pred_sdf_hand, pred_class_hand,  \
                    pred_sdf_obj, pred_class_obj = encoderDecoder(encoder_input_hand, xyz_hand, xyz_obj)

                if enforce_minmax:
                    if hand_branch:
                        pred_sdf_hand = torch.clamp(pred_sdf_hand, minT, maxT)
                    if obj_branch:
                        pred_sdf_obj = torch.clamp(pred_sdf_obj, minT, maxT)

                ## Compute losses
                sigma_recon = 0.005 * 10.0
                if hand_branch:
                    if "1decoder" in model_type and ignore_other:
                        pred_sdf_hand = torch.mul(pred_sdf_hand, mask_hand)
                        loss_hand = loss_l1(pred_sdf_hand, sdf_gt_hand) / mask_hand.sum()
                    else:
                        loss_hand = loss_l1(pred_sdf_hand, sdf_gt_hand)
                else:
                    loss_hand = 0.
                if obj_branch:
                    if "1decoder" in model_type and ignore_other:
                        pred_sdf_obj = torch.mul(pred_sdf_obj, mask_obj)
                        loss_obj = loss_l1(pred_sdf_obj, sdf_gt_obj) / mask_obj.sum()
                    else:
                        loss_obj = loss_l1(pred_sdf_obj, sdf_gt_obj)
                else:
                    loss_obj = 0.
                
                if classifier_branch:
                    if not '1decoder' in model_type:
                        loss_ce = (criterion_ce(pred_class_hand, hand_labels) + 
                                   criterion_ce(pred_class_obj, obj_labels) ) * classifier_weight
                    else:
                        loss_ce = criterion_ce(pred_class, labels) * classifier_weight
                else:
                    loss_ce = 0

                loss = loss_hand + loss_obj
                if epoch >= start_additional_loss:
                    loss = loss + loss_ce

                if 'VAE' in model_type:
                    # KL-divergence
                    kl_loss_raw = kl_loss.mean()
                    # print("kl loss after mean", kl_loss.size())
                    kl_loss = kl_weight * kl_loss_raw
                    loss = loss + kl_loss

                if hand_branch:
                    scaled_pred_sdf_hand = pred_sdf_hand
                if obj_branch:
                    scaled_pred_sdf_obj = pred_sdf_obj 
                if do_penetration_loss:
                    pen_loss = torch.max(-(scaled_pred_sdf_hand + scaled_pred_sdf_obj), torch.Tensor([0]).cuda()).mean() * penetration_loss_weight
                    if epoch >= start_additional_loss:
                        loss = loss + pen_loss
                
                if do_contact_loss:
                    alpha = 1. / contact_loss_sigma**2
                    contact_loss = torch.min(alpha * (scaled_pred_sdf_hand**2 + scaled_pred_sdf_obj**2), torch.Tensor([1]).cuda()).mean() * contact_loss_weight
                    if epoch >= start_additional_loss:
                        loss = loss + contact_loss
                loss.backward()

                batch_loss += loss.item()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(encoderDecoder.parameters(), grad_clip)

            if ((epoch-1) * len(sdf_loader) + i) % log_frequency_step == 0:
                loss_hand_out = loss_hand.item() if hand_branch else 0
                loss_obj_out = loss_obj.item() if obj_branch else 0
                loss_ce_out = loss_ce.item() if classifier_branch else 0
                pen_loss_out = pen_loss.item() if do_penetration_loss else 0
                contact_loss_out = contact_loss.item() if do_contact_loss else 0

                print('step {}, loss {:.5f}, hand loss {:.5f}, object loss {:.5f}:, classifier loss {:.5f}, penetration loss {:.5f}, contact loss {:.5f} '.format(
                    (epoch-1) * len(sdf_loader) + i, 
                     loss.item(), loss_hand_out, loss_obj_out, loss_ce_out,
                     pen_loss_out, contact_loss_out))
                if 'VAE' in model_type:
                    print('KL loss {:.5f}'.format(kl_loss.item()))
                    writer.add_scalar('KL_loss_1e-3', kl_loss.item() * 1000.0, (epoch-1) * len(sdf_loader) + i)
                    writer.add_scalar('KL_loss_raw_1e-3', kl_loss_raw.item() * 1000.0, (epoch-1) * len(sdf_loader) + i)
                    
                writer.add_scalar('training_loss_1e-3', loss.item() * 1000.0, (epoch-1) * len(sdf_loader) + i)
                writer.add_scalar('loss_hand_1e-3', loss_hand_out * 1000.0, (epoch-1) * len(sdf_loader) + i)
                writer.add_scalar('loss_object_1e-3', loss_obj_out * 1000.0, (epoch-1) * len(sdf_loader) + i)
                writer.add_scalar('loss_classifier_1e-3', loss_ce_out * 1000.0, (epoch-1) * len(sdf_loader) + i)
                writer.add_scalar('loss_penetration_1e-3', pen_loss_out * 1000.0, (epoch-1) * len(sdf_loader) + i)
                writer.add_scalar('loss_contact_1e-3', contact_loss_out * 1000.0, (epoch-1) * len(sdf_loader) + i)

            optimizer_all.step()
        
        end = time.time()

        seconds_elapsed = end - start
        print("time used:", seconds_elapsed)

        for idx, schedule in enumerate(lr_schedules):
            writer.add_scalar('learning_rate_' + str(idx),
                schedule.get_learning_rate(epoch),
                epoch * len(sdf_loader)
            )

        recon_scale = 0.5 if not indep_obj_scale else 1.0
        
        if epoch in checkpoints and val_split_file:
            save_checkpoints(epoch)
            print("reconstruct mesh at {}".format(epoch))
            recon_st = time.time()
            reconstruct.reconstruct_training(experiment_directory, 
                val_split_file,
                input_type,
                encoder_input_source,
                encoderDecoder,
                epoch,
                specs,
                hand_branch,
                obj_branch,
                model_type=model_type,
                scale=recon_scale, #
                cube_dim=128,
                fhb=is_fhb,
                dataset_name=dataset_name)
            print("- Reconstruction used {}".format(time.time()-recon_st))
            
            # chamfer_st = time.time()
            # object_type, chamfer_mean_list, chamfer_med_list, = evaluate.evaluate(
            #     experiment_directory,
            #     str(epoch),
            #     data_source,
            #     val_split_file,
            # )
            # print("calculate chamfer dist used {}".format(time.time()-chamfer_st))
            # print(" - Chamfer distance:")
            # for i, obj_type in enumerate(object_type):
            #     print("{}: mean: {:.5f}, med: {:.5f}".format(obj_type, chamfer_mean_list[i], chamfer_med_list[i]))
            #     writer.add_scalar(obj_type+'_val_chamfer_mean_x1000', 
            #         chamfer_mean_list[i] * 1000.0, epoch)
            #     writer.add_scalar(obj_type+'_val_chamfer_med_x1000', 
            #         chamfer_med_list[i] * 1000.0, epoch)
        

        if epoch % log_frequency == 0:
            save_latest(epoch)
            print("save at {}".format(epoch))
    

    # End of training
    if val_split_file:
        print("Final reconstruct mesh at {}".format(num_epochs))
        recon_st = time.time()
        reconstruct.reconstruct_training(experiment_directory, 
            val_split_file,
            input_type,
            encoder_input_source,
            encoderDecoder,
            num_epochs,
            specs,
            hand_branch,
            obj_branch,
            model_type=model_type,
            scale=recon_scale, #
            cube_dim=256,
            fhb=is_fhb,
            dataset_name=dataset_name)
        print("- Final Reconstruction used {}".format(time.time()-recon_st))
        
        # chamfer_st = time.time()
        # object_type, chamfer_mean_list, chamfer_med_list, = evaluate.evaluate(
        #     experiment_directory,
        #     str(num_epochs),
        #     data_source,
        #     val_split_file,
        # )
        # print("calculate final chamfer dist used {}".format(time.time()-chamfer_st))
        # print(" - Chamfer distance:")
        # for i, obj_type in enumerate(object_type):
        #     print("{}: mean: {:.5f}, med: {:.5f}".format(obj_type, chamfer_mean_list[i], chamfer_med_list[i]))
        #     writer.add_scalar(obj_type+'_val_chamfer_mean_x1000', 
        #         chamfer_mean_list[i] * 1000.0, num_epochs)
        #     writer.add_scalar(obj_type+'_val_chamfer_med_x1000', 
        #         chamfer_med_list[i] * 1000.0, num_epochs)

    writer.close()


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Train a DeepSDF autodecoder")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json', and logging will be "
        + "done in this directory as well.",
    )
    arg_parser.add_argument(
        "--continue",
        "-c",
        dest="continue_from",
        help="A snapshot to continue from. This can be 'latest' to continue"
        + "from the latest running snapshot, or an integer corresponding to "
        + "an epochal snapshot. Load latest checkpoint by default.",
    )
    arg_parser.add_argument(
        "--batch_split",
        dest="batch_split",
        default=1,
        help="This splits the batch into separate subbatches which are "
        + "processed separately, with gradients accumulated across all "
        + "subbatches. This allows for training with large effective batch "
        + "sizes in memory constrained environments.",
    )

    utils.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    utils.configure_logging(args)
    
    main_function(args.experiment_directory, args.continue_from, int(args.batch_split))