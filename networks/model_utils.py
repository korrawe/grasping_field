import os
import torch
# import utils
import networks.model as arch
# import pcl2mano.pcl2mano as mano_helper
# import utils.misc as misc_utils
# from utils.misc import get_spec_with_default
# from networks.model_utils import get_model


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


def get_model(model_directory, specs, device):
    model_type = specs["ModelType"]
    latent_size = specs["LatentSize"]
    nb_classes = get_spec_with_default(specs["NetworkSpecs"], "num_class", 6)
    classifier_branch = get_spec_with_default(specs, "ClassifierBranch", False)

    if model_type == "PC_2encoder1decoder_VAE":
        # input_type = 'point_cloud'
        # If use 2 encoders, each encoder produces latent vector with half of the total size.
        half_latent_size = int(latent_size / 2)
        # print("Point cloud encoder, each branch has latent size", half_latent_size)

        encoder_obj = arch.ResnetPointnet(c_dim=half_latent_size, hidden_dim=256)
        # hand encoder get 2xlatent_size, half for mean, another for variance.
        encoder_hand = arch.ResnetPointnet(c_dim=latent_size, hidden_dim=256, cond_dim=latent_size)

        combined_decoder = arch.CombinedDecoder(latent_size, **specs["NetworkSpecs"],
                                                use_classifier=classifier_branch)

        encoderDecoder = arch.ModelTwoEncodersOneDecoderVAE(
            encoder_hand, encoder_obj, combined_decoder,
            nb_classes, specs["SamplesPerScene"],
            classifier_branch
        )
    elif model_type == "1encoder1decoder":
        # use_combined_decoder = True
        encoder, encoder_input_size = arch.get_encoder('resnet', output_size=latent_size, use_pretrained=True,
                                                       feature_extract=False)  # True
        combined_decoder = arch.CombinedDecoder(latent_size, **specs["NetworkSpecs"],
                                                use_classifier=classifier_branch)

        encoderDecoder = arch.ModelOneEncodersOneDecoder(
            encoder, combined_decoder,
            nb_classes, specs["SamplesPerScene"],
            classifier_branch
        )

    encoderDecoder = torch.nn.DataParallel(encoderDecoder)

    # Load weights
    saved_model_state = torch.load(
        os.path.join(model_directory, "model.pth")
    )
    saved_model_epoch = saved_model_state["epoch"]
    # logging.info("using model from epoch {}".format(saved_model_epoch))

    encoderDecoder.load_state_dict(saved_model_state["model_state_dict"])

    encoderDecoder = encoderDecoder.to(device)  # .cuda()

    return encoderDecoder  # loaded_model
