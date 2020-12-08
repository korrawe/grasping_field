import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist
from torchvision import datasets, models, transforms


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def get_encoder(model_name, use_pretrained, feature_extract, output_size):
    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, output_size)
        input_size = 224
    else:
        print("Invalid model name, exiting...")
        exit()
    return model_ft, input_size


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.
    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class ResnetPointnet(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks.
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, cond_dim=None):
        super().__init__()
        self.c_dim = c_dim
        self.cond_dim = cond_dim

        if cond_dim is None:
            self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        else:
            self.fc_pos = nn.Linear(dim, hidden_dim)
        self.block_0 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_1 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_2 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_3 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_4 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p, cond=None):
        # batch_size, T, D = p.size()

        # output size: B x T X F
        # print("p", p.type())
        net = self.fc_pos(p)
        if self.cond_dim is not None:
            # print("cond size", cond.size())
            # print("net size", net.size())
            # concat conditioning vertor
            cond = cond.unsqueeze(1).expand(net.size())
            net = torch.cat([net, cond.expand(net.size())], dim=2)
        net = self.block_0(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_1(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_2(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_3(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_4(net)
        # Recude to  B x F
        net = self.pool(net, dim=1)

        c = self.fc_c(self.actvn(net))

        return c


class ResnetPointnetCond(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks.
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, cond_dim=128):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, hidden_dim)  # 2*hidden_dim)
        self.block_0 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_1 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_2 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_3 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_4 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p, cond):
        # batch_size, T, D = p.size()

        # output size: B x T X F
        net = self.fc_pos(p)
        # concat conditioning vertor
        net = torch.cat([net, cond.expand(net.size())], dim=2)

        net = self.block_0(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_1(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_2(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_3(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_4(net)
        # Recude to  B x F
        net = self.pool(net, dim=1)

        c = self.fc_c(self.actvn(net))

        return c


class ModelTwoEncodersOneDecoderVAE(nn.Module):
    def __init__(self, encoder_hand, encoder_obj, decoder, nb_classes, num_samp_per_scene,
                 classifier_branch=False, use_sampling_trick=False):
        super(ModelTwoEncodersOneDecoderVAE, self).__init__()
        self.encoder_hand = encoder_hand
        self.encoder_obj = encoder_obj
        self.decoder = decoder

        self.num_class = nb_classes
        self.num_samp_per_scene = num_samp_per_scene

        self.use_sampling_trick = use_sampling_trick
        self.GMM = None
        # print("Sample per scene", self.num_samp_per_scene)

        self.hand_encoder_c_dim = int(encoder_hand.c_dim / 2)

    def forward(self, x_hand=None, x_obj=None, xyz=None, sample=False):
        x_obj = self.encoder_obj(x_obj)
        latent_obj = x_obj.repeat_interleave(self.num_samp_per_scene, dim=0)

        if not sample:
            if self.use_sampling_trick:
                x_hand = self.encoder_hand(x_hand)
            else:
                # condition hand on obj
                x_hand = self.encoder_hand(x_hand, x_obj)
            mean_z = x_hand[:, :self.hand_encoder_c_dim]  # first half
            logstd_z = x_hand[:, self.hand_encoder_c_dim:]  # second half
        else:
            batch_size = x_obj.size(0)
            mean_z = torch.empty(batch_size, 0).cuda()
            logstd_z = torch.empty(batch_size, 0).cuda()
        q_z = dist.Normal(mean_z, torch.exp(logstd_z))
        z_hand = q_z.rsample()

        # KL-divergence
        p0_z = dist.Normal(torch.zeros(self.hand_encoder_c_dim).cuda(), torch.ones(self.hand_encoder_c_dim).cuda())
        kl_loss = dist.kl_divergence(q_z, p0_z)

        latent_hand = z_hand.repeat_interleave(self.num_samp_per_scene, dim=0)
        latent = torch.cat([latent_hand, latent_obj], 1)
        decoder_inputs = torch.cat([latent, xyz], 1)
        x_hand, x_obj, x_class = self.decoder(decoder_inputs)

        return x_hand, x_obj, x_class, kl_loss, z_hand

    def compute_obj_latent(self, x_obj):
        x_obj = self.encoder_obj(x_obj)
        return x_obj

    def compute_latent(self, x_hand=None, x_obj=None, sample=True):
        x_obj = self.encoder_obj(x_obj)

        if not sample:
            # condition hand on obj
            x_hand = self.encoder_hand(x_hand, x_obj)
            mean_z = x_hand[:, :self.hand_encoder_c_dim]  # first half
            logstd_z = x_hand[:, self.hand_encoder_c_dim:]  # second half
            std_z = torch.zeros(x_obj.size()).cuda()
            q_z = dist.Normal(mean_z, torch.exp(logstd_z))

            z_hand = mean_z
        else:
            batch_size = x_obj.size(0)
            mean_z = torch.empty(batch_size, 0).cuda()
            std_z = torch.empty(batch_size, 0).cuda()
            q_z = dist.Normal(torch.zeros(x_obj.size()).cuda(), torch.ones(x_obj.size()).cuda())
            z_hand = q_z.rsample()

        latent = torch.cat([z_hand, x_obj], 1)
        # print("latent size", latent.size())

        return latent


class CombinedDecoder(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        num_class,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        use_tanh=False,
        latent_dropout=False,
        use_classifier=False,
    ):
        super(CombinedDecoder, self).__init__()

        def make_sequence():
            return []

        dims = [latent_size + 3] + dims + [2]  # <<<< 2 outputs instead of 1.

        self.num_layers = len(dims)
        self.num_class = num_class
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm
        self.use_classifier = use_classifier

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= 3
            # print("out dim", out_dim)

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

            # print(dims[layer], out_dim)
            # classifier
            if self.use_classifier and layer == self.num_layers - 2:
                # print("dim last_layer", dims[layer])
                self.classifier_head = nn.Linear(dims[layer], self.num_class)

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    # input: N x (L+3)
    def forward(self, input):
        xyz = input[:, -3:]

        if input.shape[1] > 3 and self.latent_dropout:
            latent_vecs = input[:, :-3]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = input

        for layer in range(0, self.num_layers - 1):
            # classify
            if self.use_classifier and layer == self.num_layers - 2:
                predicted_class = self.classifier_head(x)

            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            x = lin(x)
            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)

        # hand, object, class label
        if self.use_classifier:
            return x[:, 0].unsqueeze(1), x[:, 1].unsqueeze(1), predicted_class
        else:
            return x[:, 0].unsqueeze(1), x[:, 1].unsqueeze(1), torch.Tensor([0]).cuda()


class ModelOneEncodersOneDecoder(nn.Module):
    def __init__(self, encoder, decoder, nb_classes, num_samp_per_scene,
                 hand_branch=True, obj_branch=True, classifier_branch=False):
        super(ModelOneEncodersOneDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_class = nb_classes
        self.num_samp_per_scene = num_samp_per_scene

    def forward(self, x1, xyz):
        x1 = self.encoder(x1)
        latent = x1.repeat_interleave(self.num_samp_per_scene, dim=0)

        decoder_inputs = torch.cat([latent, xyz], 1)
        x_hand, x_obj, x_class = self.decoder(decoder_inputs)
        return x_hand, x_obj, x_class


class ModelTwoEncodersOneDecoder(nn.Module):
    def __init__(self, encoder_hand, encoder_obj, decoder, nb_classes, num_samp_per_scene,
                 hand_branch=True, obj_branch=True, classifier_branch=False):
        super(ModelTwoEncodersOneDecoder, self).__init__()
        self.encoder_hand = encoder_hand
        self.encoder_obj = encoder_obj
        self.decoder = decoder
        self.num_class = nb_classes
        self.num_samp_per_scene = num_samp_per_scene

        print(self.num_samp_per_scene)

    def forward(self, x_hand, x_obj, xyz):
        x_hand = self.encoder_hand(x_hand)
        latent_hand = x_hand.repeat_interleave(self.num_samp_per_scene, dim=0)
        x_obj = self.encoder_obj(x_obj)
        latent_obj = x_obj.repeat_interleave(self.num_samp_per_scene, dim=0)

        latent = torch.cat([latent_hand, latent_obj], 1)

        decoder_inputs = torch.cat([latent, xyz], 1)
        x_hand, x_obj, x_class = self.decoder(decoder_inputs)
        return x_hand, x_obj, x_class
