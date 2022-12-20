import torch
from torch import nn
from module import ConvBlock, DownSample, UpSample, SkipBlock, \
    MemModule, TemporalShift, ChannelAttention
from einops import rearrange


class Encoder(nn.Module):
    def __init__(self, initial_channel=12, skip=False):
        super(Encoder, self).__init__()
        self.block1 = ConvBlock(initial_channel, 64, 64)
        self.downblock1 = DownSample(64)
        self.block2 = ConvBlock(64, 128, 128)
        self.downblock2 = DownSample(128)
        self.block3 = ConvBlock(128, 256, 256)
        self.downblock3 = DownSample(256)
        self.block4 = ConvBlock(256, 512, 512)
        self.skip = skip

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.downblock1(x1)
        x2 = self.block2(x2)
        x3 = self.downblock2(x2)
        x3 = self.block3(x3)
        x4 = self.downblock3(x3)
        x4 = self.block4(x4)
        if self.skip:
            return x4, x3, x2, x1
        else:
            return x4


class Decoder(nn.Module):
    def __init__(self, with_last_relu=False):
        super(Decoder, self).__init__()
        self.block1 = ConvBlock(512, 256, 256)
        self.upblock1 = UpSample(256)
        self.block2 = ConvBlock(256, 128, 128)
        self.upblock2 = UpSample(128)
        self.block3 = ConvBlock(128, 64, 64)
        self.upblock3 = UpSample(64)
        self.block4 = ConvBlock(64, 3, 3, with_last_relu)

    def forward(self, x):
        x = self.block1(x)
        x = self.upblock1(x)
        x = self.block2(x)
        x = self.upblock2(x)
        x = self.block3(x)
        x = self.upblock3(x)
        x = self.block4(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, cfg, reduction='mean'):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(cfg.with_last_relu)
        self.Loss = torch.nn.MSELoss(reduction=reduction)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def loss(self, pred, true):
        predloss = self.Loss(pred, true)
        return dict(Pred_Loss=predloss, Total_Loss=predloss)


class SkipDecoder(nn.Module):
    def __init__(self, with_last_relu=False):
        super(SkipDecoder, self).__init__()
        self.skipblock1 = SkipBlock(512, 256)
        self.skipblock2 = SkipBlock(256, 128)
        self.convblock1 = ConvBlock(128, 64, 64)
        self.upsample = UpSample(64)
        self.convblock2 = ConvBlock(64, 3, 3, with_last_relu)

    def forward(self, x, x3, x2):
        # x4 [512, 14] x3 [256, 28]  x2 [128, 56]  x1 [64, 112]
        x = self.skipblock1(x, x3)
        x = self.skipblock2(x, x2)
        x = self.convblock1(x)
        x = self.upsample(x)
        x = self.convblock2(x)  # [64, 112] -> [3, 112]
        return x


class SkipAutoEncoder(nn.Module):
    def __init__(self, cfg, reduction='mean'):
        super(SkipAutoEncoder, self).__init__()
        self.encoder = Encoder(skip=True)
        self.decoder = SkipDecoder(cfg.with_last_relu)
        self.Loss = torch.nn.MSELoss(reduction=reduction)

    def forward(self, x):
        x, x3, x2, x1 = self.encoder(x)
        x = self.decoder(x, x3, x2)
        return x

    def loss(self, pred, true):
        predloss = self.Loss(pred, true)
        return dict(Pred_Loss=predloss, Total_Loss=predloss)


class MemAutoEncoder(nn.Module):
    def __init__(self, cfg, num_protos=20, fea_dim=512, reduction='mean'):
        super(MemAutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(cfg.with_last_relu)
        self.mem = MemModule(num_protos, fea_dim)
        self.Loss = torch.nn.MSELoss(reduction=reduction)

    def forward(self, x):
        x = self.encoder(x)
        x = self.mem(x)
        y = self.decoder(x['output'])
        return y, x['similarity']

    def loss(self, pred, label):
        pred_recon = pred[0]
        similarity = pred[1]
        truedata = label
        recon_loss = self.Loss(pred_recon, truedata)
        entropy_loss = self.EntropyLoss(similarity)
        total_loss = recon_loss + 0.0002 * entropy_loss
        return dict(Pred_Loss=recon_loss, Entropy_Loss=entropy_loss, Total_Loss=total_loss)

    def EntropyLoss(self, similarity, eps=1e-12):
        similarity = similarity.permute(0, 2, 3, 1)  # [B, N, H, W] -> [B, H, W, N]
        similarity = similarity.contiguous().view(-1, similarity.shape[-1])  # [BHW, N]
        res = similarity * torch.log(similarity + eps)  # [BHW, N]
        res = -1.0 * res.sum(dim=1)  # [BHW]
        loss = res.mean()
        return loss


class Mean_Var_Forward_Projection(nn.Module):
    def __init__(self):
        super(Mean_Var_Forward_Projection, self).__init__()
        self.downblock = DownSample(512)
        self.fc = nn.Linear(512 * 7 * 7, 1024)
        # self.dropout = nn.Dropout(0.3)
        self.relu = nn.LeakyReLU(inplace=True)
        self.fc_mean = nn.Linear(1024, 256)
        self.fc_logvar = nn.Linear(1024, 256)

    def forward(self, x):
        x = self.downblock(x)
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc(x))
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar


class Mean_Var_Backward_Projection(nn.Module):
    def __init__(self):
        super(Mean_Var_Backward_Projection, self).__init__()
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 512 * 7 * 7)
        self.relu = nn.LeakyReLU(inplace=True)
        self.upblock = UpSample(512)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = x.view(-1, 512, 7, 7)
        x = self.upblock(x)
        return x


class VAE(nn.Module):
    def __init__(self, cfg, num_z=5, reduction='mean'):
        super(VAE, self).__init__()
        self.num_z = num_z
        self.encoder = Encoder()
        self.mean_var_forward_projection = Mean_Var_Forward_Projection()
        self.mean_var_backward_projection = Mean_Var_Backward_Projection()
        self.decoder = Decoder(cfg.with_last_relu)
        self.pred_loss = torch.nn.MSELoss(reduction=reduction)

    def forward(self, x):
        Z = []
        mean, logvar = self.mean_var_forward_projection(self.encoder(x))
        for i in range(self.num_z):
            z = (self.reparameterize(mean, logvar))
            Z.append(self.decoder(self.mean_var_backward_projection(z)))
        return Z, mean, logvar

    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        return eps * std + mean

    def loss(self, X, truelabel):
        recon_Loss = 0
        Z, mean, logvar = X
        for i in range(self.num_z):
            recon_Loss += self.pred_loss(Z[i], truelabel)
        recon_Loss /= self.num_z
        kld_Loss = self.KL_loss(mean, logvar)
        total_Loss = recon_Loss + 0.8 * kld_Loss
        return dict(Pred_Loss=recon_Loss, KL_Loss=kld_Loss, Total_Loss=total_Loss)

    @staticmethod
    def KL_loss(mean, var):
        return torch.mean(-0.5 * torch.sum(1 + var - mean ** 2 - var.exp(), dim=1), dim=0)


class NewTemporalBranch(nn.Module):
    def __init__(self, num_protos, num_frames=4, n_div=4, direction='left', with_temporal_shift=True):
        super(NewTemporalBranch, self).__init__()
        self.num_frames = num_frames
        self.with_temporal_shift = with_temporal_shift
        if with_temporal_shift:
            self.TSM = TemporalShift(n_div=n_div, direction=direction)
        print('using Temporal Shift Module' if with_temporal_shift else 'not using Temporal Shift Module')
        self.reduce = nn.Conv2d(512, 512 // (8 if with_temporal_shift else 4), kernel_size=1, bias=False)
        self.TemporalMem = MemModule(num_protos=num_protos, fea_dim=196, branch=True)

    def forward(self, x):
        # [BN, C, H, W]  (32, 512, 14, 14)
        BN, C, H, W = x.shape
        batch = BN // self.num_frames  # batch = 8
        if self.with_temporal_shift:
            x = x.view(-1, self.num_frames, C, H, W).contiguous()  # [B, N, C, H, W]  (8, 4, 512, 14, 14)
            x = torch.cat((self.TSM(x), x), 1)  # [B, 2N, C, H, W]  (8, 8, 512, 14, 14)
            x = x.view(-1, C, H, W)  # [B * 2N, C, H, W]  (64, 512, 14, 14)
        x = self.reduce(x)  # [B * 2N, C1, H, W]   (64, 64, 14, 14)
        x = rearrange(x, '(B N) C H W -> B (N C) (H W)', B=batch)   # (8*8, 64, 14, 14) -> (8, 8*64, 196)
        x = self.TemporalMem(x)  # [B, 2N, H*W] (8, 512, 196)
        x['output'] = x['output'].contiguous().view(batch, -1, H, W)  # [B, 2N, H, W]  (8, 512, 14, 14)
        return x


class NewSpatialBranch(nn.Module):
    def __init__(self, num_protos, num_frames=4):
        super(NewSpatialBranch, self).__init__()
        self.num_frames = num_frames
        self.SpatialMem = MemModule(num_protos=num_protos, branch=True)
        self.reduce = nn.Conv2d(512 * num_frames, 512, kernel_size=1, bias=False)
        self.channelattention = ChannelAttention(512)

    def forward(self, x):
        # [BN, C, H, W]  (32, 512, 14, 14)
        BN, C, H, W = x.shape
        x = rearrange(x, '(B N) C H W -> B (N C) H W', N=self.num_frames)  # (8, 2048, 14, 14)
        x1 = self.reduce(x)  # [B, NC, H, W] -> [B, C, H, W]  (8, 512, 14, 14)
        x, att = self.channelattention(x1)  # (8, 512, 14, 14)
        x = rearrange(x, 'B C H W -> B (H W) C')
        y = self.SpatialMem(x)  # [B, 196, 512]
        y['output'] = rearrange(y['output'], 'B (H W) C -> B C H W', H=H)  # [B, 512, 14, 14]
        return y, att, x1


class NewSpatialTemporalMemAutoEncoder(nn.Module):
    def __init__(self, cfg, reduction='mean'):
        super(NewSpatialTemporalMemAutoEncoder, self).__init__()
        print('this is new version')
        self.encoder = Encoder(initial_channel=3, skip=False)
        self.decoder = Decoder(cfg.with_last_relu)
        self.temporalbranch = NewTemporalBranch(cfg.num_protos, with_temporal_shift=cfg.with_temporal_shift)
        self.spatialbranch = NewSpatialBranch(cfg.num_protos)
        self.Loss = torch.nn.MSELoss(reduction=reduction)
        self.Gradient_Loss = torch.nn.MSELoss()

    def forward(self, x):
        x = self.encoder(x)  # [4*B, 512, 14, 14]
        _, C, size, size = x.shape
        temporal_fea = self.temporalbranch(x)  # (8, 512, 14, 14)
        spatial_fea, channel_att, channel_feat = self.spatialbranch(x)  # [B, 512, 14, 14]
        total_fea = temporal_fea['output'] + spatial_fea['output']
        resdict = dict(temporal_similarity=temporal_fea['similarity'],
                       spatial_similarity=spatial_fea['similarity'],
                       att=channel_att,
                       feat=channel_feat)
        y = self.decoder(total_fea)
        return y, resdict

    def loss(self, pred, label):
        pred_img = pred[0]
        similarity = pred[1]
        truedata = label
        pred_loss = self.Loss(pred_img, truedata)
        entropy_loss_temporal = self.EntropyLoss(similarity['temporal_similarity'])
        entropy_loss_spatial = self.EntropyLoss(similarity['spatial_similarity'])
        gradient_loss = self.GradientLoss(pred_img, truedata)
        total_loss = pred_loss + gradient_loss + 0.0002 * entropy_loss_temporal + 0.0002 * entropy_loss_spatial
        loss_dict = dict(Pred_Loss=pred_loss, Gradient_Loss=gradient_loss, Entropy_Loss_Temporal=entropy_loss_temporal,
                         Entropy_Loss_Spatial=entropy_loss_spatial, Total_Loss=total_loss)
        return loss_dict

    @staticmethod
    def EntropyLoss(similarity, eps=1e-12):
        similarity = similarity.permute(0, 2, 1)  # [B, N, X] -> [B, X, N]
        similarity = similarity.contiguous().view(-1, similarity.shape[-1])  # [BX, N]
        res = similarity * torch.log(similarity + eps)  # [BX, N]
        res = -1.0 * res.sum(dim=1)  # [BX]
        loss = res.mean()
        return loss

    def GradientLoss(self, generated, real_image):  # b x c x h x w
        true_x_shifted_right = real_image[:, :, 1:, :]
        true_x_shifted_left = real_image[:, :, :-1, :]
        true_x_gradient = torch.abs(true_x_shifted_left - true_x_shifted_right)

        generated_x_shift_right = generated[:, :, 1:, :]
        generated_x_shift_left = generated[:, :, :-1, :]
        generated_x_gradient = torch.abs(generated_x_shift_left - generated_x_shift_right)
        loss_x_gradient = self.Gradient_Loss(generated_x_gradient, true_x_gradient)

        true_y_shifted_right = real_image[:, :, :, 1:]
        true_y_shifted_left = real_image[:, :, :, :-1]
        true_y_gradient = torch.abs(true_y_shifted_left - true_y_shifted_right)

        generated_y_shift_right = generated[:, :, :, 1:]
        generated_y_shift_left = generated[:, :, :, :-1]
        generated_y_gradient = torch.abs(generated_y_shift_left - generated_y_shift_right)
        loss_y_gradient = self.Gradient_Loss(generated_y_gradient, true_y_gradient)
        total_gradient_loss = (loss_x_gradient + loss_y_gradient) / 2
        return total_gradient_loss


if __name__ == '__main__':
    from Config.Config_AE import AEConfig
    a = torch.ones(32, 3, 112, 112)
    cfg = AEConfig()
    model = NewSpatialTemporalMemAutoEncoder(cfg)  # [8, 512, 14, 14]
    # print(model)
    out = model(a)
    for i in out:
        if isinstance(i, dict):
            for k, v in i.items():
                print(v.shape)
        else:
            print(i.shape)

# spatial
# torch.Size([8, 512, 196])
# torch.Size([8, 20, 196])

# temporal
# torch.Size([8, 196, 512])
# torch.Size([8, 20, 512])
