import math
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from configs.paths_config import model_paths
from criteria.lpips.lpips import LPIPS
from models.invertibility.deeplab import DeepLab
from models.segmenter import SegmenterFace
from models.stylegan2.model import Generator
from utils.train_utils import load_train_checkpoint
from inference.inference import BaseInference
from models.encoder import Encoder
import tqdm


latent_names = "W+,F4,F6,F8,F10"


def get_mvg_stats(G, device=torch.device('cuda')):
    # label_c = torch.zeros([1, G.c_dim], device=device)
    buf_v = np.zeros((5000, 512))
    buf_w = np.zeros((5000, 512))
    for i in range(5000):
        _z = torch.randn(1, 512).to(device)
        with torch.no_grad():
            _w = G.style(_z)
        _v = F.leaky_relu(_w, negative_slope=5.0)
        buf_w[i, :] = _w.cpu().numpy().reshape(512)
        buf_v[i, :] = _v.cpu().numpy().reshape(512)
    cov_v_np, cov_w_np = np.cov(buf_v.T)+np.eye(512)*1e-8, np.cov(buf_w.T)+np.eye(512)*1e-8
    inv_cov_v_np, inv_cov_w_np = np.linalg.inv(cov_v_np), np.linalg.inv(cov_w_np)
    inv_cov_v, inv_cov_w = torch.tensor(inv_cov_v_np).cuda().double(), torch.tensor(inv_cov_w_np).cuda().double()
    mean_w = torch.tensor(np.mean(buf_w, axis=0)).cuda().float()
    mean_v = F.leaky_relu(mean_w, negative_slope=5.0)
    return {
        "mean_v": mean_v,
        "mean_w": mean_w,
        "inv_cov_v": inv_cov_v,
        "inv_cov_w": inv_cov_w,
    }


def refine(d_invmaps, seg_map, tau):
    H, W = seg_map.shape
    refined = np.zeros((H, W))
    idx2latent = ["F10", "F8", "F6", "F4", "W+"]
    latent2idx = {n: i for i, n in enumerate(idx2latent)}
    # iterate through each segment index
    for v in np.unique(seg_map):
        curr_segment = (seg_map == v)
        latent_for_this_segment = "F10"
        for l_name in idx2latent:
            # check the average inv value inside the segment
            avg_val = (d_invmaps[l_name].detach().cpu()*curr_segment).sum() / curr_segment.sum()
            if avg_val <= tau:
                latent_for_this_segment = l_name
        refined[curr_segment] = latent2idx[latent_for_this_segment]
    # expand the latent map into individual binary masks
    d_refined = {name: torch.tensor((refined == idx)[None, None]) for idx, name in enumerate(idx2latent)}
    return d_refined


def resize_binary_masks(d_refined_invmap):
    d_out = {
        "W+": d_refined_invmap["W+"][0, 0].detach().cpu().numpy(),
        "F4": resize_single_channel(d_refined_invmap["F4"][0, 0].detach().cpu().numpy(), (16, 16), Image.LANCZOS),
        "F6": resize_single_channel(d_refined_invmap["F6"][0, 0].detach().cpu().numpy(), (32, 32), Image.LANCZOS),
        "F8": resize_single_channel(d_refined_invmap["F8"][0, 0].detach().cpu().numpy(), (64, 64), Image.LANCZOS),
        "F10": resize_single_channel(d_refined_invmap["F10"][0, 0].detach().cpu().numpy(), (128, 128), Image.LANCZOS),
    }
    d_out = {k: torch.tensor(d_out[k][None, None]).cuda() for k in d_out.keys()}
    return d_out


def resize_single_channel(x_np, S, k=Image.LANCZOS):
    s1, s2 = S
    img = Image.fromarray(x_np.astype(np.float32), mode='F')
    img = img.resize(S, resample=k)
    return np.asarray(img).reshape(s2, s1).clip(0, x_np.max())


def compute_mvg(d_latents, latent_name, mean_v, inv_cov_v):
    if latent_name == "W":
        _w = d_latents["W"]
        _v = F.leaky_relu(_w, negative_slope=5.0)
        dv = (_v - mean_v)
        loss = (dv.matmul(inv_cov_v).matmul(dv.T))
        return loss
    elif latent_name == "W+":
        _wp = d_latents["W+"].double()
        _vp = F.leaky_relu(_wp, negative_slope=5.0)
        loss = 0.0
        for idx in range(_vp.shape[1]):
            _v = _vp[:, idx, :]
            dv = (_v - mean_v)
            loss += (dv@inv_cov_v@dv.T)
        return loss.squeeze(0).squeeze(0)


def delta_loss(latent):
    loss = 0.0
    first_w = latent[:, 0, :]
    for i in range(1, latent.shape[1]):
        delta = latent[:, i, :] - first_w
        delta_loss = torch.norm(delta, 2, dim=1).mean()
        loss += delta_loss
    return loss


class SamInference(BaseInference):

    def __init__(self, opts, decoder=None):
        super(SamInference, self).__init__()
        self.opts = opts
        self.device = 'cuda'
        self.opts.device = self.device
        self.opts.n_styles = int(math.log(opts.resolution, 2)) * 2 - 2

        # resume from checkpoint
        checkpoint = load_train_checkpoint(opts)

        # initialize decoder
        latent_avg = None
        if decoder is not None:
            self.decoder = decoder
        else:
            self.decoder = Generator(opts.resolution, 512, 8).to(self.device)
            self.decoder.train()
            if checkpoint is not None:
                self.decoder.load_state_dict(checkpoint['decoder'], strict=True)
            else:
                decoder_checkpoint = torch.load(opts.stylegan_weights, map_location='cpu')
                self.decoder.load_state_dict(decoder_checkpoint['g_ema'])
                latent_avg = decoder_checkpoint['latent_avg']
        if latent_avg is None:
            latent_avg = self.decoder.mean_latent(int(1e5))[0].detach() if checkpoint is None else None
        # self.encoder = Encoder(opts, checkpoint, latent_avg, device=self.device).to(self.device)
        # self.encoder.set_progressive_stage(self.opts.n_styles)
        self.d_stats = get_mvg_stats(self.decoder)

        # initialize invertibility predictor
        self.invert_predictor = DeepLab(num_classes=8, backbone="resnet", output_stride=16, sync_bn=False, freeze_bn=False).to(self.device)
        self.invert_predictor.eval()
        sd = torch.load(model_paths['invert_predictor_faces'], map_location='cpu')
        self.invert_predictor.load_state_dict(sd["sd_base"])
        self.d_heads = {}
        for name in latent_names.split(","):
            self.d_heads[name] = LayerHead().cuda()
            self.d_heads[name].load_state_dict(sd[name])
            self.d_heads[name].eval()

        # initialize segmenter
        self.segmenter = SegmenterFace(model_paths['segmenter_faces'], fuse_face_regions=True)

        self.lpips_loss = LPIPS(net_type='vgg').to(self.device).eval()

    def inverse(self, images, images_resize, image_paths, emb_codes, emb_images, emb_info, images_seg):
        with torch.no_grad():
            # segment the target image
            segments = self.segmenter.segment_pil(images_seg)

            # make the invertibility latent map
            d_invmaps = self.invert_predictor(images)
            d_invmaps = {n: self.d_heads[n](d_invmaps) for n in latent_names.split(",")}

        thresh = 0.2
        # refine the invertibility map
        d_refined_invmap = refine(d_invmaps, segments, thresh)
        # resize the masks
        d_refined_resized_invmap = resize_binary_masks(d_refined_invmap)

        # W+ is initialized with e4e encoder outputs
        d_latents_init = {
            "W+": emb_codes.detach().clone().to(self.device),
            "F4": torch.zeros((1, 512, 16, 16)).to(self.device),
            "F6": torch.zeros((1, 512, 32, 32)).to(self.device),
            "F8": torch.zeros((1, 512, 64, 64)).to(self.device),
            "F10": torch.zeros((1, 256, 128, 128)).to(self.device),
        }
        d_latents = {k: d_latents_init[k].detach().clone() for k in d_latents_init}
        for k in d_latents:
            d_latents[k].requires_grad = True
        # define the optimizer
        optimizer = torch.optim.Adam([d_latents[k] for k in d_latents], lr=0.05, betas=(0.9, 0.999))

        # optimization loop
        for i in tqdm.tqdm(range(1001)):
            # learning rate scheduling
            t = i / 1001
            lr_ramp = min(1.0, (1.0 - t) / 0.25)
            lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
            lr_ramp = lr_ramp * min(1.0, t / 0.05)
            lr = 0.05 * lr_ramp
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            log_str = f"[(step {i:04d})]: "
            rec_full, result_latent = self.decoder([d_latents['W+']],
                                                   input_is_latent=True,
                                                   return_latents=True,
                                                   randomize_noise=False,
                                                   sam_masks=d_refined_resized_invmap,
                                                   sam_features=d_latents,
                                                   sam_idxes=[3, 5, 7, 9])

            # compute the reconstruction losses using smaller 256x256 images
            rec = F.interpolate(rec_full, size=(256, 256), mode='area').clamp(-1, 1)

            # image reconstruction losses
            rec_losses = 0.0
            rec_losses += F.mse_loss(rec_full, images)
            rec_losses += self.lpips_loss(rec, images_resize)
            log_str += f"rec: {rec_losses:.3f} "
            # latent regularization
            latent_losses = 0.0

            mvg = compute_mvg(d_latents, "W+", self.d_stats["mean_v"], self.d_stats["inv_cov_v"]) * 1e-8
            latent_losses += mvg
            log_str += f"mvg: {mvg:.3f} "

            delta = delta_loss(d_latents["W+"]) * 1e-3
            latent_losses += delta
            log_str += f"delta: {delta:.3f} "

            frec = F.mse_loss(d_latents["F4"], d_latents_init["F4"]) * 5
            frec += F.mse_loss(d_latents["F6"], d_latents_init["F6"]) * 5
            frec += F.mse_loss(d_latents["F8"], d_latents_init["F8"]) * 5
            frec += F.mse_loss(d_latents["F10"], d_latents_init["F10"]) * 5
            latent_losses += frec
            log_str += f"frec: {frec:.3f} "
            # update the parameters
            optimizer.zero_grad()
            (rec_losses + latent_losses).backward()
            optimizer.step()
            if i % 250 == 0:
                print(log_str)

        with torch.no_grad():
            images, result_latent = self.decoder([d_latents['W+']],
                                                 input_is_latent=True,
                                                 return_latents=True,
                                                 randomize_noise=False,
                                                 sam_masks=d_refined_resized_invmap,
                                                 sam_features=d_latents,
                                                 sam_idxes=[3, 5, 7, 9])
        return images, result_latent, None

    def edit(self, images, images_resize, image_path, editor):
        images, codes, _ = self.inverse(images, images_resize, image_path)
        edit_codes = editor.edit_code(codes)
        edit_images = self.generate(edit_codes)
        return images, edit_images, codes, edit_codes, None


class LayerHead(torch.nn.Module):
    def __init__(self,):
        super(LayerHead, self).__init__()
        self.m = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(8, 4, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(4),
            torch.nn.LeakyReLU(0.2),
            torch.nn.ConvTranspose2d(4, 1, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU()
        )

    def forward(self, x):
        return self.m(x)
