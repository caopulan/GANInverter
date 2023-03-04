import torch
from torchvision.transforms import transforms

from models.bisenet.model import BiSeNet


class SegmenterFace:
    def __init__(self, ckpt_path="ckpt/79999_iter.pth", fuse_face_regions=True):

        self.net = BiSeNet(n_classes=19).cuda()
        self.net.load_state_dict(torch.load(ckpt_path))
        self.net.eval()
        self.fuse_face_regions = fuse_face_regions

    def segment_pil(self, img_pil):
        out = self.net(img_pil)[0]
        parsed = out.squeeze(0).detach().cpu().numpy().argmax(0)
        if self.fuse_face_regions:
            """
            1 - skin
            2/3 - left/right brow
            4/5 - left/right eye
            7/8 - left/right ear
            10 - nose
            11 - mouth
            12/13 - upper/lower lips
            14 - neck
            17 - hair
            """
            for idx in [1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 17]:
                parsed[parsed == idx] = 3
        return parsed
