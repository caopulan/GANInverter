from .base_editing import BaseEditing
import torch


class GANSpace(BaseEditing):
    def __init__(self, opts):
        super(GANSpace, self).__init__()
        ganspace_pca = torch.load(opts.edit_path, map_location='cpu')
        self.pca_idx, self.start, self.end, self.strength = opts.ganspace_directions
        self.code_mean = ganspace_pca['mean'].cuda()
        self.code_comp = ganspace_pca['comp'].cuda()[self.pca_idx]
        self.code_std = ganspace_pca['std'].cuda()[self.pca_idx]
        self.save_folder = f'ganspace_{self.pca_idx}_{self.start}_{self.end}_{self.strength}'

    def edit_code(self, code):
        edit_codes = []
        for c in code:
            w_centered = c - self.code_mean
            w_coord = torch.sum(w_centered[0].reshape(-1) * self.code_comp.reshape(-1)) / self.code_std
            delta = (self.strength - w_coord) * self.code_comp * self.code_std
            delta_padded = torch.zeros(c.shape).to('cuda')
            delta_padded[self.start:self.end] += delta.repeat(self.end - self.start, 1)
            edit_codes.append(c + delta_padded)
        return torch.stack(edit_codes)