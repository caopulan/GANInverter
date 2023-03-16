from .base_editing import BaseEditing
import torch
import os


class InterFaceGAN(BaseEditing):
    def __init__(self, opts):
        super(InterFaceGAN, self).__init__()
        self.opts = opts
        self.edit_vector = torch.load(opts.edit_path, map_location='cpu').cuda()
        self.factor = opts.edit_factor

        if opts.edit_save_path == '':
            self.save_folder = f'{os.path.basename(opts.edit_path).split(".")[0]}_{self.factor}'
        else:
            self.save_folder = opts.edit_save_path

        if self.edit_vector.dim() == 2:
            self.edit_vector = self.edit_vector[None]
        # elif self.edit_vector.dim() == 3:
        #     self.edit_vector = self.edit_vector[None]

    def edit_code(self, code):
        return code + self.edit_vector * self.factor
