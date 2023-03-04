from .code_infer import CodeInference
from .encoder_infer import EncoderInference
from .optim_infer import OptimizerInference
from .pti_infer import PTIInference
from .dhr_infer import DHRInference
from .sam_infer import SamInference
from inference.inference import BaseInference


class TwoStageInference(BaseInference):
    def __init__(self, opts, decoder=None):
        super(TwoStageInference, self).__init__()
        # mode in two stages
        embed_mode = opts.embed_mode
        refine_mode = opts.refine_mode
        self.refine_mode = refine_mode

        # Image Embedding
        if embed_mode == 'encoder':
            self.embedding_module = EncoderInference(opts)
        elif embed_mode == 'optim':
            self.embedding_module = OptimizerInference(opts)
        elif embed_mode == 'code':
            self.embedding_module = CodeInference(opts)
        else:
            raise Exception(f'Wrong embedding mode: {embed_mode}.')

        # Result Refinement
        if refine_mode == 'pti':
            self.refinement_module = PTIInference(opts)
        elif refine_mode == 'dhr':
            self.refinement_module = DHRInference(opts)
        elif refine_mode == 'sam':
            self.refinement_module = SamInference(opts)
        elif refine_mode is None:
            self.refinement_module = None
        else:
            raise Exception(f'Wrong embedding mode: {refine_mode}.')

    def inverse(self, images, images_resize, image_paths, images_seg, **kwargs):
        emb_images, emb_codes, emb_info = self.embedding_module.inverse(images, images_resize, image_paths)
        if self.refine_mode is not None:
            refine_images, refine_codes, refine_info = \
                self.refinement_module.inverse(images, images_resize, image_paths, images_seg, emb_codes, emb_images, emb_info)
        else:
            refine_images, refine_codes, refine_info = None, None, None

        return emb_images, emb_codes, emb_info, refine_images, refine_codes, refine_info

    def edit(self, images, images_resize, image_paths, editor):
        emb_codes, emb_codes_edit, emb_images, emb_images_edit, emb_info, \
        refine_codes, refine_codes_edit, refine_images, refine_images_edit, refine_info = [None] * 10
        emb_images, emb_images_edit, emb_codes, emb_codes_edit, emb_info = \
            self.embedding_module.edit(images, images_resize, image_paths, editor)
        if self.refine_mode is not None:
            refine_images, refine_images_edit, refine_codes, refine_codes_edit, refine_info = \
                self.refinement_module.inverse(images, images_resize, image_paths, emb_codes, emb_images, emb_info, editor)

        return emb_images_edit, emb_codes_edit, emb_info, refine_images_edit, refine_codes_edit, refine_info
