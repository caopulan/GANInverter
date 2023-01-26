from .code_infer import CodeInference
from .encoder_infer import EncoderInference
from .optim_infer import OptimizerInference
from .pti_infer import PTIInference


class BaseInference(object):
    def __init__(self, decoder=None, **kwargs):
        self.decoder = decoder

    def inverse(self, images, images_resize, image_paths, **kwargs):
        pass

    def generate(self, codes, **kwargs):
        return self.decoder([codes], input_is_latent=True, return_latents=False)[0]

    def edit(self, codes, edit_info, **kwargs):
        pass


class TwoStageInference(BaseInference):
    def __init__(self, opts, decoder=None):
        super(TwoStageInference, self).__init__()
        # mode in two stages
        embedding_mode = opts.embed_mode
        refinement_mode = opts.refinement_mode
        self.refinement_mode = refinement_mode

        # Image Embedding
        if embedding_mode == 'encoder':
            self.embedding_module = EncoderInference(opts)
        elif embedding_mode == 'optim':
            self.embedding_module = OptimizerInference(opts)
        elif embedding_mode == 'code':
            self.embedding_module = CodeInference(opts)
        else:
            raise Exception(f'Wrong embedding mode: {embedding_mode}.')

        # Result Refinement
        if refinement_mode == 'pti':
            self.refinement_module = PTIInference(opts)
        elif refinement_mode is None:
            pass
        else:
            raise Exception(f'Wrong embedding mode: {embedding_mode}.')

    def inverse(self, images, images_resize, image_paths, **kwargs):
        emb_images, emb_codes, emb_intermediate = self.embedding_module.inverse(images, images_resize, image_paths)
        if self.refinement_mode is not None:
            refine_images, refine_codes, refine_intermediate = \
                self.refinement_module.inverse(images, images_resize, image_paths, emb_codes, emb_images, emb_intermediate)
        else:
            refine_codes, refine_images, refine_intermediate = None, None, None

        return emb_codes, emb_images, emb_intermediate, refine_codes, refine_images, refine_intermediate
