from .code_infer import CodeInference
from .encoder_infer import EncoderInference
from .optim_infer import OptimizerInference
from .pti_infer import PTIInference

class TwoStageInference:
    def __init__(self, opts):
        # mode in two stages
        embedding_mode = opts.embedding_mode
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

    def inverse(self, images, images_resize, images_path):
        emb_images, emb_codes = self.embedding_module.inverse(images, images_resize, images_path)
        if self.refinement_mode is not None:
            refine_images, refine_codes = self.refinement_module.inverse(images, images_resize, images_path,
                                                                         emb_codes, emb_images)
            return emb_codes, emb_images, refine_codes, refine_images
        else:
            return emb_codes, emb_images, None, None
