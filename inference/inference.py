class BaseInference(object):
    def __init__(self):
        self.decoder = None

    def inverse(self, images, images_resize, image_name, emb_codes, emb_images):
        pass

    def generate(self, codes):
        return self.decoder([codes], input_is_latent=True, return_latents=False)[0]

    def edit(self):
        pass
