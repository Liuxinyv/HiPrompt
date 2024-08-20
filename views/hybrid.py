from PIL import Image
import numpy as np

import torch
import torchvision.transforms.functional as TF

from .base import BaseView

'''
n.b.
small sigma => low freq is easier to see
larger sigma => high freq easier to see
kernel size must be large enough to avoid edge effects
    (kernel_size = 5 x sigma seems good enough)
'''

def make_frame_hybrid(im, t, resize_factor):
    im_size = im.size[0]
    frame_size = int(im_size * 1.5)
    new_size = int( im_size / (1 + (resize_factor - 1) * t) )

    # Convert to tensor
    im = torch.tensor(np.array(im) / 255.).permute(2,0,1)

    # Resize to new size
    im = TF.resize(im, new_size)

    # Convert back to PIL
    im = (np.array(im.permute(1,2,0)) * 255.)
    im = im.astype(np.uint8)
    im = Image.fromarray(im)

    # Paste on to canvas
    frame = Image.new('RGB', (frame_size, frame_size), (255, 255, 255))
    coords = ((frame_size - new_size) // 2, (frame_size - new_size) // 2)
    frame.paste(im, coords)

    return frame

class HybridLowPassView(BaseView):
    def __init__(self, sigma=2, kernel_size=33):
        self.sigma = sigma
        self.kernel_size = kernel_size

    def make_frame(self, im, t, resize_factor=12):
        return make_frame_hybrid(im, t, resize_factor)

    def view(self, im):
        # For factorized diffusion, we don't change the input to the model
        return im

    def inverse_view(self, noise):
        c, h, w = noise.shape

        # To account for two stages, scale kernel size and sigma
        # based on image size (either 64x64 or 256x256)
        factor = h // 64#
        k = self.kernel_size * factor + ((factor + 1) % 2)
        sigma = self.sigma * factor

        # Low pass noise estimate
        noise[:3] = TF.gaussian_blur(noise[:3], k, sigma)

        return noise


class HybridHighPassView(BaseView):
    def __init__(self, sigma=2,beta=0.95, kernel_size=33):
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.beta = beta

    def make_frame(self, im, t, resize_factor=12):
        return make_frame_hybrid(im, t, resize_factor)

    def view(self, im):
        # For factorized diffusion, we don't change the input to the model
        return im

    def inverse_view(self, noise):
        c, h, w = noise.shape

        # To account for two stages, scale kernel size and sigma
        # based on image size (either 64x64 or 256x256)
        factor = h // 64#h=128,factor=2
        k = self.kernel_size * factor + ((factor + 1) % 2)
        sigma = self.sigma * factor#4.0
        

        loss_pass_noise = TF.gaussian_blur(noise[:3], k, sigma)
        high_pass_noise = (noise[:3] - loss_pass_noise)
        

        high_pass_noise *= self.beta 
        
        noise[:3] = high_pass_noise
        return noise


