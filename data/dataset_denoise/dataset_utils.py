import torch

class Augment_RGB_torch:
    '''
    Performs dat augmentation of the input image
    Input:
        image: a pytorch tensor image  (C,h,w)
        mode: int. Choice of transformation to apply to the image
                0 - no transformation
                1 - flip up and down
                2 - rotate counterwise 90 degree
                3 - rotate 90 degree and flip up and down
                4 - rotate 180 degree
                5 - rotate 180 degree and flip
                6 - rotate 270 degree
                7 - rotate 270 degree and flip
    '''
    def __init__(self):
        pass
    def transform0(self, torch_tensor):
        return torch_tensor   
    def transform1(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=1, dims=[-1,-2])
        return torch_tensor
    def transform2(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=2, dims=[-1,-2])
        return torch_tensor
    def transform3(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=3, dims=[-1,-2])
        return torch_tensor
    def transform4(self, torch_tensor):
        torch_tensor = torch_tensor.flip(-2)
        return torch_tensor
    def transform5(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=1, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform6(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=2, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform7(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=3, dims=[-1,-2])).flip(-2)
        return torch_tensor


class MixUp_AUG:
    def __init__(self):
        self.dist = torch.distributions.beta.Beta(torch.tensor([1.2]), torch.tensor([1.2]))

    def aug(self, rgb_gt, rgb_noisy):
        bs = rgb_gt.size(0)
        indices = torch.randperm(bs)
        rgb_gt2 = rgb_gt[indices]
        rgb_noisy2 = rgb_noisy[indices]

        lam = self.dist.rsample((bs,1)).view(-1,1,1,1).cuda()

        rgb_gt    = lam * rgb_gt + (1-lam) * rgb_gt2
        rgb_noisy = lam * rgb_noisy + (1-lam) * rgb_noisy2

        return rgb_gt, rgb_noisy

import numpy as np
def burst_image_filter(image):
    img_shape = image.shape
    # print('img_shape: ' + str(img_shape))
    h = img_shape[1]
    w = img_shape[2]
    im1 = image[:,0:h:2, 0:w:2]
    im2 = image[:,0:h:2, 1:w:2]
    im3 =image[:,1:h:2, 1:w:2]
    im4 =image[:,1:h:2, 0:w:2]

    img_arithmetic_mean = (im1 +im2+im3+im4) / 4
    img_geometric_mean = torch.pow(im1 *im2*im3*im4,1 / 4)
    img_harmonic_mean = 4 / (1 / im1 + 1 / im2 + 1 / im3 + 1 / im4)
    img_harmonic_mean = torch.clamp(img_harmonic_mean - 1, 0, 255)

    burst = torch.stack((im1.unsqueeze(0), im2.unsqueeze(0), im3.unsqueeze(0), im4.unsqueeze(0)), dim=0)
    img_min = torch.min(burst, dim=0)[0]
    img_max = torch.max(burst, dim=0)[0]

    img_midpoint_filter = (img_min + img_max) / 2
    return torch.cat((img_arithmetic_mean.unsqueeze(0),img_geometric_mean.unsqueeze(0),img_harmonic_mean.unsqueeze(0),img_midpoint_filter),dim=0)

def arithmetic_mean(img):
    assert len(img.shape) == 3
    img = img.astype(np.float32)
    H, W = img.shape[:2]

    img_mean = (img[0:H:2, 0:W:2] + img[0:H:2, 1:W:2] +
                img[1:H:2, 0:W:2] + img[1:H:2, 1:W:2]) / 4

    return img_mean


def geometric_mean(img):
    assert len(img.shape) == 3
    img = img.astype(np.float32)
    H, W = img.shape[:2]

    img_mean = np.power(img[0:H:2, 0:W:2] * img[0:H:2, 1:W:2] *
                        img[1:H:2, 0:W:2] * img[1:H:2, 1:W:2],
                        1/4)

    return img_mean


def harmonic_mean(img):
    assert len(img.shape) == 3
    img = img.astype(np.float32) + 1
    H, W = img.shape[:2]

    img_mean = 4 / (1 / img[0:H:2, 0:W:2] + 1 / img[0:H:2, 1:W:2] +
                    1 / img[1:H:2, 0:W:2] + 1 / img[1:H:2, 1:W:2])
    img_mean = np.clip(img_mean - 1, 0, 255)
    return img_mean


def midpoint_filter(img):
    assert len(img.shape) == 3
    img = img.astype(np.float32)
    H, W = img.shape[:2]

    burst = np.stack([img[0:H:2, 0:W:2], img[0:H:2, 1:W:2],
                    img[1:H:2, 0:W:2], img[1:H:2, 1:W:2]], axis=1)
    img_min = np.min(burst, axis=1)
    img_max = np.max(burst, axis=1)

    filtered_img = (img_min + img_max) / 2
    return filtered_img


def median_filter(img):
    assert len(img.shape) == 3
    img = img.astype(np.float32)
    H, W = img.shape[:2]

    burst = np.stack([img[0:H:2, 0:W:2], img[0:H:2, 1:W:2],
                    img[1:H:2, 0:W:2], img[1:H:2, 1:W:2]], axis=1)

    filtered_img = np.median(burst, axis=1)
    return filtered_img


def max_filter(img):
    assert len(img.shape) == 3
    img = img.astype(np.float32)
    H, W = img.shape[:2]

    burst = np.stack([img[0:H:2, 0:W:2], img[0:H:2, 1:W:2],
                    img[1:H:2, 0:W:2], img[1:H:2, 1:W:2]], axis=1)

    return np.max(burst, axis=1)


def min_filter(img):
    assert len(img.shape) == 3
    img = img.astype(np.float32)
    H, W = img.shape[:2]

    burst = np.stack([img[0:H:2, 0:W:2], img[0:H:2, 1:W:2],
                    img[1:H:2, 0:W:2], img[1:H:2, 1:W:2]], axis=1)
    return np.min(burst, axis=1)
