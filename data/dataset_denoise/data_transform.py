import torchvision.transforms.functional as TF

def random_flip(image,rand_hflip,rand_vflip):
    # Random horizontal flipping
    # print(rand_hflip, rand_vflip)
    if rand_hflip < 0.2:
        # print("rand_hflip")
        image = TF.hflip(image)
    # Random vertical flipping
    if rand_vflip < 0.2:
        # print("rand_vflip")
        image = TF.vflip(image)
    return image

def random_rotate(image, rand_affine, angle):
    # print(rand_affine)
    if rand_affine < 0.1:
        image = TF.rotate(image,angle=angle)
    return image