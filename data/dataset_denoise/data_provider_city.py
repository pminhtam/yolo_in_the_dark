
from data.dataset_denoise.data_provider import *
##


class SingleLoader(data.Dataset):
    """
    Args:

     Attributes:
        noise_path (list):(image path)
    """

    def __init__(self, noise_dir, gt_dir, image_size=512):

        self.noise_dir = noise_dir
        self.gt_dir = gt_dir
        self.image_size = image_size
        self.noise_path = []
        for files_ext in IMG_EXTENSIONS:
            self.noise_path.extend(glob.glob(self.noise_dir + "/**/*" + files_ext, recursive=True))
        self.gt_path = []
        for files_ext in IMG_EXTENSIONS:
            self.gt_path.extend(glob.glob(self.gt_dir + "/**/*" + files_ext, recursive=True))

        if len(self.noise_path) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + self.noise_dir + "\n"
                                                                                       "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.transforms = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, groundtrue) where image is a noisy version of groundtrue
        """
        # rand_hflip = torch.rand(1)[0]
        # rand_vflip = torch.rand(1)[0]
        image_noise = Image.open(self.noise_path[index]).convert('RGB')
        # name_image_gt = self.noise_path[index].split("/")[-1].replace("NOISY_", "GT_")
        # image_folder_name_gt = self.noise_path[index].split("/")[-2].replace("NOISY_", "GT_")
        # image_gt = Image.open(os.path.join(self.gt_dir, image_folder_name_gt, name_image_gt)).convert('RGB')
        # image_gt = Image.open(self.noise_path[index].replace("noise30",'gt')).convert('RGB')
        image_gt = Image.open(os.path.join(self.gt_dir, self.noise_path[index].split("/")[-1])).convert('RGB')
        # image_noise = random_flip(image_noise,rand_hflip,rand_vflip)
        # image_gt = random_flip(image_gt,rand_hflip,rand_vflip)
        image_noise = self.transforms(image_noise)
        image_gt = self.transforms(image_gt)
        image_noise, image_gt = random_cut(image_noise, image_gt, w=self.image_size)
        apply_trans = transforms_aug[random.getrandbits(3)]

        image_gt = getattr(augment, apply_trans)(image_gt)
        image_noise = getattr(augment, apply_trans)(image_noise)

        return image_noise, image_gt

    def __len__(self):
        return len(self.noise_path)



