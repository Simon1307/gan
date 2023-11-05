import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import random
from PIL import Image
import torchvision.utils as vutils


class ImageDataset(Dataset):
    """ Custom Dataset class """
    def __init__(self, root: str, transform: transforms.Compose):
        self.root = root
        self.transform = transform
        self.all_imgs = tuple(os.path.join(root, p) for p in os.listdir(root) if p.endswith('.jpg'))

    def __len__(self) -> int:
        return len(self.all_imgs)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.transform(self.load_image(self.all_imgs[idx]))
    
    @staticmethod
    def load_image(path: str) -> np.ndarray:
        with Image.open(path) as p:
            img = np.asarray(p)
        return img


    def plot_random_images(self, n_img: int) -> None:
        # Sample random number of images
        sampled_images = random.sample(self.all_imgs, n_img)
        
        images_list = list(np.transpose(torch.from_numpy(self.load_image(p)), (2, 0, 1)) for p in sampled_images)
        image_grid = vutils.make_grid(images_list)
        plt.figure(figsize=(20, 20))
        plt.imshow(np.transpose(image_grid, (1, 2, 0)))
        plt.axis('off')
        plt.show()


class CombinedDataset(Dataset):
    def __init__(self, dataset_A, dataset_B):
        self.dataset_A = dataset_A
        self.dataset_B = dataset_B
        self.len_A = len(dataset_A)
        self.len_B = len(dataset_B)
        self.max_len = max(self.len_A, self.len_B)
    
    def __len__(self):
        return self.max_len
    
    def __getitem__(self, idx):
        idx_A = idx % self.len_A
        idx_B = idx % self.len_B

        sample_A = self.dataset_A[idx_A]
        sample_B = self.dataset_B[idx_B]

        return {'A': sample_A, 'B': sample_B}
