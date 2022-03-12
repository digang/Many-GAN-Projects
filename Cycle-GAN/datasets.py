import glob
import random
import os

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transforms = transforms.Compose(transforms_)
        self.unaligned = unaligned
        
        # ex) root/mode/A/zebra_01.png
        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))
    def __getitem__(self, index):
        item_A = self.transforms(Image.open(self.files_A[index % len(self.files_A)]))
        
        if self.unaligned:
            item_B = self.transforms(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transforms(Image.open(self.files_B[index % len(self.files_B)]))
        
        return {'A' : item_A, 'B' : item_B}
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))