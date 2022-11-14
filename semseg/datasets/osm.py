import torch 
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from pathlib import Path
from typing import Tuple
import cv2


class OSM(Dataset):
    CLASSES = [
        'Impervious surface', 'Building', 'Low vegetation', 'Tree', 'Car', 'Clutter'
    ]

    PALETTE = torch.tensor([
        [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3], [120, 120, 80], 
    ])

    def __init__(self, root: str, split: str = 'train', transform = None) -> None:
        super().__init__()
        assert split in ['train', 'val']
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = -1

        if split == 'train':
            img_path = Path(root) / 'osm_split_img'
        else:
            img_path = Path(root) / 'osm_split_img_test'

        self.files = list(img_path.glob('*.tif'))
    
        if not self.files:
            raise Exception(f"No images found in {img_path}")
        print(f"Found {len(self.files)} {split} images.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        img_path = str(self.files[index])
        lbl_path = str(self.files[index]).replace('osm_split_img', 'osm_split_label').replace('.tif', '.png')
        osm_path = str(self.files[index]).replace('osm_split_img', 'osm_split_osm_label').replace('.tif', '.png')

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.tensor(image).permute(2,0,1)
        label = io.read_image(lbl_path).type(torch.float32)
        osm_path = io.read_image(lbl_path).type(torch.float32)
        
        if self.transform:
            image, label, osm = self.transform(image, label, osm_path)
        return image, label.squeeze().long(), osm


if __name__ == '__main__':
    from semseg.utils.visualize import visualize_dataset_sample
    visualize_dataset_sample(OSM, '/home/sithu/datasets/ADEChallenge/ADEChallengeData2016')