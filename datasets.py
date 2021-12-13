import h5py
import numpy as np
import torchvision
from torchvision.datasets import VisionDataset

from typing import Any, Callable, Optional 


class TrainDataset(VisionDataset):
    def __init__(
        self, 
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None      
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

    def __getitem__(self, idx):
        with h5py.File(self.root, 'r') as f:
            img, target = np.expand_dims(f['lr'][idx] / 255., 0), np.expand_dims(f['hr'][idx] / 255., 2) # normailization
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return  img, target 

    def __len__(self):
        with h5py.File(self.root, 'r') as f:
            return len(f['lr'])


class EvalDataset(VisionDataset):
    def __init__(
        self, 
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None      
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

    def __getitem__(self, idx):
        with h5py.File(self.root, 'r') as f:
            img, target = np.expand_dims(f['lr'][str(idx)][:, :] / 255., 0), np.expand_dims(f['hr'][str(idx)][:, :] / 255., 2) 
            # 0-4 하위 그룹 존재
            # 5개 모두 다른 shape를 가짐(논문 참조)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            self.target_transform_cropped = self.target_transform.crop(target.shape[0],target.shape[1])
            target = self.target_transform_cropped(target)

        return  img, target 

    def __len__(self):
        with h5py.File(self.root, 'r') as f:
            return len(f['lr'])
