import torch.utils.data as data
import torch
import numpy as np
from .build import DATASETS
from .io import IO

@DATASETS.register_module()
class MyCustomDataset(data.Dataset):
    def __init__(self, config):
        # The 'subset' variable is passed from the main pretrain.yaml
        self.subset = config.subset 
        
        if self.subset == 'train':
            self.data_path = config.TRAIN_DATA_PATH
        else:
            self.data_path = config.VAL_DATA_PATH
            
        self.all_points = IO.get(self.data_path)
        print(f'[CustomDataset] Subset: {self.subset} | Loaded {len(self.all_points)} samples.')

    def __getitem__(self, index):
        """
        Returns a single point cloud sample.
        """
        # 1. Get the points for the specific index (Shape: 300, 3)
        points = self.all_points[index]
        
        # 2. Convert the numpy array to a Torch FloatTensor
        # Point-MAE expects float32 coordinates
        pt_coords = torch.from_numpy(points).float()
        
        # 3. The model expects (Sample_ID, Coordinates)
        # We can generate a simple ID based on the index
        sample_id = f'sample_{index}'
        
        return sample_id, pt_coords

    def __len__(self):
        """
        Returns the total number of objects (20,000).
        """
        return self.all_points.shape[0]