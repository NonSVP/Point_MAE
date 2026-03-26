import os
import vtk
import pickle
import torch
import numpy as np
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from dataset_SATO.serialize import Point

from torch.utils.data import Dataset

import h5py


def pos_to_order_inverse_index(pos, tensor=False):
    device = pos.device
    
    if pos.dim() == 2:
        pos = pos.unsqueeze(0)
    
    B, N, C = pos.shape
    
    # 1. Prepare data for SATO Point serialization
    data_dict = dict(coord=pos, grid_size=torch.tensor([0.05, 0.05, 0.05]).to(device))
    data_dict['batch'] = torch.arange(B).repeat_interleave(N).to(device)
    data_dict['coord'] = data_dict['coord'].view(B*N, C).to(device)
    
    # 2. Compute Serialization (Z-order / Hilbert)
    point = Point(data_dict)
    point.serialization(order=["z", "z-trans", "hilbert", "hilbert-trans"], shuffle_orders=False)
    
    # 3. Extract Raw Global Indices (Shape: [4, B*N])
    raw_order = point['serialized_order']    # Indices from 0 to B*N-1
    raw_inverse = point['serialized_inverse']
    
    # 4. Reshape and Localize Indices
    # We want shape (B, 4, N) with indices in range [0, N-1]
    
    # Reshape: (4, B*N) -> (4, B, N)
    order_reshaped = raw_order.view(4, B, N)
    inverse_reshaped = raw_inverse.view(4, B, N)
    
    # Modulo N to convert global indices (e.g., 205) to local indices (e.g., 5)
    # This works because the sorting logic strictly separates batches (Batch 0 comes before Batch 1)
    order_local = order_reshaped % N
    inverse_local = inverse_reshaped % N
    
    # Permute to match Model Expectation: (B, 4, N)
    order_final = order_local.permute(1, 0, 2)
    inverse_final = inverse_local.permute(1, 0, 2)
    
    if not tensor:
        return order_final.cpu().numpy(), inverse_final.cpu().numpy()
    else:
        return order_final, inverse_final

def sato_collate_fn(batch):
    lengths = [item['x'].shape[0] for item in batch]
    min_len = min(lengths)
    
    batch_x = []
    batch_y = []
    
    for item in batch:
        x = item['x']
        y = item['y']
        
        if x.shape[0] > min_len:
            # Randomly sample min_len indices
            idx = torch.randperm(x.shape[0])[:min_len]
            x = x[idx]
            y = y[idx]
            
        batch_x.append(x)
        batch_y.append(y)
        
    return {
        'x': torch.stack(batch_x),
        'y': torch.stack(batch_y)
    }


class SATO_Dataset(Dataset):
    def __init__(self, data_list, config=None, is_train=True):
        self.data_list = data_list
        self.config = config
        self.is_train = is_train

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        x = data['Surface_data']['Surface_points']
        y = data['Surface_data']['Surface_pressure']

        # Move downsample logic here to allow parallel processing by DataLoader workers
        if self.config is not None and hasattr(self.config.model, 'down_sample'):
            # Use numpy random generation which is process-safe in workers
            num_points = x.shape[0]
            sample_size = int(num_points * self.config.model.down_sample)
            
            # Generate indices (on CPU)
            sampled_indices = np.random.choice(num_points, sample_size, replace=False)
            
            x = x[sampled_indices]
            y = y[sampled_indices]

        return {'x': x, 'y': y}


class VTKDataset():
    def __init__(self):
        pass

    def get_all_file_paths(self, directory):
        file_paths = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_paths.append(os.path.join(root, file))
        return file_paths

    # generate data dictionary
    def get_data_dict(self, directory):
        # read all SurfacePressure file names
        SurfacePressure_file_paths = self.get_all_file_paths(os.path.join(directory, 'SurfacePressure', 'VTK'))

        # load train/test/val index
        with open(os.path.join(directory, 'train_val_test_splits/train_design_ids.txt'), 'r') as file:
            train_index = [line.strip()[-4:] for line in file]
        with open(os.path.join(directory, 'train_val_test_splits/test_design_ids.txt'), 'r') as file:
            test_index = [line.strip()[-4:] for line in file]
        with open(os.path.join(directory, 'train_val_test_splits/val_design_ids.txt'), 'r') as file:
            val_index = [line.strip()[-4:] for line in file]

        with open(os.path.join(directory, 'norm', 'mean.pkl'), 'rb') as f:
            mean_data = pickle.load(f)
        with open(os.path.join(directory, 'norm', 'std.pkl'), 'rb') as f:
            std_data = pickle.load(f)

        train_data_lst, test_data_lst, val_data_lst = [], [], []
        for file_path in SurfacePressure_file_paths:
            index = file_path[-8:-4]
            Surface_points = np.load(os.path.join(directory, 'SurfacePressure', 'points', f'points_{index}.npy'))
            Surface_pressure = np.load(os.path.join(directory, 'SurfacePressure', 'pressure', f'pressure_{index}.npy'))

            Surface_points = torch.Tensor(Surface_points).float()
            Surface_pressure = torch.Tensor(Surface_pressure).float()

            Surface_data = {
                'Surface_points': Surface_points,
                'Surface_pressure': Surface_pressure
            }

            data = {'index': index, 'Surface_data': Surface_data}

            if index in train_index:
                train_data_lst.append(data)
            elif index in test_index:
                test_data_lst.append(data)
            else:
                val_data_lst.append(data)

        return train_data_lst, test_data_lst, val_data_lst, mean_data, std_data
    




class PressureHDF5Dataset(Dataset):
    def __init__(self, h5_path, config=None, is_train=True):
        self.h5_path = h5_path
        
        with h5py.File(h5_path, 'r') as f:
            self.data_shape = f['data'].shape
        
        # Shape: (1250, 40, 200, 12) -> Flatten first two dims
        self.n_sims = self.data_shape[0]
        self.n_steps = self.data_shape[1]
        self.n_points = self.data_shape[2]
        self.total_samples = self.n_sims * self.n_steps

        # Compute Stats (Per Channel)
        with h5py.File(h5_path, 'r') as f:
            # Use first 10 sims (400 samples) for stats
            subset = f['data'][:10].reshape(-1, self.n_points, 12)
            
            # INPUT: XREL (Cols 0, 1, 2)
            self.mean_pos = np.mean(subset[..., 0:3], axis=(0, 1))
            self.std_pos  = np.std(subset[..., 0:3], axis=(0, 1))
            
            # TARGET: PRESSURE (Col 9)
            self.mean_p   = np.mean(subset[..., 9:10], axis=(0, 1))
            self.std_p    = np.std(subset[..., 9:10], axis=(0, 1))

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        sim_idx = idx // self.n_steps
        step_idx = idx % self.n_steps
        
        with h5py.File(self.h5_path, 'r') as f:
            data = f['data'][sim_idx, step_idx] # (200, 12)
            
            # Input: Geometry only (X, Y, Z)
            x_input = data[:, 0:3] 
            
            # Target: Pressure
            y_target = data[:, 9:10]
            
            return {
                'x': torch.from_numpy(x_input).float(),        # (200, 3)
                'y': torch.from_numpy(y_target).float().squeeze(-1) # (200,)
            }

    def get_stats(self):
        # Return standard dictionary structure
        return (
            {'Surface_mean': {
                'Surface_points': torch.from_numpy(self.mean_pos).float(), 
                'Surface_pressure': torch.from_numpy(self.mean_p).float()
             }},
            {'Surface_std': {
                'Surface_points': torch.from_numpy(self.std_pos).float(), 
                'Surface_pressure': torch.from_numpy(self.std_p).float()
            }}    
        )
    

    

class TimeStepPressureDataset(Dataset):
    def __init__(self, h5_path, config=None, is_train=True):
        self.h5_path = h5_path
        
        # 1. Open file to get shapes
        with h5py.File(h5_path, 'r') as f:
            self.data_shape = f['data'].shape
            
        self.n_sims = self.data_shape[0]
        self.n_steps = self.data_shape[1]
        self.n_points = self.data_shape[2]
        self.total_samples = self.n_sims * self.n_steps

        # 2. Compute Stats (CORRECTED)
        # We calculate mean per-channel so shapes match [x,y,z,p]
        with h5py.File(h5_path, 'r') as f:
            # Take a larger subset for better estimation (e.g., first 10 simulations)
            # Flatten time steps: (10 * 40, 200, 12)
            subset = f['data'][:10].reshape(-1, self.n_points, 12)
            
            # XREL_next (Indices 6,7,8) -> Shape (3,)
            self.mean_pos = np.mean(subset[..., 6:9], axis=(0, 1))
            self.std_pos  = np.std(subset[..., 6:9], axis=(0, 1))
            
            # PRESSURE (Index 9) -> Shape (1,)
            self.mean_p   = np.mean(subset[..., 9:10], axis=(0, 1))
            self.std_p    = np.std(subset[..., 9:10], axis=(0, 1))
            
            # PRESSURE_next Target (Index 11) -> Shape (1,)
            self.mean_tgt = np.mean(subset[..., 11:12], axis=(0, 1))
            self.std_tgt  = np.std(subset[..., 11:12], axis=(0, 1))

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        sim_idx = idx // self.n_steps
        step_idx = idx % self.n_steps
        
        with h5py.File(self.h5_path, 'r') as f:
            data = f['data'][sim_idx, step_idx] # (200, 12)
            
            # Input: [X_next, Y_next, Z_next, P_t]
            pos = data[:, 6:9]   # (200, 3)
            feat_p = data[:, 9:10] # (200, 1)
            x_input = np.hstack([pos, feat_p]) # (200, 4)
            
            # Target: P_t+1
            y_target = data[:, 11:12]
            
            return {
                'x': torch.from_numpy(x_input).float(),
                'y': torch.from_numpy(y_target).float().squeeze(-1)
            }

    def get_stats(self):
        # Concatenate: [Mean_X, Mean_Y, Mean_Z] + [Mean_P] = [Mean_4D]
        combined_mean = np.hstack([self.mean_pos, self.mean_p])
        combined_std  = np.hstack([self.std_pos, self.std_p])
        
        return (
            {'Surface_mean': {
                'Surface_points': torch.from_numpy(combined_mean).float(), 
                # Ensure Target mean is a Tensor
                'Surface_pressure': torch.from_numpy(self.mean_tgt).float()
             }},
            {'Surface_std': {
                'Surface_points': torch.from_numpy(combined_std).float(), 
                # Ensure Target std is a Tensor
                'Surface_pressure': torch.from_numpy(self.std_tgt).float()
            }}    
        )