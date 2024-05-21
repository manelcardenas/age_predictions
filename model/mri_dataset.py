import torch
import h5py
import numpy as np

from torch.utils.data import Dataset

class MRIDataset(Dataset):
    def __init__(self, h5_path, keys, age_dist, transform=None):
        self.h5_path = h5_path
        self.keys = keys
        self.age_dist = age_dist
        self.length = len(keys)
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as file:
            subject_id = self.keys[idx]
            subject_group = file[subject_id]
            mri_data = subject_group['MRI'][:]
            # Obtener la edad real del sujeto del archivo HDF5
            age = subject_group.attrs['Age']
            age_tensor = torch.tensor(age).float()  # Convertir a tensor
            
            # Normalización Z de los datos MRI
            mri_data = (mri_data - np.mean(mri_data)) / np.std(mri_data)
            
            # Convertir los datos a tensores de PyTorch
            mri_data_tensor = torch.from_numpy(mri_data).float().unsqueeze(0)  # [C, D, H, W]
            age_dist_tensor = torch.from_numpy(self.age_dist[idx]).float()
        
        if self.transform:
           mri_data_tensor = self.transform(mri_data_tensor) 

        return mri_data_tensor, age_dist_tensor, age_tensor, subject_id