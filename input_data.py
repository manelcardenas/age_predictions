from m_utils.data_transform import num2vect
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
#import sklearn
from sklearn.model_selection import train_test_split

class MRIDataset(Dataset):
    def __init__(self, h5_path, keys, age_dist):
        self.h5_path = h5_path
        self.keys = keys
        self.age_dist = age_dist
        self.length = len(keys)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as file:
            subject_id = self.keys[idx]
            subject_group = file[subject_id]
            mri_data = subject_group['MRI'][:]
            
            # Normalización Z de los datos MRI
            mri_data = (mri_data - np.mean(mri_data)) / np.std(mri_data)
            
            # Convertir los datos a tensores de PyTorch
            mri_data_tensor = torch.from_numpy(mri_data).unsqueeze(0).float()  # Añade un canal
            age_dist_tensor = torch.from_numpy(self.age_dist[idx]).float()
            
        return mri_data_tensor, age_dist_tensor

# Ruta al archivo .h5 de mujeres
h5_path = '/home/usuaris/imatge/joan.manel.cardenas/females_data.h5'
#h5_path = '/home/usuaris/imatge/joan.manel.cardenas/males_data.h5'

with h5py.File(h5_path, 'r') as h5_file:
    sigma = len(h5_file.keys())  # Número total de sujetos
    keys = list(h5_file.keys())
    ages = [h5_file[subject_id].attrs['Age'] for subject_id in keys]

# Calcular las distribuciones de edad
age_array = np.array(ages)
age_min, age_max = int(np.floor(min(ages))), int(np.ceil(max(ages)))
age_range = [age_min, age_max]
age_step = 1  # Paso de edad
print(f"Número total de sujetos (sigma): {sigma}")
print(f"Rango de edad (age_range): {age_range}")
print(f"Edades después de la conversión a np.array: {age_array}")
age_dist, bin_centers = num2vect(age_array, age_range, age_step, sigma)
print(f"Distribuciones de probabilidad para las edades: {age_dist}")
print(f"Centros de los bins: {bin_centers}")


# Dividir los datos
keys_train, keys_test, age_dist_train, age_dist_test = train_test_split(keys, age_dist, test_size=0.2, random_state=42)  #, stratify=age_dist
keys_train, keys_val, age_dist_train, age_dist_val = train_test_split(keys_train, age_dist_train, test_size=0.25, random_state=42) #, stratify=age_dist_train

# Crear instancias del dataset para cada subconjunto
dataset_train = MRIDataset(h5_path, keys_train, age_dist_train)
dataset_val = MRIDataset(h5_path, keys_val, age_dist_val)
dataset_test = MRIDataset(h5_path, keys_test, age_dist_test)


'''
y = torch.tensor(y, dtype=torch.float32)
print(f'Label shape: {y.shape}')
'''

