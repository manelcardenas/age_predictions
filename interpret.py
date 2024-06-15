import torch
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from collections import OrderedDict

from m_utils.data_transform import num2vect
from model.mri_dataset import MRIDataset
from model.model import CNNmodel
from captum.attr import GuidedGradCam

# Directorio y archivo HDF5 con los datos
h5_path = '/mnt/work/datasets/UKBiobank/MN_females_data.h5'
save_dir = 'subjects_data'
os.makedirs(save_dir, exist_ok=True)

# Leer las edades de los sujetos desde el archivo HDF5
with h5py.File(h5_path, 'r') as h5_file:
    keys = list(h5_file.keys())
    ages = [h5_file[subject_id].attrs['Age'] for subject_id in keys]

# Definir los parámetros para la transformación num2vect
age_range = [42, 82]
age_step = 1
sigma = 1

# Preparar las distribuciones de edad y los centros de los bins
age_dist_list = []
bin_center_list = []

for age in ages:
    age_array = np.array([age, ])
    age_dist, bc = num2vect(age_array, age_range, age_step, sigma)
    age_dist_list.append(age_dist)
    bin_center_list.append(bc)

# Convertir a numpy arrays para manejo posterior
age_dist_array = np.array(age_dist_list)

# Dividir los datos en conjuntos de entrenamiento y prueba
keys_train_val, keys_test, age_dist_train_val, age_dist_test = train_test_split(keys, age_dist_array, test_size=0.2, random_state=42)

# Dataset y DataLoader para el conjunto de prueba
dataset_test = MRIDataset(h5_path, keys_test, age_dist_test)
test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

print(f'Número de muestras en el conjunto de prueba: {len(dataset_test)}')

# Modelo y carga de pesos preentrenados
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNmodel().to(device)
state_dict = torch.load('best_models/best_model_female_DA.p')
new_state_dict = OrderedDict()

for k, v in state_dict.items():
    name = k[7:]  # Eliminar el prefijo 'module.'
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)

# Select the target layer for Guided Grad-CAM
target_layer = model.feature_extractor.conv_5  # Adjust this to your model's specific layer

# Initialize Guided Grad-CAM
guided_grad_cam = GuidedGradCam(model, target_layer)

# Visualization setup
fig, axs = plt.subplots(2, 1, figsize=(10, 5))

with torch.no_grad():
    for inputs, age_dist, age_real, subject_ids in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        
        # Compute Guided Grad-CAM attributions
        attributions = guided_grad_cam.attribute(inputs, target=0)  # Adjust target as necessary
        
        # Convertir la imagen de entrada y las atribuciones a numpy arrays
        input_image_np = inputs.cpu().data[0].numpy().transpose(1, 2, 0)  # Assuming shape is (C, H, W)
        attributions_np = attributions.cpu().data[0].numpy().squeeze()  # Assuming shape is (H, W)

        # Visualizar la imagen de entrada y las atribuciones de Guided Grad-CAM
        axs[0].imshow(input_image_np, cmap='gray')  # Ajustar el cmap según corresponda
        axs[1].imshow(attributions_np, cmap='jet', alpha=0.5)  # Overlay de las atribuciones
        plt.show()

        break  # Visualizamos solo la primera muestra por simplicidad
