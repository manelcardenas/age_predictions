import os
import torch
import pandas as pd
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import LayerGradCam, GuidedBackprop
from collections import OrderedDict
from model.model import CNNmodel  # Ensure this import is correct
import torch.nn.functional as F
from scipy.ndimage import rotate

import torch
import h5py
import os
import numpy as np

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from model.loss import my_KLDivLoss
from model.model import CNNmodel
from m_utils.data_transform import num2vect
from model.mri_dataset import MRIDataset

h5_path = '/mnt/work/datasets/UKBiobank/MN_males_data.h5'

save_dir = 'subjects_data'
os.makedirs(save_dir, exist_ok=True)

with h5py.File(h5_path, 'r') as h5_file:
    keys = list(h5_file.keys())
    ages = [h5_file[subject_id].attrs['Age'] for subject_id in keys]

age_range = [42,82]
age_step = 1  # Paso de edad
sigma = 1
age_dist_list = []
bin_center_list = []

for age in ages:
    age_array = np.array([age,])
    age_dist, bc = num2vect(age_array, age_range, age_step, sigma)
    age_dist_list.append(age_dist)
    bin_center_list.append(bc)

# Convertir la lista de distribuciones a un arreglo de numpy para facilitar el manejo posterior
age_dist_array = np.array(age_dist_list)    


# Dividir los datos
keys_train_val, keys_test, age_dist_train_val, age_dist_test = train_test_split(keys, age_dist_array, test_size=0.2, random_state=42)


dataset_test = MRIDataset(h5_path, keys_test, age_dist_test)

#dataloader
test_loader = DataLoader(dataset_test, batch_size=8, shuffle=False, num_workers=2, pin_memory=True) #DROP_LAST 

print(f'Number of samples in the test set: {len(dataset_test)}')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNmodel()
# Cargar el state_dict guardado
state_dict = torch.load('best_models/best_model_male_DA.p')

# Crear un nuevo state_dict en el que las claves no tienen el prefijo "module."
from collections import OrderedDict
new_state_dict = OrderedDict()

for k, v in state_dict.items():
    name = k[7:]  # eliminar el prefijo 'module.'
    new_state_dict[name] = v

# Cargar el nuevo state_dict en el modelo
model.load_state_dict(new_state_dict)

# 3. Create an instance of LayerGradCam
layer_grad_cam = LayerGradCam(model, model.feature_extractor.conv_5)  # Using conv_1 as an example

# 4. Create an instance of GuidedBackprop
guided_bp = GuidedBackprop(model)

for i in range(10):  # Visualize attention maps for the first 10 subjects
    input_data = torch.tensor(cropped_images[i]['data']).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    input_data.requires_grad = True  # Ensure the input tensor requires gradients

    # Obtain attention maps using Grad-CAM
    attr_gradcam = layer_grad_cam.attribute(input_data, target=1)
    print(f"Dimensiones antes de squeeze: {attr_gradcam.size()}")  # Tensor before removing dimensions
    attr_gradcam_np = attr_gradcam.detach().numpy()
    attention_map = attr_gradcam_np.squeeze()  # Remove batch and channel dimensions
    print(f"Dimensiones después de squeeze: {attention_map.shape}")  # Array after removing dimensions

    # Interpolate the attention map to match the original dimensions
    attention_map_resized = F.interpolate(torch.tensor(attention_map).unsqueeze(0).unsqueeze(0), size=(160, 192, 160), mode='trilinear', align_corners=False).squeeze().numpy()

    # Obtain guided backpropagation gradients
    guided_grads = guided_bp.attribute(input_data, target=1).squeeze().detach().numpy()

    # Combine Grad-CAM with Guided Backpropagation to get Guided Grad-CAM
    guided_gradcam = guided_grads * attention_map_resized

    # Select a central slice of the volume to display
    central_slice_z = input_data.squeeze().detach().numpy()[80, :, :]  # Central slice in the Z dimension
    guided_gradcam_slice_z = guided_gradcam[80, :, :]
    gradcam_slice_z = attention_map_resized[80, :, :]
    guided_grads_slice_z = guided_grads[80, :, :]

    central_slice_x = input_data.squeeze().detach().numpy()[:, :, 80]  # Central slice in the X dimension
    guided_gradcam_slice_x = guided_gradcam[:, :, 80]
    gradcam_slice_x = attention_map_resized[:, :, 80]
    guided_grads_slice_x = guided_grads[:, :, 80]

    central_slice_y = input_data.squeeze().detach().numpy()[:, 96, :]  # Central slice in the Y dimension
    guided_gradcam_slice_y = guided_gradcam[:, 96, :]
    gradcam_slice_y = attention_map_resized[:, 96, :]
    guided_grads_slice_y = guided_grads[:, 96, :]

    # Rotate the images (if necessary)
    central_slice_z_rotated = rotate(central_slice_z, 90)
    guided_gradcam_slice_z_rotated = rotate(guided_gradcam_slice_z, 90)
    gradcam_slice_z_rotated = rotate(gradcam_slice_z, 90)
    guided_grads_slice_z_rotated = rotate(guided_grads_slice_z, 90)

    central_slice_x_rotated = rotate(central_slice_x, 90)
    guided_gradcam_slice_x_rotated = rotate(guided_gradcam_slice_x, 90)
    gradcam_slice_x_rotated = rotate(gradcam_slice_x, 90)
    guided_grads_slice_x_rotated = rotate(guided_grads_slice_x, 90)

    central_slice_y_rotated = rotate(central_slice_y, 90)
    guided_gradcam_slice_y_rotated = rotate(guided_gradcam_slice_y, 90)
    gradcam_slice_y_rotated = rotate(gradcam_slice_y, 90)
    guided_grads_slice_y_rotated = rotate(guided_grads_slice_y, 90)

    # Normalize the central slices for visualization
    central_slice_z_norm = (central_slice_z_rotated - central_slice_z_rotated.min()) / (central_slice_z_rotated.max() - central_slice_z_rotated.min())
    central_slice_x_norm = (central_slice_x_rotated - central_slice_x_rotated.min()) / (central_slice_x_rotated.max() - central_slice_x_rotated.min())
    central_slice_y_norm = (central_slice_y_rotated - central_slice_y_rotated.min()) / (central_slice_y_rotated.max() - central_slice_y_rotated.min())

    # Visualize the central slices and attention maps overlay
    plt.figure(figsize=(30, 15))

    # Original slices
    plt.subplot(3, 3, 1)
    plt.imshow(central_slice_z_norm, cmap='gray')
    plt.title('Sección central (Z)')
    plt.axis('off')

    plt.subplot(3, 3, 4)
    plt.imshow(central_slice_x_norm, cmap='gray')
    plt.title('Sección central (X)')
    plt.axis('off')

    plt.subplot(3, 3, 7)
    plt.imshow(central_slice_y_norm, cmap='gray')
    plt.title('Sección central (Y)')
    plt.axis('off')

    # Guided Backpropagation
    plt.subplot(3, 3, 2)
    plt.imshow(guided_grads_slice_z_rotated, cmap='jet', alpha=0.5)
    plt.title('Guided Backprop (Z)')
    plt.axis('off')

    plt.subplot(3, 3, 5)
    plt.imshow(guided_grads_slice_x_rotated, cmap='jet', alpha=0.5)
    plt.title('Guided Backprop (Z)')
    plt.axis('off')

    plt.savefig(f'input_data_volume_overlay_subject_{i+1}.png')
    plt.show()
