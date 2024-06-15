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

# Define the center_crop function
def center_crop(data, target_shape=(160, 192, 160)):
    current_shape = data.shape
    crop_slices = tuple(slice((current_dim - target_dim) // 2, (current_dim + target_dim) // 2)
                        for current_dim, target_dim in zip(current_shape, target_shape))
    return data[crop_slices]

# Directory containing the folders with .nii.gz files
zip_directory = '/mnt/work/datasets/UKBiobank/MN_Process_alfa_mni/Process_alfa_mni'

# Read the CSV file with participant information
demographics = pd.read_csv('/mnt/work/datasets/UKBiobank/MN_Process_alfa_mni/ALFA_PLUS_3D.csv')

# Extract relevant columns
participants = demographics[['IdParticipante', 'Sexo', 'Edad_CI']]

# Initialize a list to store cropped images
cropped_images = []

# Count the number of 'first.nii.gz' files
count = 0

# Iterate over each participant
for _, row in participants.iterrows():
    participant_id = str(int(row['IdParticipante']))
    sex = row['Sexo']
    age = row['Edad_CI']

    if sex == 1.0:  # man=1, woman=2
        continue  # Skip if sex is 1

    # Find the corresponding folder for the participant
    for folder_name in os.listdir(zip_directory):
        if folder_name.startswith(participant_id):
            folder_path = os.path.join(zip_directory, folder_name)
            
            # Check if it is a directory
            if os.path.isdir(folder_path):
                # Find the file ending with 'first.nii.gz'
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('first.nii.gz'):
                        file_path = os.path.join(folder_path, file_name)
                        
                        # Load the NIfTI image
                        img = nib.load(file_path)
                        data = img.get_fdata()
                        
                        # Apply center crop
                        cropped_data = center_crop(data)
                        
                        # Store the cropped image and metadata in the list
                        cropped_images.append({
                            'data': cropped_data.astype(np.float32),
                            'IdParticipante': participant_id,
                            'Sexo': sex,
                            'Edad_CI': age
                        })
                        
                        # Increment the counter
                        count += 1

print(f'Número de imágenes recortadas: {count}')
print(f'Número de participantes: {participants.shape[0]}')
for i, image_info in enumerate(cropped_images[:10]):
    participant_info = {k: v for k, v in image_info.items() if k != 'data'}
    print(f'Info del participante {i+1}: {participant_info}')

# Use the first cropped image as input_data
input_data = torch.tensor(cropped_images[0]['data']).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
input_data.requires_grad = True  # Ensure the input tensor requires gradients

# 1. Create an instance of the model
model = CNNmodel()

# 2. Load the pretrained model weights
state_dict = torch.load('best_models/best_model_female_DA_1.p')
# Create a new state_dict without the 'module.' prefix
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:]  # Remove the 'module.' prefix
    new_state_dict[name] = v
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
    guided_gradcam = guided_grads #* attention_map_resized

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
