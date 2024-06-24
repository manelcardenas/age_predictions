import os
import torch
import zipfile
import pandas as pd
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import GuidedGradCam, LayerGradCam
from collections import OrderedDict
from model.model import CNNmodel  # Ensure this import is correct
import torch.nn.functional as F
from scipy.ndimage import rotate

from m_utils.data_transform import *

# Define the center_crop function
def center_crop(data, target_shape=(160, 192, 160)):
    current_shape = data.shape
    crop_slices = tuple(slice((current_dim - target_dim) // 2, (current_dim + target_dim) // 2)
                        for current_dim, target_dim in zip(current_shape, target_shape))
    return data[crop_slices]

# Directory containing the folders with .nii.gz files
zip_directory = '/mnt/work/datasets/UKBiobank'

# Read the CSV file with participant information
demographics = pd.read_csv('/home/usuaris/imatge/joan.manel.cardenas/age_predictions/demographics_icd_new_date3.csv')

# Ordenar los datos por el ID de los sujetos
demographics.loc[:, 'Label'] = demographics['ID'].astype(str)
reorder_demogs = demographics.sort_values('Label').reset_index(drop=True)

# Initialize a list to store cropped images
cropped_images = []

# Count the number of 'first.nii.gz' files
count = 0

# Iterate over each participant
for index, row in reorder_demogs.iterrows():
    subject_id = row['Label']
    sex = row['Sex']
    #actual_age = row['Age_modif']
    actual_age = round(row['Age_modif'], 2)

    if sex == 1.0:  # man=1, woman=0
        continue  # Skip if sex is 1

    # Limitar la ejecución para fines de prueba
    if int(subject_id) >= 1015000:  #25
        break

    zip_filename = f"{subject_id}_20252_2_0.zip"
    zip_filepath = os.path.join(zip_directory, zip_filename)
    if os.path.exists(zip_filepath):
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            zip_file_contents = zip_ref.namelist()
            if 'T1/T1_brain_to_MNI.nii.gz' in zip_file_contents:
                temp_dir = '/tmp/extracted_files'
                os.makedirs(temp_dir, exist_ok=True)
                zip_ref.extract('T1/T1_brain_to_MNI.nii.gz', path=temp_dir)

                nifti_file_path = os.path.join(temp_dir, 'T1/T1_brain_to_MNI.nii.gz')
                img = nib.load(nifti_file_path)
                data = img.get_fdata()

                # Aplicar el recorte central
                cropped_data = center_crop(data)

                # Store the cropped image and metadata in the list
                cropped_images.append({
                    'data': cropped_data.astype(np.float32),
                    'IdParticipante': subject_id,
                    'Sexo': sex,
                    'Edad_CI': actual_age
                })

                count += 1

print(f'Número de imágenes recortadas: {count}')
print(f'Número de participantes: {reorder_demogs.shape[0]}')
for i, image_info in enumerate(cropped_images[:10]):
    participant_info = {k: v for k, v in image_info.items() if k != 'data'}
    print(f'Info del participante {i+1}: {participant_info}')


# 1. Create an instance of the model
model = CNNmodel()

# 2. Load the pretrained model weights
state_dict = torch.load('best_models/best_model_female_DA.p')
# Create a new state_dict without the 'module.' prefix
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:]  # Remove the 'module.' prefix
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)

model.eval()  # Set the model to evaluation mode

# 3. Create an instance of LayerGradCam
guided_grad_cam = LayerGradCam(model, model.feature_extractor.conv_4)  

output_dir = 'attention_maps'
os.makedirs(output_dir, exist_ok=True)

mean_attention_map_sum = np.zeros((160, 192, 160), dtype=np.float32)

for i in range(25):  # Visualize attention maps for the first 10 subjects
    input_data = cropped_images[i]['data']  # Add batch and channel dimensions
    age = cropped_images[i]['Edad_CI']
    subject_id = cropped_images[i]['IdParticipante']

    input_data = (input_data - input_data.mean()) / input_data.std()  # Normalize the input tensor
    input_data = input_data[np.newaxis, np.newaxis, ...]  # Add batch and channel dimensions
    input_data = torch.tensor(input_data, dtype=torch.float32, requires_grad=True)  # Convert to a PyTorch tensor


    bin_range = [42, 82]
    bin_step = 1
    sigma = 1

    # Transformar la edad a etiqueta suave (distribución de probabilidad)
    label = np.array([age])  # La edad real del sujeto
    y, bc = num2vect(label, bin_range, bin_step, sigma)
    y = torch.tensor(y, dtype=torch.float32).cpu()

    with torch.no_grad():
        output = model(input_data)
        x = output[0].cpu().view(-1, 1, 40)
        probabilities = torch.exp(x)  # Convertir log_softmax a probabilidades
        target_class = torch.argmax(probabilities).item()
    
    print(f'target_class: {target_class}')
    # Obtain attention maps using Grad-CAM
    attributions = guided_grad_cam.attribute(input_data, target=target_class)
    print(f"Dimensiones antes de squeeze: {attributions.size()}")  # Tensor before removing dimensions
    print(f"Predicted class for subject {i+1}: {subject_id}")
    # Convert attributions to numpy for visualization
    guided_gradcam = attributions.detach().numpy().squeeze()
    print(f"Dimensiones después de squeeze: {guided_gradcam.shape}")  # Array after removing dimensions

    # Interpolate the attention map to match the original dimensions
    attention_map_resized = F.interpolate(torch.tensor(guided_gradcam).unsqueeze(0).unsqueeze(0), size=(160, 192, 160), mode='trilinear', align_corners=False).squeeze().numpy()
    # Verificar los valores de las atribuciones
    print(f"Valores mínimos y máximos de las atribuciones: {attention_map_resized.min()}, {attention_map_resized.max()}")
    plt.figure()
    plt.hist(attention_map_resized.flatten(), bins=50)
    plt.title(f'Histograma de valores de atribuciones - Sujeto {i+1}')
    plt.savefig(f'Histogram{i+1}.png')
    plt.show()

    guided_gradcam_norm = (attention_map_resized - attention_map_resized.min()) / (attention_map_resized.max() - attention_map_resized.min())
    mean_attention_map_sum += guided_gradcam_norm

    attention_map_nii = nib.Nifti1Image(guided_gradcam_norm, affine=np.eye(4))
    nib.save(attention_map_nii, os.path.join(output_dir, f'guided_gradcam_subject_{i+1}_class_{subject_id}.nii'))

    # Select a central slice of the volume to display
    central_slice_z = input_data.detach().squeeze().numpy()[80, :, :]  # Sección central en la dimensión Z
    attention_slice_z = guided_gradcam_norm[80, :, :]


    central_slice_x = input_data.detach().squeeze().numpy()[:, :, 80]  # Sección central en la dimensión X
    attention_slice_x = guided_gradcam_norm[:, :, 80]


    central_slice_y = input_data.detach().squeeze().numpy()[:, 96, :]  # Sección central en la dimensión Y
    attention_slice_y = guided_gradcam_norm[:, 96, :]


    # Rotar las imágenes (si es necesario)
    central_slice_z_rotated = rotate(central_slice_z, 90)
    attention_slice_z_rotated = rotate(attention_slice_z, 90)

    central_slice_x_rotated = rotate(central_slice_x, 90)
    attention_slice_x_rotated = rotate(attention_slice_x, 90)

    central_slice_y_rotated = rotate(central_slice_y, 90)
    attention_slice_y_rotated = rotate(attention_slice_y, 90)


    # Normalize the central slices for visualization
    central_slice_z_norm = (central_slice_z_rotated - central_slice_z_rotated.min()) / (central_slice_z_rotated.max() - central_slice_z_rotated.min())
    central_slice_x_norm = (central_slice_x_rotated - central_slice_x_rotated.min()) / (central_slice_x_rotated.max() - central_slice_x_rotated.min())
    central_slice_y_norm = (central_slice_y_rotated - central_slice_y_rotated.min()) / (central_slice_y_rotated.max() - central_slice_y_rotated.min())

    # Visualize the central slices and attention maps overlay
    plt.figure(figsize=(30, 15))

    plt.subplot(3, 2, 1)
    plt.imshow(central_slice_z_norm, cmap='gray')  # Muestra la sección central del volumen (dimensión Z)
    plt.imshow(attention_slice_z_rotated, cmap='jet', alpha=0.5)  # Superpone el mapa de atención (dimensión Z)
    plt.title('Sección central del volumen con mapa de atención (Z)')
    plt.axis('off')

    plt.subplot(3, 2, 3)
    plt.imshow(central_slice_x_norm, cmap='gray')  # Muestra la sección central del volumen (dimensión X)
    plt.imshow(attention_slice_x_rotated, cmap='jet', alpha=0.5)  # Superpone el mapa de atención (dimensión X)
    plt.title('Sección central del volumen con mapa de atención (X)')
    plt.axis('off')

    plt.subplot(3, 2, 5)
    plt.imshow(central_slice_y_norm, cmap='gray')  # Muestra la sección central del volumen (dimensión Y)
    plt.imshow(attention_slice_y_rotated, cmap='jet', alpha=0.5)  # Superpone el mapa de atención (dimensión Y)
    plt.title('Sección central del volumen con mapa de atención (Y)')
    plt.axis('off')

    plt.savefig(f'input_data_volume_overlay_subject_{i+1}.png')
    plt.show()
    

mean_attention_map = mean_attention_map_sum / len(cropped_images)

attention_map_nii = nib.Nifti1Image(mean_attention_map, affine=np.eye(4))
nib.save(attention_map_nii, os.path.join(output_dir, f'gradcam_mean.nii'))

# Select a central slice of the volume to display
central_slice_z = input_data.detach().squeeze().numpy()[80, :, :]  # Sección central en la dimensión Z
attention_slice_z = mean_attention_map[80, :, :]


central_slice_x = input_data.detach().squeeze().numpy()[:, :, 80]  # Sección central en la dimensión X
attention_slice_x = mean_attention_map[:, :, 80]


central_slice_y = input_data.detach().squeeze().numpy()[:, 96, :]  # Sección central en la dimensión Y
attention_slice_y = mean_attention_map[:, 96, :]


# Rotar las imágenes (si es necesario)
central_slice_z_rotated = rotate(central_slice_z, 90)
attention_slice_z_rotated = rotate(attention_slice_z, 90)

central_slice_x_rotated = rotate(central_slice_x, 90)
attention_slice_x_rotated = rotate(attention_slice_x, 90)

central_slice_y_rotated = rotate(central_slice_y, 90)
attention_slice_y_rotated = rotate(attention_slice_y, 90)


# Normalize the central slices for visualization
central_slice_z_norm = (central_slice_z_rotated - central_slice_z_rotated.min()) / (central_slice_z_rotated.max() - central_slice_z_rotated.min())
central_slice_x_norm = (central_slice_x_rotated - central_slice_x_rotated.min()) / (central_slice_x_rotated.max() - central_slice_x_rotated.min())
central_slice_y_norm = (central_slice_y_rotated - central_slice_y_rotated.min()) / (central_slice_y_rotated.max() - central_slice_y_rotated.min())
    
# Visualize the central slices and attention maps overlay
plt.figure(figsize=(30, 15))

plt.subplot(3, 2, 1)
plt.imshow(central_slice_z_norm, cmap='gray')  # Muestra la sección central del volumen (dimensión Z)
plt.imshow(attention_slice_z_rotated, cmap='jet', alpha=0.5)  # Superpone el mapa de atención (dimensión Z)
plt.title('Sección central del volumen con mapa de atención (Z)')
plt.axis('off')

plt.subplot(3, 2, 3)
plt.imshow(central_slice_x_norm, cmap='gray')  # Muestra la sección central del volumen (dimensión X)
plt.imshow(attention_slice_x_rotated, cmap='jet', alpha=0.5)  # Superpone el mapa de atención (dimensión X)
plt.title('Sección central del volumen con mapa de atención (X)')
plt.axis('off')

plt.subplot(3, 2, 5)
plt.imshow(central_slice_y_norm, cmap='gray')  # Muestra la sección central del volumen (dimensión Y)
plt.imshow(attention_slice_y_rotated, cmap='jet', alpha=0.5)  # Superpone el mapa de atención (dimensión Y)
plt.title('Sección central del volumen con mapa de atención (Y)')
plt.axis('off')

plt.savefig(f'Mean Attention Map.png')
plt.show()

