import torch
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from captum.attr import LayerGradCam, GuidedBackprop
from collections import OrderedDict

from model.loss import my_KLDivLoss
from model.model import CNNmodel
from m_utils.data_transform import num2vect
from model.mri_dataset import MRIDataset

# Función para calcular el MAE
def calculate_mae(predictions, targets):
    return torch.mean(torch.abs(predictions - targets))

# Función para realizar el centro crop
def center_crop(data, target_shape=(160, 192, 160)):
    current_shape = data.shape
    crop_slices = tuple(slice((current_dim - target_dim) // 2, (current_dim + target_dim) // 2)
                        for current_dim, target_dim in zip(current_shape, target_shape))
    return data[crop_slices]

# Directorio y archivo HDF5 con los datos
h5_path = '/mnt/work/datasets/UKBiobank/MN_males_data.h5'
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
state_dict = torch.load('best_models/best_model_male_DA.p')
new_state_dict = OrderedDict()

for k, v in state_dict.items():
    name = k[7:]  # Eliminar el prefijo 'module.'
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
model.eval()

# Función para calcular y visualizar Guided Grad-CAM
def visualize_guided_grad_cam(input_data, model, layer_name='conv_5'):
    # Crear instancias de GuidedBackprop y LayerGradCam
    guided_bp = GuidedBackprop(model)
    layer_grad_cam = LayerGradCam(model, getattr(model.feature_extractor, layer_name))

    # Asegurar que el tensor de entrada requiere gradientes
    input_data = input_data.to(device)
    input_data.requires_grad = True

    # Calcular atribuciones de Grad-CAM
    attr_gradcam = layer_grad_cam.attribute(input_data, target=1)
    attr_gradcam_np = attr_gradcam.detach().cpu().numpy()
    attention_map = attr_gradcam_np.squeeze()

    # Calcular gradientes guiados
    guided_grads = guided_bp.attribute(input_data, target=1).squeeze().detach().cpu().numpy()

    # Combinar Grad-CAM y Guided Backpropagation para obtener Guided Grad-CAM
    guided_gradcam = guided_grads * attention_map

    # Seleccionar una rebanada central del volumen para mostrar
    central_slice_z = input_data.squeeze().detach().cpu().numpy()[80, :, :]
    guided_gradcam_slice_z = guided_gradcam[80, :, :]

    # Rotar las imágenes (si es necesario) y normalizarlas para la visualización
    central_slice_z_rotated = rotate(central_slice_z, 90)
    guided_gradcam_slice_z_rotated = rotate(guided_gradcam_slice_z, 90)
    central_slice_z_norm = (central_slice_z_rotated - central_slice_z_rotated.min()) / (central_slice_z_rotated.max() - central_slice_z_rotated.min())

    # Visualizar la rebanada central y el mapa de atención
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(central_slice_z_norm, cmap='gray')
    plt.imshow(guided_gradcam_slice_z_rotated, cmap='jet', alpha=0.5)
    plt.title('Central Slice with Guided Grad-CAM (Z)')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(guided_gradcam_slice_z_rotated, cmap='jet')
    plt.title('Guided Grad-CAM (Z)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Evaluación en el conjunto de pruebas con Guided Grad-CAM
for inputs, age_dist, age_real, subject_ids in test_loader:
    inputs = inputs.to(device)
    outputs = model(inputs)

    # Calcular Guided Grad-CAM en la primera muestra del conjunto de prueba
    visualize_guided_grad_cam(inputs, model, layer_name='conv_5')
    break  # Solo visualizamos el primero por simplicidad

# Cálculo de la pérdida y MAE promedio en el conjunto de pruebas
test_loss = 0.0
test_mae = 0.0
all_predictions = []
all_real_ages = []

with torch.no_grad():
    for inputs, age_dist, age_real, subject_ids in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)

        # Calcular la edad predicha y MAE
        x = outputs[0].cpu().view(-1, 1, 40)
        preds = []
        for i in range(inputs.size(0)):
            x_sample = x[i].detach().numpy().reshape(-1)
            prob = np.exp(x_sample)
            pred = prob @ bin_center_list[i]
            preds.append(pred)

        preds = torch.tensor(preds)
        preds_rounded = torch.round(preds * 100) / 100
        mae = calculate_mae(preds_rounded, age_real)
        loss = my_KLDivLoss(x, age_dist)

        test_loss += loss.item()
        test_mae += mae.item()

        all_predictions.extend(preds_rounded.tolist())
        all_real_ages.extend(age_real.tolist())

test_loss /= len(test_loader)
test_mae /= len(test_loader)

print(f'Test Loss: {test_loss}, Test MAE: {test_mae}')

# Imprimir la edad real y la predicción para cada sujeto
for real_age, prediction in zip(all_real_ages, all_predictions):
    print(f'Real Age: {real_age}, Predicted Age: {prediction}')


