import torch
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from model.loss import my_KLDivLoss
from model.model import CNNmodel
from m_utils.data_transform import num2vect
from model.mri_dataset import MRIDataset

# Ruta del archivo h5
h5_path = '/mnt/work/datasets/UKBiobank/MN_females_data.h5'

# Directorio para guardar los resultados
save_dir = 'subjects_data'
os.makedirs(save_dir, exist_ok=True)

# Cargar datos desde el archivo h5
with h5py.File(h5_path, 'r') as h5_file:
    keys = list(h5_file.keys())
    ages = [h5_file[subject_id].attrs['Age'] for subject_id in keys]

age_range = [42, 82]
age_step = 1  # Paso de edad
sigma = 1
age_dist_list = []
bin_center_list = []

for age in ages:
    age_array = np.array([age])
    age_dist, bc = num2vect(age_array, age_range, age_step, sigma)
    age_dist_list.append(age_dist)
    bin_center_list.append(bc)

# Convertir la lista de distribuciones a un arreglo de numpy
age_dist_array = np.array(age_dist_list)

# Dividir los datos
keys_train_val, keys_test, age_dist_train_val, age_dist_test = train_test_split(keys, age_dist_array, test_size=0.2, random_state=42)

dataset_test = MRIDataset(h5_path, keys_test, age_dist_test)

# DataLoader
test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

print(f'Number of samples in the test set: {len(dataset_test)}')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNmodel()

# Cargar el state_dict guardado
state_dict = torch.load('best_models/best_model_female_DA.p')

# Crear un nuevo state_dict en el que las claves no tienen el prefijo "module."
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:]  # eliminar el prefijo 'module.'
    new_state_dict[name] = v

# Cargar el nuevo state_dict en el modelo
model.load_state_dict(new_state_dict)
model = model.to(device)

def calculate_mae(predictions, targets):
    return torch.mean(torch.abs(predictions - targets))

# Grad-CAM implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_hook(self, module, input, output):
        self.activations = output
        if output.requires_grad:
            output.register_hook(self.save_gradient)

    def __call__(self, input, index=None):
        self.hook_handles.append(self.target_layer.register_forward_hook(self.forward_hook))

        output = self.model(input)[0]
        if index is None:
            index = np.argmax(output.cpu().data.numpy())
        
        self.model.zero_grad()
        class_loss = output[0, index]
        class_loss.backward()

        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()
        weights = np.mean(gradients, axis=(1, 2, 3))
        
        grad_cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            grad_cam += w * activations[i, :, :, :]

        grad_cam = np.maximum(grad_cam, 0)
        
        # Adjust the resizing to match the input dimensions
        depth, height, width = input.shape[2:]
        grad_cam = np.transpose(grad_cam, (1, 2, 0))  # Ensure it has shape (height, width, depth)
        grad_cam = cv2.resize(grad_cam, (width, height), interpolation=cv2.INTER_LINEAR)
        grad_cam = np.transpose(grad_cam, (2, 0, 1))  # Convert back to (depth, height, width)

        grad_cam = grad_cam - np.min(grad_cam)
        grad_cam = grad_cam / np.max(grad_cam)
        return grad_cam

# Inicializar GradCAM
target_layer = model.feature_extractor[-1][0]  # Usar la última capa convolucional
grad_cam = GradCAM(model, target_layer)

# Evaluación en el conjunto de pruebas
model.eval()
test_loss = 0.0
test_mae = 0.0
all_predictions = []
all_real_ages = []

with torch.no_grad():
    for inputs, age_dist, age_real, subject_ids in test_loader:
        inputs = inputs.to(device)
        outputs, _ = model(inputs)
        x = outputs.cpu().view(-1, 1, 40)

        preds = []
        for i in range(inputs.size(0)):
            x_sample = x[i].detach().numpy().reshape(-1)
            prob = np.exp(x_sample)
            pred = prob @ bin_center_list[i]
            preds.append(pred)
        
        # Convertir a tensor y calcular el MAE
        preds = torch.tensor(preds)
        preds_rounded = torch.round(preds * 100) / 100  
        mae = calculate_mae(preds_rounded, age_real)  
        loss = my_KLDivLoss(x, age_dist)
        
        test_loss += loss.item()
        test_mae += mae.item()

        all_predictions.extend(preds_rounded.tolist())
        all_real_ages.extend(age_real.tolist())

# Calculando el loss de prueba promedio
test_loss /= len(test_loader)
test_mae /= len(test_loader)
print(f'Test Loss: {test_loss}, Test MAE: {test_mae}')

# Generar y guardar mapas de GradCAM solo para los primeros 5 sujetos
num_subjects_to_visualize = 5
num_subjects_visualized = 0

for i, (inputs, _, _, subject_ids) in enumerate(test_loader):
    if num_subjects_visualized >= num_subjects_to_visualize:
        break  # Salir del bucle si ya hemos visualizado los primeros 5 sujetos

    inputs = inputs.to(device)
    grad_cam_map = grad_cam(inputs)
    
    for j, sid in enumerate(subject_ids):
        heatmap = grad_cam_map
        print(f"Shape of heatmap for subject {sid}: {heatmap.shape}")
        print(f"Shape of inputs: {inputs.shape}")

        # Seleccionar la rebanada en el medio del eje z
        mid_slice = heatmap[:, int(heatmap.shape[1] / 2), :]
        
        # Asegurarnos de que el rango de valores sea correcto
        print(f"Range of values in mid_slice before normalization: {mid_slice.min()} to {mid_slice.max()}")

        # Visualizar el mapa de calor
        plt.figure(figsize=(10, 5))
        plt.imshow(mid_slice, cmap='jet', aspect='auto', alpha=0.5)  # Aspect ratio set to 'auto'
        plt.colorbar()
        plt.title(f'GradCAM Heatmap for Subject ID: {sid}')
        plt.savefig(os.path.join(save_dir, f'gradcam_subject_{sid}.png'))
        plt.close()

        num_subjects_visualized += 1

print("GradCam results for the first 5 subjects are saved in the 'subjects_data' directory.")
