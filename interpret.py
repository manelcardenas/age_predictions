import torch
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import nibabel as nib

from model.loss import my_KLDivLoss
from model.model import CNNmodel
from m_utils.data_transform import num2vect
from model.mri_dataset import MRIDataset

h5_path = '/mnt/work/datasets/UKBiobank/MN_females_data.h5'
save_dir = 'subjects_data'
os.makedirs(save_dir, exist_ok=True)

with h5py.File(h5_path, 'r') as h5_file:
    keys = list(h5_file.keys())
    ages = [h5_file[subject_id].attrs['Age'] for subject_id in keys]

age_range = [42, 82]
age_step = 1
sigma = 1
age_dist_list = []
bin_center_list = []

for age in ages:
    age_array = np.array([age,])
    age_dist, bc = num2vect(age_array, age_range, age_step, sigma)
    age_dist_list.append(age_dist)
    bin_center_list.append(bc)

age_dist_array = np.array(age_dist_list)

keys_train_val, keys_test, age_dist_train_val, age_dist_test = train_test_split(keys, age_dist_array, test_size=0.2, random_state=42)

dataset_test = MRIDataset(h5_path, keys_test, age_dist_test)

test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

print(f'Number of samples in the test set: {len(dataset_test)}')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNmodel()
state_dict = torch.load('best_models/best_model_woman.p')

from collections import OrderedDict
new_state_dict = OrderedDict()

for k, v in state_dict.items():
    name = k[7:]  # eliminar el prefijo 'module.'
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
model = model.to(device)

def calculate_mae(predictions, targets):
    return torch.mean(torch.abs(predictions - targets))

class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]
        
        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))
    
    def __call__(self, input, class_idx=None):
        self.model.zero_grad()
        output = self.model(input)
        
        if class_idx is None:
            class_idx = output[0].argmax(dim=1).item()
        
        target = output[0][0, class_idx]
        target.backward()
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3, 4])
        activations = self.activations.squeeze().cpu().numpy()
        
        for i in range(len(pooled_gradients)):
            activations[i, :, :, :] *= pooled_gradients[i].cpu().numpy()
        
        heatmap = np.mean(activations, axis=0)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        return heatmap

model.eval()

# Seleccionamos la última capa convolucional de la feature_extractor
target_layer = model.feature_extractor[-1]

grad_cam = GradCAM3D(model, target_layer)

test_loss = 0.0
test_mae = 0.0
all_predictions = []
all_real_ages = []
processed_samples = 0

for inputs, age_dist, age_real, subject_ids in test_loader:
    if processed_samples >= 5:
        break
    
    inputs = inputs.to(device)
    outputs = model(inputs)
    x = outputs[0].cpu().view(-1, 1, 40)

    x_sample = x[0].detach().numpy().reshape(-1)
    prob = np.exp(x_sample)
    pred = prob @ bin_center_list[0]
    
    preds = torch.tensor([pred])
    preds_rounded = torch.round(preds * 100) / 100  
    mae = calculate_mae(preds_rounded, age_real)  
    loss = my_KLDivLoss(x, age_dist)
    
    test_loss += loss.item()
    test_mae += mae.item()

    all_predictions.extend(preds_rounded.tolist())
    all_real_ages.extend(age_real.tolist())

    # Aplicar Grad-CAM y visualizar para el sujeto actual
    input_sample = inputs[0:1]
    input_sample.requires_grad = True
    heatmap = grad_cam(input_sample)
    
    # Visualizar el mapa de activación
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = heatmap.nonzero()
    ax.scatter(x, y, z, c=heatmap[x, y, z], cmap='jet')
    plt.title(f'Subject ID: {subject_ids[0]}, Predicted Age: {preds_rounded[0]:.2f}, Real Age: {age_real[0]:.2f}')
    plt.show()

    # Superponer el mapa de activación en la imagen original
    superimposed_img = heatmap * 0.4 + input_sample.cpu().numpy().squeeze()
    save_path = os.path.join(save_dir, f'{subject_ids[0]}_grad_cam.nii')
    nib.save(nib.Nifti1Image(superimposed_img, np.eye(4)), save_path)
    
    processed_samples += 1

# Calculando el loss de prueba promedio
test_loss /= processed_samples
test_mae /= processed_samples
print(f'Test Loss: {test_loss}, Test MAE: {test_mae}')

# Imprimir la edad real y la predicción para cada sujeto
for real_age, prediction in zip(all_real_ages, all_predictions):
    print(f'Real Age: {real_age}, Predicted Age: {prediction}')
