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
model = model.to(device)

def calculate_mae(predictions, targets):
    return torch.mean(torch.abs(predictions - targets))

# Evaluación en el conjunto de pruebas
model.eval()
test_loss = 0.0
test_mae = 0.0
all_predictions = []
all_real_ages = []
with torch.no_grad():
    for inputs, age_dist, age_real, subject_ids in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        x = outputs[0].cpu().view(-1, 1, 40)

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

# Imprimir la edad real y la predicción para cada sujeto
for real_age, prediction in zip(all_real_ages, all_predictions):
    print(f'Real Age: {real_age}, Predicted Age: {prediction}')
