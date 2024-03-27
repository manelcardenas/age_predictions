from m_utils.data_transform import num2vect
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
#import sklearn
from sklearn.model_selection import train_test_split
from model.model import CNNmodel
from model.loss import my_KLDivLoss

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
            mri_data_tensor = torch.from_numpy(mri_data).float().unsqueeze(0)  # [C, D, H, W]
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
#age_min, age_max = int(np.floor(min(ages))), int(np.ceil(max(ages)))
#age_range = [age_min, age_max]
age_range = [42,82]
age_step = 1  # Paso de edad
print(f"Número total de sujetos (sigma): {sigma}")
print(f"Rango de edad (age_range): {age_range}")
print(f"Edades después de la conversión a np.array: {age_array}")
age_dist, bin_centers = num2vect(age_array, age_range, age_step, sigma)

# Dividir los datos
keys_train_val, keys_test, age_dist_train_val, age_dist_test = train_test_split(keys, age_dist, test_size=0.2, random_state=42) # (train,val) / test      stratify=age_dist
keys_train, keys_val, age_dist_train, age_dist_val = train_test_split(keys_train_val, age_dist_train_val, test_size=0.25, random_state=42)#,  train/val  stratify=age_dist_train

#TODO. CREAR GRAFICAS DE LAS DISTRIBUCIONES DE EDAD DE CADA SET. TAMBIEN DE PASO GARANTIZAR QUE KEYS TIENE LA LISTA DE IDS CORRECTO
# Crear instancias del dataset para cada subconjunto
dataset_train = MRIDataset(h5_path, keys_train, age_dist_train)
dataset_val = MRIDataset(h5_path, keys_val, age_dist_val)
dataset_test = MRIDataset(h5_path, keys_test, age_dist_test)

#dataloader
train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=64, shuffle=False)
test_loader = DataLoader(dataset_test, batch_size=64, shuffle=False)
#NUM_WORKERS,PIN_MEMORY, DROP_LAST 

model = CNNmodel()
#loss_function = torch.nn.MSELoss()
#loss_function = torch.nn.KLDivLoss(reduction='batchmean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

num_epochs = 10

# Ciclo de entrenamiento con validación
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, ages in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = my_KLDivLoss(outputs[0].squeeze(), ages)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Calculando el loss de entrenamiento promedio por época
    train_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch + 1}, Train Loss: {train_loss}')
    
    # Fase de validación
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, ages in val_loader:
            outputs = model(inputs)
            loss = my_KLDivLoss(outputs[0].squeeze(), ages)
            val_loss += loss.item()
    
    # Calculando el loss de validación promedio por época
    val_loss /= len(val_loader)
    print(f'Epoch {epoch + 1}, Validation Loss: {val_loss}')

print('Entrenamiento finalizado')

