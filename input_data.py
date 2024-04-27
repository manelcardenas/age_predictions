from m_utils.data_transform import num2vect
from m_utils.plots import *
from model.loss import my_KLDivLoss
from model.model import CNNmodel

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import torch
import wandb
import psutil

from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DataParallel
from torch.utils.data.distributed import DistributedSampler
from torchsummary import summary
from sklearn.model_selection import train_test_split


wandb.init(project='Brain_age', entity='manelcardenas')


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

            # Obtener la edad real del sujeto del archivo HDF5
            age = subject_group.attrs['Age']
            age_tensor = torch.tensor(age).float()  # Convertir a tensor
            
        return mri_data_tensor, age_dist_tensor, age_tensor, subject_id

# Ruta al archivo .h5 de mujeres
#h5_path = '/home/usuaris/imatge/joan.manel.cardenas/MN_females_data.h5'
h5_path = '/mnt/work/datasets/UKBiobank/MN_females_data.h5'

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
keys_train_val, keys_test, age_dist_train_val, age_dist_test = train_test_split(keys, age_dist_array, test_size=0.2, random_state=42) # (train,val) / test      stratify=age_dist
keys_train, keys_val, age_dist_train, age_dist_val = train_test_split(keys_train_val, age_dist_train_val, test_size=0.25, random_state=42)#,  train/val  stratify=age_dist_train
#datasets
dataset_train = MRIDataset(h5_path, keys_train, age_dist_train)
dataset_val = MRIDataset(h5_path, keys_val, age_dist_val)
dataset_test = MRIDataset(h5_path, keys_test, age_dist_test)
#paralelization
train_sampler = DistributedSampler(dataset_train)
val_sampler = DistributedSampler(dataset_val)
test_sampler = DistributedSampler(dataset_test)
#dataloader
train_loader = DataLoader(dataset_train, batch_size=8, sampler=train_sampler, num_workers=10, pin_memory=True) #DROP_LAST
val_loader = DataLoader(dataset_val, batch_size=8, sampler=val_sampler, num_workers=10, pin_memory=True)
test_loader = DataLoader(dataset_test, batch_size=8, sampler=test_sampler, num_workers=10, pin_memory=True)

'''
#dataloader
train_loader = DataLoader(dataset_train, batch_size=8, shuffle=True, num_workers=10, pin_memory=True) #DROP_LAST 
val_loader = DataLoader(dataset_val, batch_size=8, shuffle=False, num_workers=10, pin_memory=True) #DROP_LAST 
test_loader = DataLoader(dataset_test, batch_size=8, shuffle=False, num_workers=10, pin_memory=True) #DROP_LAST 
'''
model = CNNmodel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001) 
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#summary(model, input_size=(1, 160, 192, 160))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA is available:", torch.cuda.is_available())  
model = model.to(device)

# Paralelización del modelo
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs for model parallelization.")
    model = DataParallel(model)

num_epochs = 100

def calculate_mae(predictions, targets):
    return torch.mean(torch.abs(predictions - targets))

def adjust_learning_rate(optimizer, epoch):
    lr = 0.01 * (0.3 ** (epoch // 30))  # Multiplicar por 0.3 cada 30 épocas
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

train_losses = []
val_losses = []
train_maes = []
val_maes = []

# Ciclo de entrenamiento 
for epoch in range(num_epochs):
    start_epoch = time.time()
    model.train()
    adjust_learning_rate(optimizer, epoch)
    running_loss = 0.0
    running_mae = 0.0
    for inputs, age_dist, age_real, subject_ids in train_loader:
        optimizer.zero_grad()
        inputs = inputs.to(device)
        outputs = model(inputs)
        x = outputs[0].cpu().view(-1, 1, 40)
        #print(f'Input train data shape: {inputs.shape}')
        #print(f'ages: {ages.shape}')  #torch.Size([8, 1, 40]).
        #print(f'outputs: {x.shape}')  #torch.Size([8, 1, 40]).

        preds = []
        for i in range(inputs.size(0)):
            x_sample = x[i].detach().numpy().reshape(-1)
            prob = np.exp(x_sample)
            pred = prob @ bin_center_list[i]
            preds.append(pred)
            #print(f'Predicción ajustada para el sujeto {subject_ids[i]}: {pred}')
        
        # Convertir a tensor y calcular el MAE
        preds = torch.tensor(preds)
        preds_rounded = torch.round(preds * 100) / 100  
        mae = calculate_mae(preds_rounded, age_real)
        loss = my_KLDivLoss(x, age_dist)
        loss.backward()

        # Monitoreo de recursos del sistema
        # Obtener información sobre el uso de la CPU
        cpu_percent = psutil.cpu_percent()
        cpu_count = psutil.cpu_count()
        # Obtener información sobre el uso de la memoria
        memory = psutil.virtual_memory()
        total_memory = memory.total
        available_memory = memory.available
        used_memory = memory.used
        memory_percent = memory.percent

        # Registrar métricas en W&B
        wandb.log({'loss': loss.item(), 'mae': mae.item(), 
                   'cpu_percent': cpu_percent, 'cpu_count': cpu_count,
                   'total_memory': total_memory, 'available_memory': available_memory,
                   'used_memory': used_memory, 'memory_percent': memory_percent})

        optimizer.step()
        running_loss += loss.item()
        running_mae += mae.item()

    # Calculando el loss de entrenamiento promedio por época
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    train_mae = running_mae / len(train_loader)
    train_maes.append(train_mae) 

    # Registrar métricas en Weights & Biases
    wandb.log({'train_loss': train_loss, 'train_mae': train_mae})

    end_epoch = time.time()  # Termina el temporizador para esta época
    epoch_time = end_epoch - start_epoch  
    print(f'Epoch {epoch + 1}, Time: {epoch_time:.2f} seconds')
    print(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Train MAE: {train_mae}')
    
    # Fase de validación
    model.eval()
    val_loss = 0.0
    val_mae = 0.0
    with torch.no_grad():
        for inputs, age_dist, age_real, subject_ids in val_loader:
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
            preds_rounded = torch.round(preds * 100) / 100  # Redondear a dos decimales
            mae = calculate_mae(preds_rounded, age_real)  
            loss = my_KLDivLoss(x, age_dist)
            
            val_loss += loss.item()
            val_mae += mae.item()
    
    # Calculando el loss de validación promedio por época
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    val_mae /= len(val_loader)
    val_maes.append(val_mae)
    print(f'Epoch {epoch + 1}, Val Loss: {val_loss}, Val MAE: {val_mae}')
    # Registrar métricas en Weights & Biases
    wandb.log({'val_loss': val_loss, 'val_mae': val_mae})

# Llamar a estas funciones después de que finalice el entrenamiento
plot_and_save_loss(train_losses, val_losses, save_dir)
plot_and_save_mae(train_maes, val_maes, save_dir)

print('Entrenamiento finalizado')

