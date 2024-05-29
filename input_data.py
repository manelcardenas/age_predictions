from m_utils.data_transform import *
from m_utils.plots import *
from model.loss import my_KLDivLoss
from model.model import CNNmodel
from model.mri_dataset import MRIDataset

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import torch
import wandb
import psutil

from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
#from torch.utils.data.distributed import DistributedSampler as DDP
from tensorfn import distributed as dist
from torchsummary import summary
from sklearn.model_selection import train_test_split
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomCrop, Compose


wandb.init(project='Brain_age', entity='manelcardenas')

# Ruta al archivo .h5 de mujeres
#h5_path = '/home/usuaris/imatge/joan.manel.cardenas/MN_females_data.h5'
h5_path = '/mnt/work/datasets/UKBiobank/MN_females_data.h5'

save_dir = 'subjects_data'
os.makedirs(save_dir, exist_ok=True)

import numpy as np
import torch

import numpy as np
import torch

class RandomShift:
    def __init__(self, max_shift=2):
        self.max_shift = max_shift

    def __call__(self, mri_data_tensor):
        shift = np.random.randint(-self.max_shift, self.max_shift + 1, size=3)
        mri_data_tensor = np.roll(mri_data_tensor, shift, axis=(1, 2, 3))
        return torch.from_numpy(mri_data_tensor).float()

import random
import torch

class RandomMirror:
    def __call__(self, mri_data_tensor):
        if random.random() > 0.5:
            mri_data_tensor = torch.flip(mri_data_tensor, dims=[3])  # Flip along the width axis
        return mri_data_tensor


   
age_dist_array, keys, bin_center_list = get_age_distribution(h5_path)


from torchvision.transforms import Compose

train_transform = Compose([
    RandomShift(max_shift=2),
    RandomMirror()
])

# Dividir los datos
keys_train_val, keys_test, age_dist_train_val, age_dist_test = train_test_split(keys, age_dist_array, test_size=0.2, random_state=42)    #stratify=age_dist_train
keys_train, keys_val, age_dist_train, age_dist_val = train_test_split(keys_train_val, age_dist_train_val, test_size=0.125, random_state=42)


# Crear los datasets
dataset_train = MRIDataset(h5_path, keys_train, age_dist_train, transform=train_transform)
dataset_val = MRIDataset(h5_path, keys_val, age_dist_val)
dataset_test = MRIDataset(h5_path, keys_test, age_dist_test)

#dataloader
train_loader = DataLoader(dataset_train, batch_size=8, shuffle=True, num_workers=10, pin_memory=True) #DROP_LAST 
val_loader = DataLoader(dataset_val, batch_size=8, shuffle=False, num_workers=10, pin_memory=True) #DROP_LAST 
test_loader = DataLoader(dataset_test, batch_size=8, shuffle=False, num_workers=10, pin_memory=True) #DROP_LAST 

model = CNNmodel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001) 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA is available:", torch.cuda.is_available())  
model = model.to(device)


# Paralelización del modelo
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs for model parallelization.")
    model = DataParallel(model)


num_epochs = 110
best_val_mae = float('inf')
best_model_path = 'best_model.p'

def calculate_mae(predictions, targets):
    return torch.mean(torch.abs(predictions - targets))

import torch.nn as nn

mae_loss = nn.L1Loss()

def adjust_learning_rate(optimizer, epoch):
    lr = 0.01 * (0.3 ** (epoch // 30))  # Multiplicar por 0.3 cada 30 épocas
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

num_subjects_processed = 0


# Ciclo de entrenamiento 
for epoch in range(num_epochs):
    start_epoch = time.time()
    model.train()
    adjust_learning_rate(optimizer, epoch)
    running_loss = 0.0
    running_mae = 0.0
    num_subjects_epoch = 0

    for inputs, age_dist, age_real, subject_ids in train_loader:
        optimizer.zero_grad()
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
        #mae = mae_loss(preds_rounded, age_real)
        loss = my_KLDivLoss(x, age_dist)
        loss.backward()

        # Monitoreo de recursos del sistema
        # Obtener información sobre el uso de la CPU
        #cpu_percent = psutil.cpu_percent()
        #cpu_count = psutil.cpu_count()
        # Obtener información sobre el uso de la memoria
        memory = psutil.virtual_memory()
        #total_memory = memory.total
        available_memory = memory.available
        used_memory = memory.used
        memory_percent = memory.percent

        # Registrar métricas en W&B
        wandb.log({'available_memory': available_memory,'used_memory': used_memory, 'memory_percent': memory_percent})

        optimizer.step()
        running_loss += loss.item()
        running_mae += mae.item()
        num_subjects_epoch += inputs.size(0)

    num_subjects_processed += num_subjects_epoch
    # Calculando el loss de entrenamiento promedio por época
    train_loss = running_loss / len(train_loader)
    train_mae = running_mae / len(train_loader)

    # Registrar métricas en Weights & Biases
    wandb.log({'train_loss': train_loss, 'train_mae': train_mae})

    end_epoch = time.time()  # Termina el temporizador para esta época
    epoch_time = end_epoch - start_epoch  
    print(f'Epoch {epoch + 1}, Time: {epoch_time:.2f} seconds')
    print(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Train MAE: {train_mae}')
    print(f'Epoch {epoch + 1}, Subjects processed this epoch: {num_subjects_epoch}')

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
            #mae = mae_loss(preds_rounded, age_real)
            loss = my_KLDivLoss(x, age_dist)
            
            val_loss += loss.item()
            val_mae += mae.item()
    
    # Calculando el loss de validación promedio por época
    val_loss /= len(val_loader)
    val_mae /= len(val_loader)
    print(f'Epoch {epoch + 1}, Val Loss: {val_loss}, Val MAE: {val_mae}')
    print(f'Epoch {epoch + 1}, Total subjects processed so far: {num_subjects_processed}')
    # Registrar métricas en Weights & Biases
    wandb.log({'val_loss': val_loss, 'val_mae': val_mae})

    if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), best_model_path)
            print('Modelo guardado en', best_model_path)
print('Entrenamiento finalizado')

