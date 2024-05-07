import torch
import torch.nn as nn
import torch.optim as optim
import psutil
import wandb
import time
import os
import h5py
import numpy as np

from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from torchsummary import summary
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

from m_utils.data_transform import *
from m_utils.plots import *
from model.loss import my_KLDivLoss
from model.model import CNNmodel
from model.mri_dataset import MRIDataset

wandb.init(project='Brain_age', entity='manelcardenas')

#H5 file path
#h5_path = '/home/usuaris/imatge/joan.manel.cardenas/MN_females_data.h5'
h5_path = '/mnt/work/datasets/UKBiobank/MN_females_data.h5'

#Storage path
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

#dataloader
train_loader = DataLoader(dataset_train, batch_size=8, shuffle=True, num_workers=10, pin_memory=True) #DROP_LAST 
val_loader = DataLoader(dataset_val, batch_size=8, shuffle=False, num_workers=10, pin_memory=True) #DROP_LAST 
test_loader = DataLoader(dataset_test, batch_size=8, shuffle=False, num_workers=10, pin_memory=True) #DROP_LAST 


def main():

    model = CNNmodel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Paralelización del modelo
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs for model parallelization.")
        model = DataParallel(model)
    
    num_epochs = 130
    best_val_mae = float('inf')
    best_model_path = 'best_model.pth'

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
            
            loss.backward()
            optimizer.step()

            # Monitoreo de recursos del sistema
            # Obtener información sobre el uso de la memoria
            memory = psutil.virtual_memory()
            available_memory = memory.available
            used_memory = memory.used
            memory_percent = memory.percent

            # Registrar métricas en W&B
            wandb.log({'loss': loss.item(), 'mae': mae.item(), 
                        'available_memory': available_memory,
                        'used_memory': used_memory, 'memory_percent': memory_percent})

            running_loss += loss.item()
            running_mae += mae.item()

        # Calculando el loss de entrenamiento promedio por época
        train_loss = running_loss / len(train_loader)
        train_mae = running_mae / len(train_loader)

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
        val_pearson = 0.0
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
                
                pearson_corr, _ = pearsonr(preds_rounded.numpy().flatten(), age_real.numpy().flatten())
                val_pearson += pearson_corr
                val_loss += loss.item()
                val_mae += mae.item()

        # Calculando el loss de validación promedio por época
        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        print(f'Epoch {epoch + 1}, Val Loss: {val_loss}, Val MAE: {val_mae}, Val Pearson: {val_pearson}')
        wandb.log({'val_loss': val_loss, 'val_mae': val_mae, 'val_pearson': val_pearson})
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), best_model_path)

    # Evaluación en el conjunto de pruebas
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    test_loss = 0.0
    test_mae = 0.0
    test_pearson = 0.0
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

            pearson_corr, _ = pearsonr(preds_rounded.numpy().flatten(), age_real.numpy().flatten())
            test_pearson += pearson_corr        
            test_loss += loss.item()
            test_mae += mae.item()

    # Calculando el loss de prueba promedio
    test_loss /= len(test_loader)
    test_mae /= len(test_loader)
    print(f'Test Loss: {test_loss}, Test MAE: {test_mae}, Test Pearson: {test_pearson}')
    wandb.log({'test_loss': test_loss, 'test_mae': test_mae, 'test_pearson': test_pearson})

    print('Entrenamiento finalizado')

main()

