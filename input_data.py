from m_utils.data_transform import num2vect
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
#import sklearn
from sklearn.model_selection import train_test_split
from model.model import CNNmodel
from model.loss import my_KLDivLoss
import matplotlib.pyplot as plt
import os
import time

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
            
        return mri_data_tensor, age_dist_tensor, subject_id

# Ruta al archivo .h5 de mujeres
#h5_path = '/home/usuaris/imatge/joan.manel.cardenas/MN_females_data.h5'
h5_path = '/mnt/work/datasets/UKBiobank/MN_females_data.h5'

with h5py.File(h5_path, 'r') as h5_file:
    #sigma = len(h5_file.keys())  # Número total de sujetos
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
print("Division de datos completada")
#TODO. CREAR GRAFICAS DE LAS DISTRIBUCIONES DE EDAD DE CADA SET. TAMBIEN DE PASO GARANTIZAR QUE KEYS TIENE LA LISTA DE IDS CORRECTO
# Crear instancias del dataset para cada subconjunto
dataset_train = MRIDataset(h5_path, keys_train, age_dist_train)
print("dataset_train creado")
dataset_val = MRIDataset(h5_path, keys_val, age_dist_val)
print("dataset_val creado")
dataset_test = MRIDataset(h5_path, keys_test, age_dist_test)
print("dataset_test creado")

#dataloader
train_loader = DataLoader(dataset_train, batch_size=8, shuffle=True, num_workers=10, pin_memory=True)
print("dataloader_train creado")
val_loader = DataLoader(dataset_val, batch_size=8, shuffle=False, num_workers=10, pin_memory=True)
print("dataloader_val creado")
test_loader = DataLoader(dataset_test, batch_size=8, shuffle=False, num_workers=10, pin_memory=True)
print("dataloader_test creado")
#NUM_WORKERS,PIN_MEMORY, DROP_LAST 

model = CNNmodel()
#loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

num_epochs = 100

def calculate_mae(predictions, targets):
    return torch.mean(torch.abs(predictions - targets))

def plot_age_distribution(ages, subject_ids, bin_centers_list):
    save_dir = 'subjects_data'  # Nombre del subdirectorio
    os.makedirs(save_dir, exist_ok=True)  # Crea el subdirectorio si no existe
    
    for i in range(len(ages)):
        age = ages[i].cpu().numpy().reshape(-1)
        subject_id = subject_ids[i]
        bin_centers = bin_centers_list[i]
        plt.figure()
        plt.bar(bin_centers, age)  # Utiliza bin_centers en lugar de age_range
        plt.title(f'Subject ID: {subject_id}, Age Distribution')
        plt.xlabel('Age Bin')
        plt.ylabel('Probability')
        plt.savefig(os.path.join(save_dir, f'subject_{subject_id}_age_distribution.png'))

# Ciclo de entrenamiento 
for epoch in range(num_epochs):
    start_epoch = time.time()
    start_train = time.time()
    model.train()
    end_train = time.time()
    train_time = end_train - start_train
    print(f'model.train() done, Time: {train_time:.2f} seconds')
    running_loss = 0.0
    running_mae = 0.0
    for inputs, ages, subject_ids in train_loader:
        print(f'Input train data shape: {inputs.shape}')
        optimizer.zero_grad()
        outputs = model(inputs)
        x = outputs[0].cpu().view(-1, 1, 40)
        print(f'ages: {ages.shape}')  #torch.Size([8, 1, 40]).
        print(f'outputs: {x.shape}')  #torch.Size([8, 1, 40]).
        start_loss = time.time()
        loss = my_KLDivLoss(x, ages)
        end_loss = time.time()
        loss_time = end_loss - start_loss
        print(f'KLD loss done, Time: {loss_time:.2f} seconds')
        start_mae = time.time()
        mae = calculate_mae(x, ages)
        end_mae = time.time()
        mae_time = end_mae - start_mae
        print(f'MAE done, Time: {mae_time:.2f} seconds')
        #loss = my_KLDivLoss(outputs[0].squeeze(), ages)
        start_backward = time.time()
        loss.backward()
        end_backward = time.time()
        backward_time = end_backward - start_backward
        print(f'loss.backward() done, Time: {backward_time:.2f} seconds')
        start_optimizer = time.time()
        optimizer.step()
        end_optimizer = time.time()
        optimizer_time = end_optimizer - start_optimizer
        print(f'optimizer.step() done, Time: {optimizer_time:.2f} seconds')
        running_loss += loss.item()
        running_mae += mae.item()
    
    # Calculando el loss de entrenamiento promedio por época
    train_loss = running_loss / len(train_loader)
    train_mae = running_mae / len(train_loader)
    end_epoch = time.time()  # Termina el temporizador para esta época
    epoch_time = end_epoch - start_epoch  # Calcula el tiempo de ejecución de la época
    print(f'Epoch {epoch + 1}, Time: {epoch_time:.2f} seconds')
    print(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Train MAE: {train_mae}')
    
    # Fase de validación
    model.eval()
    val_loss = 0.0
    val_mae = 0.0
    with torch.no_grad():
        for inputs, ages, subject_ids in val_loader:
            outputs = model(inputs)
            x = outputs[0].cpu().view(-1, 1, 40)
            mae = calculate_mae(x, ages)
            loss = my_KLDivLoss(x, ages)
            val_loss += loss.item()
            val_mae += mae.item()
    
    # Calculando el loss de validación promedio por época
    val_loss /= len(val_loader)
    val_mae /= len(val_loader)
    print(f'Epoch {epoch + 1}, Validation Loss: {val_loss}, Validation MAE: {val_mae}')

    #plot_age_distribution(ages, subject_ids, bin_center_list)
    '''
    # Evaluation
    model.eval() # Don't forget this. BatchNorm will be affected if not in eval mode.
    with torch.no_grad():
        for i, (inputs, ages) in enumerate(test_loader):
            outputs = model(inputs)
            adjusted_outputs = outputs[0][:, 0].unsqueeze(0)
            # Obtener las salidas del modelo y transformarlas
            #x = outputs[0].cpu().reshape([1, -1])
            x = adjusted_outputs.cpu().reshape([1, -1])
            print(f'Output shape: {x.shape}')
            x = x.numpy().reshape(-1)
            prob = np.exp(x)

            # Calcular la predicción utilizando el producto punto con los bin_centers (bc)
            pred = prob @ bc

            # Visualizar la distribución de probabilidad
            plt.bar(bc, prob)
            plt.title(f'Prediction: age={pred:.2f}')
            plt.xlabel('Age')
            plt.ylabel('Probability')
        
            # Crear el directorio para guardar los archivos PNG si no existe
            save_dir = 'subjects_data'
            os.makedirs(save_dir, exist_ok=True)
            # Guardar la visualización como un archivo PNG
            plt.savefig(os.path.join(save_dir, f'prediction_epoch{epoch}_example_{i}.png'))
        
            # Limpiar la figura actual para la siguiente iteración
            plt.clf()

            # Imprimir la pérdida asociada con la predicción
            #loss = my_KLDivLoss(outputs[0].squeeze(), ages).item()
            #print(f'Loss: {loss}')

            # Detener después de mostrar un número específico de ejemplos (por ejemplo, 5)
            # Si deseas mostrar más ejemplos, puedes modificar este número.
            if i == 5:
                break

'''
print('Entrenamiento finalizado')

