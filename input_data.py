from m_utils.data_transform import num2vect
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torchsummary import summary
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
#TODO. CREAR GRAFICAS DE LAS DISTRIBUCIONES DE EDAD DE CADA SET. TAMBIEN DE PASO GARANTIZAR QUE KEYS TIENE LA LISTA DE IDS CORRECTO
# Crear instancias del dataset para cada subconjunto
dataset_train = MRIDataset(h5_path, keys_train, age_dist_train)
dataset_val = MRIDataset(h5_path, keys_val, age_dist_val)
dataset_test = MRIDataset(h5_path, keys_test, age_dist_test)

#dataloader
train_loader = DataLoader(dataset_train, batch_size=4, shuffle=True, num_workers=10, pin_memory=True) #DROP_LAST 
val_loader = DataLoader(dataset_val, batch_size=4, shuffle=False, num_workers=10, pin_memory=True) #DROP_LAST 
test_loader = DataLoader(dataset_test, batch_size=4, shuffle=False, num_workers=10, pin_memory=True) #DROP_LAST 

model = CNNmodel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#input_size = ([2, 1, 160, 192, 160])
#summary(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA is available:", torch.cuda.is_available())  # Print para verificar si CUDA está disponible
model = model.to(device)

num_epochs = 100

def calculate_mae(predictions, targets):
    return torch.mean(torch.abs(predictions - targets))

def plot_age_distribution(ages, subject_ids, bin_centers_list):
    
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

train_losses = []
val_losses = []
# Ciclo de entrenamiento 
for epoch in range(num_epochs):
    start_epoch = time.time()
    model.train()
    running_loss = 0.0
    running_mae = 0.0
    for inputs, ages, subject_ids in train_loader:
        #print(f'Input train data shape: {inputs.shape}')
        optimizer.zero_grad()
        inputs = inputs.to(device)
        outputs = model(inputs)
        x = outputs[0].cpu().view(-1, 1, 40)
        #print(f'ages: {ages.shape}')  #torch.Size([8, 1, 40]).
        #print(f'outputs: {x.shape}')  #torch.Size([8, 1, 40]).
        preds = []
        for i in range(inputs.size(0)):
            x_sample = x[i].detach().numpy().reshape(-1)
            prob = np.exp(x_sample)
            pred = prob @ bin_center_list[i]
            preds.append(pred)
            print(f'Predicción ajustada para el sujeto {subject_ids[i]}: {pred}')
        
        # Convertir a tensor y calcular el MAE
        preds = torch.tensor(preds)
        #mae = calculate_mae(preds, ages)
        loss = my_KLDivLoss(x, ages)
        #TODO CHANGE MAE CALCULATION. IT IS WRONG, FIRST, PROB = NP.EXP(X); PRED=PROB@BC, THEN PRED-TARGETS= MAE
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        #running_mae += mae.item()
        '''
        for i in range(inputs.size(0)):
            subject_id = subject_ids[i]
            age = ages[i].item()
            x_sample = x[i].numpy().reshape(-1)
            prob = np.exp(x_sample)
            pred = prob @ bin_center_list[i]
            plt.bar(bin_center_list[i], prob)
            plt.title(f'Subject ID: {subject_id}, Prediction: age={pred:.2f}')
            plt.xlabel('Age')
            plt.ylabel('Probability')
            plt.legend([f'Prediction for Subject ID {subject_id}'])
            plt.savefig(os.path.join(save_dir, f'prediction_epoch{epoch}_subject_{subject_id}.png'))
            plt.clf()
            '''
    # Calculando el loss de entrenamiento promedio por época
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    #train_mae = running_mae / len(train_loader)
    end_epoch = time.time()  # Termina el temporizador para esta época
    epoch_time = end_epoch - start_epoch  # Calcula el tiempo de ejecución de la época
    print(f'Epoch {epoch + 1}, Time: {epoch_time:.2f} seconds')
    print(f'Epoch {epoch + 1}, Train Loss: {train_loss}')
    
    # Fase de validación
    model.eval()
    val_loss = 0.0
    val_mae = 0.0
    with torch.no_grad():
        for inputs, ages, subject_ids in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            x = outputs[0].cpu().view(-1, 1, 40)
            mae = calculate_mae(x, ages)
            loss = my_KLDivLoss(x, ages)
            val_loss += loss.item()
            val_mae += mae.item()
    
    # Calculando el loss de validación promedio por época
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    val_mae /= len(val_loader)
    print(f'Epoch {epoch + 1}, Validation Loss: {val_loss}, Validation MAE: {val_mae}')
    #plot_age_distribution(ages, subject_ids, bin_center_list)
'''
    # Evaluation
    model.eval() # Don't forget this. BatchNorm will be affected if not in eval mode.
    with torch.no_grad():
        for inputs, ages, subject_ids in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            x = outputs[0].view(-1, 1, 40)
            for i in range(inputs.size(0)):  # Iterar sobre cada muestra en el lote
                subject_id = subject_ids[i]  # Obtener el ID del sujeto
                age = ages[i].item()  # Obtener la edad del sujeto
                x_sample = x[i].numpy().reshape(-1)
                prob = np.exp(x_sample)
                pred = prob @ bin_center_list[i]  # Calcular la predicción utilizando el producto punto con los bin_centers
                # Visualizar la distribución de probabilidad
                plt.bar(bin_center_list[i], prob)
                plt.title(f'Subject ID: {subject_id}, Prediction: age={pred:.2f}')
                plt.xlabel('Age')
                plt.ylabel('Probability')
                # Agregar el subject_id a la leyenda
                plt.legend([f'Prediction for Subject ID {subject_id}'])
                # Crear el directorio para guardar los archivos PNG si no existe
                # Guardar la visualización como un archivo PNG
                plt.savefig(os.path.join(save_dir, f'prediction_epoch{epoch}_subject_{subject_id}.png'))
                #Limpiar la figura actual para la siguiente iteración
                plt.clf()

            # Detener después de mostrar un número específico de ejemplos (por ejemplo, 5)
            # Si deseas mostrar más ejemplos, puedes modificar este número.
                if i == 5:
                    break
'''
    # Graficar y guardar las pérdidas
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig(os.path.join(save_dir, 'loss_plot.png'))  # Guardar la figura como un archivo .png
plt.close()  # Cerrar la figura actual para liberar memoria
print('Entrenamiento finalizado')

