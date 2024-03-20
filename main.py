from preprocess.load_female_data import female_data
import nibabel as nib
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from model.model import CNNmodel
from sklearn.model_selection import train_test_split

female_info_list = female_data()

# Array IDs de los sujetos
subj_id = [file_info[0] for file_info in female_info_list]

# Array datos de las imágenes MRI
brains_tmp = [file_info[1] for file_info in female_info_list]

# Array edades de los sujetos
subj_age = [file_info[2] for file_info in female_info_list]

# Nuevo tamaño deseado para las imágenes
new_shape = (160, 192, 160)

# Lista para almacenar las imágenes recortadas
brains_cropped = []

# Recortar cada imagen para centrarla en el nuevo tamaño
for brain in brains_tmp:
    start = tuple(map(lambda a, da: a//2-da//2, brain.shape, new_shape))    #slice(11, 171)
    end = tuple(map(lambda start, size: start+size, start, new_shape))      #slice(13, 205)
    slices = tuple(slice(s, e) for s, e in zip(start, end))
    cropped_brain = brain[slices]
    brains_cropped.append(cropped_brain)

'''
##TESTING
# Guardar las imágenes en formato .nii después de las modificaciones
for i, brain in enumerate(brains_cropped):
    output_path = f'/home/usuaris/imatge/joan.manel.cardenas/age_predictions/subjects_data/{subj_id[i]}_after_crop.nii'
    nifti_img = nib.Nifti1Image(brain.squeeze(), np.eye(4))  # Crear un objeto Nifti1Image
    nib.save(nifti_img, output_path)  # Guardar la imagen en formato .nii
'''

# Convertir listas a tensores
brains_tensor = np.stack(brains_cropped, axis=0)  # (num sujetos, 160,192,160)
ages_tensor = np.array(subj_age)


# División en conjuntos de entrenamiento, validación y prueba
brains_train_val, brains_test, ages_train_val, ages_test = train_test_split(brains_tensor, ages_tensor, test_size=0.15, random_state=42)
brains_train, brains_val, ages_train, ages_val = train_test_split(brains_train_val, ages_train_val, test_size=0.176, random_state=42)  # 0.176 es aproximadamente 15% del 85% restante

class BrainMRI3DDataset(Dataset):
    def __init__(self, brains, ages):   
        self.brains = brains
        self.ages = ages

    def __len__(self):     #num total de sujetos
        return len(self.brains)

    def __getitem__(self, idx):    #devuelve la edad y la MRI del sujeto idx
        brain = self.brains[idx]
        age = self.ages[idx]
        brain_tensor = torch.from_numpy(brain).float().unsqueeze(0)   # Añade un canal y convierte a tensor
        age_tensor = torch.tensor(age, dtype=torch.float)
        return brain_tensor, age_tensor

# Creación de DataLoaders para cada conjunto
train_dataset = BrainMRI3DDataset(brains_train, ages_train)
val_dataset = BrainMRI3DDataset(brains_val, ages_val)
test_dataset = BrainMRI3DDataset(brains_test, ages_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = CNNmodel()
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

# Ciclo de entrenamiento con validación
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, ages in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs[0].squeeze(), ages)
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
            loss = loss_function(outputs[0].squeeze(), ages)
            val_loss += loss.item()
    
    # Calculando el loss de validación promedio por época
    val_loss /= len(val_loader)
    print(f'Epoch {epoch + 1}, Validation Loss: {val_loss}')

print('Entrenamiento finalizado')
