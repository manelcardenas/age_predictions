import os
import nibabel as nib
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import sys
sys.path.append('/home/usuaris/imatge/joan.manel.cardenas/age_predictions')
from m_utils.crop_center import center_crop
from model.loss import my_KLDivLoss
from model.model import CNNmodel  
from m_utils.data_transform import num2vect


# Directory containing the folders with .nii.gz files
zip_directory = '/mnt/work/datasets/UKBiobank/MN_Process_alfa_mni/Process_alfa_mni'

# Leer el archivo CSV con la información de los sujetos
demographics = pd.read_csv('/mnt/work/datasets/UKBiobank/MN_Process_alfa_mni/ALFA_PLUS_3D.csv')

# Extract the relevant columns
participants = demographics[['IdParticipante', 'Sexo', 'Edad_CI']]

# Initialize a list to store the cropped images
cropped_images = []

# Count the number of 'first.nii.gz' files
count = 0

# Iterate over each participant
for _, row in participants.iterrows():
    participant_id = str(int(row['IdParticipante']))
    sex = row['Sexo']
    age = row['Edad_CI']

    if sex == 2.0: #man=1, woman=2
        continue  # Saltar si el sexo es 1

    # Find the folder corresponding to the participant
    for folder_name in os.listdir(zip_directory):
        #print(participant_id)
        #print(folder_name)
        if folder_name.startswith(participant_id):
            folder_path = os.path.join(zip_directory, folder_name)
            
            # Check if it's a directory
            if os.path.isdir(folder_path):
                # Find the file that ends with 'first.nii.gz'
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('first.nii.gz'):
                        file_path = os.path.join(folder_path, file_name)
                        
                        # Load the NIfTI image
                        img = nib.load(file_path)
                        data = img.get_fdata()
                        
                        # Apply center crop
                        cropped_data = center_crop(data)
                        
                        # Store the cropped image and metadata in the list
                        cropped_images.append({
                            'data': cropped_data.astype(np.float32),
                            'IdParticipante': participant_id,
                            'Sexo': sex,
                            'Edad_CI': age
                        })
                        
                        # Increment the count
                        count += 1

print(f'Number of cropped images: {count}')
print(f'Number of participants: {participants.shape[0]}')
for i, image_info in enumerate(cropped_images[:10]):
    participant_info = {k: v for k, v in image_info.items() if k != 'data'}
    print(f'Info of participant {i+1}: {participant_info}')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNmodel().to(device)   

# Cargar el state_dict guardado
state_dict = torch.load('best_models/best_model_man.p')

# Crear un nuevo state_dict en el que las claves no tienen el prefijo "module."
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:]  # eliminar el prefijo 'module.'
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
model.eval()  # No olvides poner el modelo en modo evaluación

bin_range = [42, 82]
bin_step = 1
sigma = 1

predicted_ages = []
real_ages = []

# Iterar sobre las primeras 10 imágenes recortadas
for i, subject_info in enumerate(cropped_images[:168]):
    data = subject_info['data']
    age = subject_info['Edad_CI']

    data = data / data.mean()
    data = data[np.newaxis, np.newaxis, ...]
    input_data = torch.tensor(data, dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(input_data)
        x = output[0].cpu().view(-1, 1, 40)

    # Transformar la edad a etiqueta suave (distribución de probabilidad)
    label = np.array([age])  # La edad real del sujeto
    y, bc = num2vect(label, bin_range, bin_step, sigma)
    y = torch.tensor(y, dtype=torch.float32).cpu()

    # Calcular la pérdida (si se conoce la edad real)
    loss = my_KLDivLoss(x, y).numpy()

    # Predicción y visualización
    x = x.numpy().reshape(-1)
    prob = np.exp(x)
    pred = prob @ bc

    predicted_ages.append(pred)
    real_ages.append(age)

    # Visualización
    plt.figure()
    plt.bar(bc, prob)
    plt.title(f'Participant {i+1} - Prediction: age={pred:.2f}\nloss={loss}')
    plt.show()

    print(f'Participant {i+1} - Predicted Age: {pred:.2f}')
    print(f'Participant {i+1} - Real Age: {age}')
    print(f'Participant {i+1} - Loss: {loss}')


mae = np.mean(np.abs(np.array(predicted_ages) - np.array(real_ages)))
print(f'MAE: {mae:.2f}')