from m_utils.data_transform import num2vect
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import matplotlib.pyplot as plt

# Ruta al archivo .h5 de mujeres
h5_path = '/home/usuaris/imatge/joan.manel.cardenas/MN_females_data.h5'
#h5_path = '/home/usuaris/imatge/joan.manel.cardenas/MN_males_data.h5'

with h5py.File(h5_path, 'r') as h5_file:
    #sigma = len(h5_file.keys())  # Número total de sujetos
    keys = list(h5_file.keys())
    ages = [h5_file[subject_id].attrs['Age'] for subject_id in keys]

# Suponiendo que ages ya ha sido definido como se mostró anteriormente
ages_nparrays = [np.array([age,]) for age in ages]

# Ver algunos ejemplos
for i, age_array in enumerate(ages_nparrays[:4]):  # Mostrar los primeros 5 para ejemplo
    subject_id = keys[i]
    print(f"Age {i}: {age_array}, Subject ID: {subject_id}")
    age_range = [42,82]
    age_step = 1  # Paso de edad
    sigma = 1
    age_dist, bin_centers = num2vect(age_array, age_range, age_step, sigma)
    age_dist = torch.tensor(age_dist, dtype=torch.float32)
    print(f'Label shape: {age_dist.shape}')
    y = age_dist.numpy().reshape(-1)
    plt.bar(bin_centers, y)
    plt.title(f'Subject ID: {subject_id}')
    plt.savefig(f'soft_label_{subject_id}.png')  # Guarda el gráfico en un archivo
    plt.close()  # Cierra la figura para liberar memoria
'''
# Calcular las distribuciones de edad
age_array = np.array(ages)
print(f"sujetos: {age_array}")
#age_min, age_max = int(np.floor(min(ages))), int(np.ceil(max(ages)))
age_range = [42,82]
age_step = 1  # Paso de edad
sigma = 1

print(f"Número total de sujetos (sigma): {len(age_array)}")
print(f"Rango de edad (age_range): {age_range}")
print(f"Edades después de la conversión a np.array: {age_array}")


# Ahora, llamar a num2vect para transformar las edades en distribuciones de probabilidad
age_dist, bin_centers = num2vect(age_array, age_range, age_step, sigma)
print(f"Distribuciones de probabilidad para las edades: {age_dist.shape}")
print(f"Centros de los bins: {bin_centers}")
'''