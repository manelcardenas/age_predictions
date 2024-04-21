import h5py
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from data_transform import num2vect

# Ruta al archivo .h5 de mujeres
#h5_path = '/home/usuaris/imatge/joan.manel.cardenas/MN_males_data.h5'
h5_path = '/mnt/work/datasets/UKBiobank/MN_males_data.h5'

save_dir = '/home/usuaris/imatge/joan.manel.cardenas/age_predictions/subjects_data/set_ages_dist/man_dt'
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

def verify_subject_ids(keys_train, keys_val, keys_test):
    # Ruta al archivo demographics.csv
    csv_path = '/home/usuaris/imatge/joan.manel.cardenas/age_predictions/demographics_icd_new_date3.csv'
    # Cargar el archivo CSV en un DataFrame de pandas
    demographics_df = pd.read_csv(csv_path)
    # Crear un conjunto de los IDs presentes en la columna 'ID' del DataFrame
    all_ids = set(demographics_df['ID'])

    # Convertir los IDs de cadenas a enteros
    keys_train_int = [int(subject_id) for subject_id in keys_train]
    keys_val_int = [int(subject_id) for subject_id in keys_val]
    keys_test_int = [int(subject_id) for subject_id in keys_test]

    # Comprobar si cada subject_id está presente en la columna 'ID' del DataFrame
    for subject_id in keys_train_int:
        if subject_id not in all_ids:
            print(f"El sujeto de train {subject_id} no está presente en la columna 'ID' del archivo demographics.csv.")
    
    for subject_id in keys_val_int:
        if subject_id not in all_ids:
            print(f"El sujeto de val {subject_id} no está presente en la columna 'ID' del archivo demographics.csv.")
    
    for subject_id in keys_test_int:
        if subject_id not in all_ids:
            print(f"El sujeto de test {subject_id} no está presente en la columna 'ID' del archivo demographics.csv.")
    
    print("Verificación completada.")
  
# Función para crear y guardar histogramas de edades
def save_age_distribution_plot(ages, dataset_name):
    plt.figure()
    plt.hist(ages, bins=20, color='skyblue', edgecolor='black')
    plt.title(f'Age Distribution in {dataset_name} Dataset')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.grid(True)
    save_path = os.path.join(save_dir, f'{dataset_name}_age_distribution.png')
    plt.savefig(save_path)
    plt.close()  # Cerrar la figura para liberar memoria

keys_train_val, keys_test, age_dist_train_val, age_dist_test = train_test_split(keys, age_dist_array, test_size=0.2, random_state=42) # (train,val) / test      stratify=age_dist
keys_train, keys_val, age_dist_train, age_dist_val = train_test_split(keys_train_val, age_dist_train_val, test_size=0.25, random_state=42)#,  train/val  stratify=age_dist_train

# Calcular las edades para cada conjunto de datos
with h5py.File(h5_path, 'r') as h5_file:
    train_ages = [h5_file[subject_id].attrs['Age'] for subject_id in keys_train]
    val_ages = [h5_file[subject_id].attrs['Age'] for subject_id in keys_val]
    test_ages = [h5_file[subject_id].attrs['Age'] for subject_id in keys_test]

# Llamada a la función verify_subject_ids
verify_subject_ids(keys_train, keys_val, keys_test)

# Guardar histogramas de edades para cada conjunto de datos
save_age_distribution_plot(train_ages, 'Training')
save_age_distribution_plot(val_ages, 'Validation')
save_age_distribution_plot(test_ages, 'Test')

