import h5py
import numpy as np

# Ruta al archivo .h5 de mujeres
female_h5_path = '/home/usuaris/imatge/joan.manel.cardenas/age_predictions/subjects_data/females_data.h5'

# Abrir el archivo .h5 de mujeres para lectura
with h5py.File(female_h5_path, 'r') as h5_file:
    # Iterar sobre cada grupo (sujeto) en el archivo
    for subject_id in h5_file:
        # Acceder al grupo del sujeto
        subject_group = h5_file[subject_id]
        
        # Leer los datos de MRI del sujeto
        mri_data = subject_group['MRI'][:]
        
        # Leer los atributos 'Age' y 'Sex'
        age = subject_group.attrs['Age']
        sex = subject_group.attrs['Sex']
        
        # Imprimir la información del sujeto
        print(f"ID: {subject_id}, Sexo: {sex}, Edad: {age}, Tamaño de datos MRI: {mri_data.shape}")

        # Si solo quieres probar con un sujeto para ver cómo funciona, puedes descomentar la siguiente línea
        # break
