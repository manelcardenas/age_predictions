import h5py
import numpy as np
import nibabel as nib
import torch
import sys
sys.path.append('/home/usuaris/imatge/joan.manel.cardenas/age_predictions')
from m_utils.data_transform import RandomShift, RandomMirror

def imprimir_info_sujetos(h5_path):
    with h5py.File(h5_path, 'r') as h5_file:
        print(f"Cantidad de sujetos: {len(h5_file)}")
        for subject_id in h5_file:
            subject_group = h5_file[subject_id]
            mri_data = subject_group['MRI'][:]
            age = subject_group.attrs['Age']
            sex = subject_group.attrs['Sex']
            print(f"ID: {subject_id}, Sexo: {sex}, Edad: {age}, Tamaño de datos MRI: {mri_data.shape}")

def crear_imagenes_mri(h5_path, output_directory, num_sujetos=3):
    with h5py.File(h5_path, 'r') as h5_file:
        sujetos_procesados = 0
        for subject_id in h5_file:
            if sujetos_procesados >= num_sujetos:
                break  # Salir del bucle después de procesar num_sujetos sujetos
            
            subject_group = h5_file[subject_id]
            mri_data = subject_group['MRI'][:]
            
            # Crear un objeto Nifti1Image con los datos de MRI y la matriz de afinidad
            img = nib.Nifti1Image(mri_data, np.eye(4))  # Usar matriz identidad para la afinidad si no se dispone de otra
            
            # Definir el nombre de archivo de salida
            output_filepath = f"{output_directory}/{subject_id}_MRI.nii"
            
            # Guardar la imagen en formato .nii
            nib.save(img, output_filepath)
            print(f"Imagen MRI guardada: {output_filepath}")
            
            sujetos_procesados += 1

def crear_imagenes_mri_con_da(h5_path, output_directory, transform, num_sujetos=5):
    with h5py.File(h5_path, 'r') as h5_file:
        sujetos_procesados = 0
        for subject_id in h5_file:
            if sujetos_procesados >= num_sujetos:
                break  # Salir del bucle después de procesar num_sujetos sujetos
            
            subject_group = h5_file[subject_id]
            mri_data = subject_group['MRI'][:]
            
            # Convertir los datos de MRI a un tensor de PyTorch
            mri_data_tensor = torch.from_numpy(mri_data).float().unsqueeze(0)  # [C, D, H, W]
            
            # Aplicar transformaciones
            mri_data_tensor = transform(mri_data_tensor)
            
            # Convertir de nuevo a numpy array para guardar la imagen
            mri_data = mri_data_tensor.squeeze(0).numpy()
            
            # Crear un objeto Nifti1Image con los datos de MRI y la matriz de afinidad
            img = nib.Nifti1Image(mri_data, np.eye(4))  # Usar matriz identidad para la afinidad si no se dispone de otra
            
            # Definir el nombre de archivo de salida
            output_filepath = f"{output_directory}/{subject_id}_MRI_DA.nii"
            
            # Guardar la imagen en formato .nii
            nib.save(img, output_filepath)
            print(f"Imagen MRI con DA guardada: {output_filepath}")
            
            sujetos_procesados += 1

# Uso de las funciones
#h5_path = '/home/usuaris/imatge/joan.manel.cardenas/MN_males_data.h5'
h5_path = '/mnt/work/datasets/UKBiobank/MN_females_data.h5'
output_directory = '/home/usuaris/imatge/joan.manel.cardenas/age_predictions/subjects_data/'

# LLamar 1a funcion: Imprimir información de los sujetos 
#imprimir_info_sujetos(h5_path)

# LLamar segunda funcion: Crear imágenes MRI para los tres primeros sujetos
#crear_imagenes_mri(h5_path, output_directory)



# Llamar a la función: Crear imágenes MRI para los tres primeros sujetos con Data Augmentation
from torchvision.transforms import Compose

train_transform = Compose([
    RandomShift(max_shift=2),  # Aplica RandomShift con una probabilidad del 50%
    RandomMirror()               # Aplica RandomMirror con una probabilidad del 50%
])

crear_imagenes_mri_con_da(h5_path, output_directory, train_transform)
