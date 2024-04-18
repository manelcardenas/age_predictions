import h5py
import numpy as np
import nibabel as nib

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

# Uso de las funciones
#h5_path = '/home/usuaris/imatge/joan.manel.cardenas/MN_males_data.h5'
h5_path = '/mnt/work/datasets/UKBiobank/MN_males_data.h5'
output_directory = '/home/usuaris/imatge/joan.manel.cardenas/age_predictions/subjects_data/'

# LLamar 1a funcion: Imprimir información de los sujetos 
imprimir_info_sujetos(h5_path)

# LLamar segunda funcion: Crear imágenes MRI para los tres primeros sujetos
#crear_imagenes_mri(h5_path, output_directory)
