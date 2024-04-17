import os
import pandas as pd
import zipfile
import nibabel as nib
import numpy as np
import h5py
import sys
sys.path.append('/home/usuaris/imatge/joan.manel.cardenas/age_predictions')
from m_utils.crop_center import center_crop


# Leer el archivo CSV con la información de los sujetos
demographics = pd.read_csv('/home/usuaris/imatge/joan.manel.cardenas/age_predictions/demographics_icd_new_date3.csv')

# Ordenar los datos por el ID de los sujetos
demographics.loc[:, 'Label'] = demographics['ID'].astype(str)
reorder_demogs = demographics.sort_values('Label').reset_index(drop=True)

zip_directory = '/mnt/work/datasets/UKBiobank'
#output_directory = '/mnt/work/datasets/UKBiobank'
output_directory = '/home/usuaris/imatge/joan.manel.cardenas/'  

# Crear o abrir archivos .h5 de salida para hombres y mujeres
female_h5_path = os.path.join(output_directory, "one_females_data.h5")
male_h5_path = os.path.join(output_directory, "one_males_data.h5")

# Inicializar archivos .h5 para mujeres y hombres
female_h5_file = h5py.File(female_h5_path, 'w')
male_h5_file = h5py.File(male_h5_path, 'w')

for index, row in reorder_demogs.iterrows():
    subject_id = row['Label']
    sex = row['Sex']
    #actual_age = row['Age_modif']
    actual_age = round(row['Age_modif'], 2)

    # Limitar la ejecución para fines de prueba
    if int(subject_id) >= 1001000:
        break
    
    zip_filename = f"{subject_id}_20252_2_0.zip"
    zip_filepath = os.path.join(zip_directory, zip_filename)
    if os.path.exists(zip_filepath):
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            zip_file_contents = zip_ref.namelist()
            if 'T1/T1_brain_to_MNI.nii.gz' in zip_file_contents:
                temp_dir = '/tmp/extracted_files'
                os.makedirs(temp_dir, exist_ok=True)
                zip_ref.extract('T1/T1_brain_to_MNI.nii.gz', path=temp_dir)

                nifti_file_path = os.path.join(temp_dir, 'T1/T1_brain_to_MNI.nii.gz')
                img = nib.load(nifti_file_path)
                data = img.get_fdata()

                # Aplicar el recorte central
                cropped_data = center_crop(data)

                # Elegir el archivo .h5 basado en el género del sujeto
                h5_file = female_h5_file if sex == 0 else male_h5_file
                subject_group = h5_file.create_group(subject_id)
                subject_group.create_dataset('MRI', data=cropped_data)
                subject_group.attrs['Age'] = actual_age
                subject_group.attrs['Sex'] = 'Female' if sex == 0 else 'Male'
                
                print(f"Datos guardados para el ID: {subject_id}, Sexo: {sex}, Edad: {actual_age}, Tamaño de datos: {cropped_data.shape}")

# Asegurar el cierre de los archivos después de su uso
female_h5_file.close()
male_h5_file.close()

