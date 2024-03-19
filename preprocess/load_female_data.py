import os
import pandas as pd
import zipfile
import nibabel as nib
import numpy as np


def female_data():
    # Leer el archivo CSV con la información de los sujetos
    demographics = pd.read_csv('/home/usuaris/imatge/joan.manel.cardenas/age_predictions/demographics_icd_new_date3.csv')

    # Ordenar los datos por el ID de los sujetos
    demographics.loc[:, 'Label'] = demographics['ID'].astype(str)
    reorder_demogs = demographics.sort_values('Label').reset_index(drop=True)

    # Inicializar la lista para almacenar la información de los archivos de mujeres
    female_files_info = []
    zip_directory = '/mnt/work/datasets/UKBiobank'

    for index, row in reorder_demogs.iterrows():
        subject_id = row['Label']
        sex = row['Sex']
        actual_age = row['Age_modif']
        #TESTING
        if int(subject_id) >= 1003000:
            break
        # Solo procesar si el sujeto es una mujer (sex == 0)
        if sex == 0:
            zip_filename = f"{subject_id}_20252_2_0.zip"
            zip_filepath = os.path.join(zip_directory, zip_filename)
            if os.path.exists(zip_filepath):
                # Abrir el archivo .zip
                with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
                    # Lista de nombres de los archivos en el .zip
                    zip_file_contents = zip_ref.namelist()
                    # Verificar si el archivo 'T1_brain_to_MNI.nii.gz' está presente en el .zip
                    if 'T1/T1_brain_to_MNI.nii.gz' in zip_file_contents:
                        # Directorio temporal para extraer los archivos
                        temp_dir = '/tmp/extracted_files'
                        os.makedirs(temp_dir, exist_ok=True)
                        # Extraer el archivo T1_brain_to_MNI.nii.gz al directorio temporal
                        zip_ref.extract('T1/T1_brain_to_MNI.nii.gz', path=temp_dir)
                        # Cargar la imagen NIfTI usando nibabel
                        nifti_file_path = os.path.join(temp_dir, 'T1/T1_brain_to_MNI.nii.gz')
                        img = nib.load(nifti_file_path)
                        # Obtener los datos de la imagen
                        data = img.get_fdata()
                        '''
                        ## TESTING Guardar la imagen como un nuevo archivo .nii
                        zoomed_img = nib.Nifti1Image(data.astype(np.float32), img.affine) 
                        output_path = f'/home/usuaris/imatge/joan.manel.cardenas/age_predictions/subjects_data/{subject_id}_before_crop.nii'
                        nib.save(zoomed_img, output_path)
                        
                        print(f"ID: {subject_id}, Tamaño de data: {data.shape}, Edad: {actual_age}")
                        '''
                        # Agregar información a la lista correspondiente
                        file_info = (subject_id, data, actual_age)
                        female_files_info.append(file_info)    #[N elementos de (ID, (182, 218, 182), edad_real)]

    return female_files_info


#resultados = female_data()