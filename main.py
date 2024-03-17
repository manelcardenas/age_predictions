from preprocess.load_female_data import female_data
from preprocess.crop_center import process_female_files
import nibabel as nib
import os
import numpy as np


female_info_list = []
brains_tmp = []
subj_id = []
female_info_list = female_data()

# Array IDs de los sujetos
subj_id = [file_info[0] for file_info in female_info_list]

# Array datos de las imágenes MRI
brains_tmp = [file_info[1] for file_info in female_info_list]

# Array edades de los sujetos
subj_age = [file_info[2] for file_info in female_info_list]

brains_cropped = process_female_files(brains_tmp, subj_id)

'''
##TESTING
# Guardar las imágenes en formato .nii después de las modificaciones
for i, brain in enumerate(brains_cropped):
    output_path = f'/home/usuaris/imatge/joan.manel.cardenas/age_predictions/subjects_data/{subj_id[i]}_after_crop.nii'
    nifti_img = nib.Nifti1Image(brain.squeeze(), np.eye(4))  # Crear un objeto Nifti1Image
    nib.save(nifti_img, output_path)  # Guardar la imagen en formato .nii
'''