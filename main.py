from preprocess.load_female_data import female_data
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

# Nuevo tamaño deseado para las imágenes
new_shape = (160, 192, 160)

# Lista para almacenar las imágenes recortadas
brains_cropped = []

# Recortar cada imagen para centrarla en el nuevo tamaño
for brain in brains_tmp:
    start = tuple(map(lambda a, da: a//2-da//2, brain.shape, new_shape))   #slice(11, 171)
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