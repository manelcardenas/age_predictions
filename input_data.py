from preprocess.data_transform import num2vect
import h5py
import numpy as np
from scipy.stats import norm

# Ruta al archivo .h5 de mujeres
female_h5_path = '/home/usuaris/imatge/joan.manel.cardenas/females_data.h5'

with h5py.File(female_h5_path, 'r') as h5_file:
    sigma = len(h5_file.keys())  # Número total de sujetos
    
    ages = []  # Lista para recopilar todas las edades
    for subject_id, subject_group in h5_file.items():
        age = subject_group.attrs['Age']
        ages.append(age)

    # Redondear y convertir a enteros los mínimos y máximos
    age_min = int(np.floor(min(ages)))
    age_max = int(np.ceil(max(ages)))
    age_range = [age_min, age_max]
    age_step = 1  # Paso de edad

    print(f"Número total de sujetos (sigma): {sigma}")
    print(f"Rango de edad (age_range): {age_range}")

    # Convertir la lista de edades a np.array
    age_array = np.array(ages)
    #print(f"Total de edades antes de la conversión a np.array: {len(ages)}")
    print(f"Edades después de la conversión a np.array: {age_array}")

    # Llamada a num2vect para el array completo de edades
    age_dist, bin_centers = num2vect(age_array, age_range, age_step, sigma)
    print(f"Distribuciones de probabilidad para las edades: {age_dist}")
    print(f"Centros de los bins: {bin_centers}")




'''

y = torch.tensor(y, dtype=torch.float32)
print(f'Label shape: {y.shape}')

# Nuevo tamaño deseado para las imágenes
new_shape = (160, 192, 160)

# Lista para almacenar las imágenes recortadas
brains_cropped = []

# Recortar cada imagen para centrarla en el nuevo tamaño
for brain in brains_tmp:
    start = tuple(map(lambda a, da: a//2-da//2, brain.shape, new_shape))    #slice(11, 171)
    end = tuple(map(lambda start, size: start+size, start, new_shape))      #slice(13, 205)
    slices = tuple(slice(s, e) for s, e in zip(start, end))
    cropped_brain = brain[slices]
    brains_cropped.append(cropped_brain)
'''
    

'''
##TESTING
# Guardar las imágenes en formato .nii después de las modificaciones
for i, brain in enumerate(brains_cropped):
    output_path = f'/home/usuaris/imatge/joan.manel.cardenas/age_predictions/subjects_data/{subj_id[i]}_after_crop.nii'
    nifti_img = nib.Nifti1Image(brain.squeeze(), np.eye(4))  # Crear un objeto Nifti1Image
    nib.save(nifti_img, output_path)  # Guardar la imagen en formato .nii
'''