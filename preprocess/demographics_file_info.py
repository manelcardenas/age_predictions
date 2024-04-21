import os
import pandas as pd
import matplotlib.pyplot as plt

# Rutas
demographics_path = '/home/usuaris/imatge/joan.manel.cardenas/age_predictions/demographics_icd_new_date3.csv'
zip_directory = '/mnt/work/datasets/UKBiobank'

# Leer el archivo CSV con la información de los sujetos
demographics = pd.read_csv(demographics_path)

# Ordenar los datos por el ID de los sujetos
demographics.loc[:, 'Label'] = demographics['ID'].astype(str)
reorder_demogs = demographics.sort_values('Label').reset_index(drop=True)

# Filtrar sujetos cuyos datos están disponibles en zip_directory
available_subjects = [filename.split('_')[0] for filename in os.listdir(zip_directory) if filename.endswith('.zip')]
filtered_demographics = reorder_demogs[reorder_demogs['Label'].isin(available_subjects)]

# Contar el número de hombres (1) y mujeres (0) de los sujetos filtrados
sex_counts = filtered_demographics['Sex'].value_counts().rename(index={0: 'Mujeres', 1: 'Hombres'})

# Gráfica de cantidad de hombres y mujeres
plt.figure(figsize=(10, 5))
sex_counts.plot(kind='bar')
plt.title('Cantidad de Hombres y Mujeres en Datos Disponibles')
plt.xlabel('Sexo')
plt.ylabel('Cantidad')
plt.xticks(rotation=0)
plt.savefig('/home/usuaris/imatge/joan.manel.cardenas/age_predictions/subjects_data/sex_distribution.png')
plt.show()

# Distribución de edad para mujeres
plt.figure(figsize=(10, 5))
ages_women = filtered_demographics[filtered_demographics['Sex'] == 0]['Age']
plt.hist(ages_women, bins=20, color='pink', edgecolor='red')
plt.title('Distribución de Edad para Mujeres')
plt.xlabel('Edad')
plt.ylabel('Cantidad')
plt.savefig('/home/usuaris/imatge/joan.manel.cardenas/age_predictions/subjects_data/age_distribution_women.png')
plt.show()

# Distribución de edad para hombres
plt.figure(figsize=(10, 5))
ages_men = filtered_demographics[filtered_demographics['Sex'] == 1]['Age']
plt.hist(ages_men, bins=20, color='lightblue', edgecolor='blue')
plt.title('Distribución de Edad para Hombres')
plt.xlabel('Edad')
plt.ylabel('Cantidad')
plt.savefig('/home/usuaris/imatge/joan.manel.cardenas/age_predictions/subjects_data/age_distribution_men.png')
plt.show()
