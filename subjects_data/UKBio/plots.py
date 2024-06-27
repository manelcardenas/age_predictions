import matplotlib.pyplot as plt
import re


# Datos proporcionados
real_ages = []
predicted_ages = []

file_path = '/home/usuaris/imatge/joan.manel.cardenas/age_predictions/test_results/test/res_male_DA_bias_corr.txt'

# Leer el archivo y extraer las edades
with open(file_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        real_age_match = re.search(r'Real Age:\s*([0-9.]+)', line)
        predicted_age_match = re.search(r'Predicted Age:\s*([0-9.]+)', line)
        if real_age_match and predicted_age_match:
            real_ages.append(float(real_age_match.group(1)))
            predicted_ages.append(float(predicted_age_match.group(1)))

# Crear la figura y el eje
fig, ax = plt.subplots()

# Crear el gráfico de dispersión
ax.scatter(real_ages, predicted_ages, c='lightgreen', edgecolors='black', linewidths=0.5, label='Predicted vs Real')

# Crear la línea donde X = Y
min_age = min(min(real_ages), min(predicted_ages))
max_age = max(max(real_ages), max(predicted_ages))
ax.plot([min_age, max_age], [min_age, max_age], color='red', linestyle='--', label='X = Y')

# Añadir etiquetas y título
ax.set_xlabel('Cronological Age')
ax.set_ylabel('Predicted Age')

# Mostrar la gráfica

plt.savefig('res_male_DA_bias_corr.png') # Guardar la figura como un archivo .png
plt.close()  # Cerrar la figura actual para liberar memoria