from preprocess.load_female_data import female_data


female_info_list = []
brains_tmp = []
subj_id = []
female_info_list = female_data()

# Obtener los IDs de los sujetos
subj_id = [file_info[0] for file_info in female_info_list]

# Obtener los datos de las imágenes MRI
brains_tmp = [file_info[1] for file_info in female_info_list]