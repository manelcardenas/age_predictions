import numpy as np

def find_image_bounds(image, subject_id):
    """
    Encuentra los límites de una imagen donde el valor del pixel es diferente de cero,
    e imprime los valores máximos y mínimos junto con el ID del sujeto.
    """
    non_zero_points = np.where(image != 0)
    bounds = []
    for dim in range(image.ndim):
        min_dim = np.min(non_zero_points[dim])
        max_dim = np.max(non_zero_points[dim])
        bounds.append((min_dim, max_dim))
        print(f"Subject ID: {subject_id}, Dimension: {dim}, Min: {min_dim}, Max: {max_dim}")
    return bounds

def find_absolute_bounds(images, subject_ids):
    """
    Encuentra los límites absolutos de un conjunto de imágenes, teniendo en cuenta los IDs de los sujetos.
    """
    all_bounds = [find_image_bounds(image, subject_id) for image, subject_id in zip(images, subject_ids)]
    abs_bounds = []
    for dim in range(3): 
        min_dim = min(bounds[dim][0] for bounds in all_bounds)
        max_dim = max(bounds[dim][1] for bounds in all_bounds)
        abs_bounds.append((min_dim, max_dim))
    # Imprimir los límites absolutos
    print("Límites absolutos por dimensión:")
    for dim, (min_dim, max_dim) in enumerate(abs_bounds):
        print(f"Dimensión {dim}: Mínimo Absoluto = {min_dim}, Máximo Absoluto = {max_dim}")
    return abs_bounds


def crop_images(images, abs_bounds):
    """
    Recorta imágenes basadas en los límites absolutos dados.
    """
    cropped_images = []
    for image in images:
        crop = image[abs_bounds[0][0]:abs_bounds[0][1]+1, 
                     abs_bounds[1][0]:abs_bounds[1][1]+1, 
                     abs_bounds[2][0]:abs_bounds[2][1]+1]
        cropped_images.append(crop)
    return cropped_images

def process_female_files(brains_tmp, subj_id):
    # Encuentra los límites absolutos para todas las imágenes, considerando los IDs de los sujetos
    abs_bounds = find_absolute_bounds(brains_tmp, subj_id)
    # Recorta todas las imágenes según los límites absolutos
    cropped_brains = crop_images(brains_tmp, abs_bounds)
    return cropped_brains



