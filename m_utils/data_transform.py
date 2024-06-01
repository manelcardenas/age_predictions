import numpy as np
import torch
import h5py
import random


from scipy.stats import norm

def calculate_mae(predictions, targets):
        return torch.mean(torch.abs(predictions - targets))

def adjust_learning_rate(optimizer, epoch):
        lr = 0.01 * (0.3 ** (epoch // 30))  # Multiplicar por 0.3 cada 30 épocas
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def num2vect(x, age_range, age_step, sigma):
    """
    v,bin_centers = number2vector(x,bin_range,bin_step,sigma)
    age_range: (start, end), size-2 tuple
    age_step: should be a divisor of |end-start|
    sigma:
    = 0 for 'hard label', v is index
    > 0 for 'soft label', v is vector
    < 0 for error messages.
    """
    bin_start = age_range[0]
    bin_end = age_range[1]
    bin_length = bin_end - bin_start
    if not bin_length % age_step == 0:
        print("age's range should be divisible by bin_step!")
        return -1
    bin_number = int(bin_length / age_step)
    bin_centers = bin_start + float(age_step) / 2 + age_step * np.arange(bin_number)

    if sigma == 0:
        x = np.array(x)
        i = np.floor((x - bin_start) / age_step)
        i = i.astype(int)
        return i, bin_centers
    elif sigma > 0:
        if np.isscalar(x):
            v = np.zeros((bin_number,))
            for i in range(bin_number):
                x1 = bin_centers[i] - float(age_step) / 2
                x2 = bin_centers[i] + float(age_step) / 2
                cdfs = norm.cdf([x1, x2], loc=x, scale=sigma)
                v[i] = cdfs[1] - cdfs[0]
            return v, bin_centers
        else:
            v = np.zeros((len(x), bin_number))
            for j in range(len(x)):
                for i in range(bin_number):
                    x1 = bin_centers[i] - float(age_step) / 2
                    x2 = bin_centers[i] + float(age_step) / 2
                    cdfs = norm.cdf([x1, x2], loc=x[j], scale=sigma)
                    v[j, i] = cdfs[1] - cdfs[0]
            return v, bin_centers
        

def get_age_distribution(h5_path, age_range=[42, 82], age_step=1, sigma=1):
    with h5py.File(h5_path, 'r') as h5_file:
        keys = list(h5_file.keys())
        ages = [h5_file[subject_id].attrs['Age'] for subject_id in keys]

    age_dist_list = []
    bin_center_list = []

    for age in ages:
        age_array = np.array([age])
        age_dist, bc = num2vect(age_array, age_range, age_step, sigma)
        age_dist_list.append(age_dist)
        bin_center_list.append(bc)

    # Convertir la lista de distribuciones a un arreglo de numpy para facilitar el manejo posterior
    age_dist_array = np.array(age_dist_list)

    return age_dist_array, keys, bin_center_list

class RandomShift:
    def __init__(self, max_shift=2 , p=0.5):
        self.max_shift = max_shift
        self.p = p

    def __call__(self, mri_data_tensor):
        if random.random() < self.p:
            shift = np.random.randint(-self.max_shift, self.max_shift + 1, size=3)
            mri_data_tensor = np.roll(mri_data_tensor.numpy(), shift, axis=(1, 2, 3))
            mri_data_tensor = torch.from_numpy(mri_data_tensor).float()
        return mri_data_tensor



class RandomMirror:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, mri_data_tensor):
        if random.random() < self.p:
            mri_data_tensor = torch.flip(mri_data_tensor, dims=[1])  # Flip along the sagittal plane
        return mri_data_tensor


