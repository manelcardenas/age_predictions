import torch
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from sklearn.linear_model import LinearRegression

from model.loss import my_KLDivLoss
from model.model import CNNmodel
from m_utils.data_transform import *
from model.mri_dataset import MRIDataset

h5_path = '/mnt/work/datasets/UKBiobank/MN_males_data.h5'

save_dir = 'subjects_data'
os.makedirs(save_dir, exist_ok=True)

with h5py.File(h5_path, 'r') as h5_file:
    keys = list(h5_file.keys())
    ages = [h5_file[subject_id].attrs['Age'] for subject_id in keys]

age_range = [42, 82]
age_step = 1
sigma = 1
age_dist_list = []
bin_center_list = []

for age in ages:
    age_array = np.array([age])
    age_dist, bc = num2vect(age_array, age_range, age_step, sigma)
    age_dist_list.append(age_dist)
    bin_center_list = bc  # Updated to store bin centers only once

# Convert the list of distributions to a numpy array for easier handling later
age_dist_array = np.array(age_dist_list).squeeze()  # squeeze to remove extra dimensions

train_transform = Compose([
    RandomShift(max_shift=2, p=0.5),
    RandomMirror(p=0.5)
])

keys_train_val, keys_test, age_dist_train_val, age_dist_test = train_test_split(keys, age_dist_array, test_size=0.2, random_state=42)
keys_train, keys_val, age_dist_train, age_dist_val = train_test_split(keys_train_val, age_dist_train_val, test_size=0.125, random_state=42)

dataset_train = MRIDataset(h5_path, keys_train, age_dist_train, transform=train_transform)
dataset_val = MRIDataset(h5_path, keys_val, age_dist_val)
dataset_test = MRIDataset(h5_path, keys_test, age_dist_test)

train_loader = DataLoader(dataset_train, batch_size=8, shuffle=True, num_workers=10, pin_memory=True)
val_loader = DataLoader(dataset_val, batch_size=8, shuffle=False, num_workers=10, pin_memory=True)
test_loader = DataLoader(dataset_test, batch_size=8, shuffle=False, num_workers=10, pin_memory=True)

print(f'Number of samples in the test set: {len(dataset_test)}')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNmodel()

state_dict = torch.load('best_models/best_model_male_DA.p')
from collections import OrderedDict
new_state_dict = OrderedDict()

for k, v in state_dict.items():
    name = k[7:]  # Remove the prefix 'module.'
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
model = model.to(device)

def calculate_mae(predictions, targets):
    return torch.mean(torch.abs(predictions - targets))

model.eval()
test_loss = 0.0
test_mae = 0.0
all_predictions = []
all_real_ages = []
all_output_dists = []

with torch.no_grad():
    for inputs, age_dist, age_real, subject_ids in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        x = outputs[0].cpu().view(-1, 1, 40)

        preds = []
        output_dists = []
        for i in range(inputs.size(0)):
            x_sample = x[i].detach().numpy().reshape(-1)
            prob = np.exp(x_sample)
            pred = prob @ bin_center_list
            preds.append(pred)
            output_dists.append(prob)
        
        preds = torch.tensor(preds)
        preds_rounded = torch.round(preds * 100) / 100  
        mae = calculate_mae(preds_rounded, age_real)  
        loss = my_KLDivLoss(x, age_dist)
        
        test_loss += loss.item()
        test_mae += mae.item()

        all_predictions.extend(preds_rounded.tolist())
        all_real_ages.extend(age_real.tolist())
        all_output_dists.extend(output_dists)

test_loss /= len(test_loader)
test_mae /= len(test_loader)
print(f'Test Loss: {test_loss}, Test MAE: {test_mae}')

# Perform linear regression on the validation set
chronological_ages_val = np.array(all_real_ages)
predicted_ages_val = np.array(all_predictions)

reg = LinearRegression().fit(chronological_ages_val.reshape(-1, 1), predicted_ages_val)
alpha = reg.coef_[0]
beta = reg.intercept_

print(f"Slope (α): {alpha}")
print(f"Intercept (β): {beta}")

# Function to apply bias correction
def correct_predicted_age(predicted_age, chronological_age, alpha, beta):
    return predicted_age + (chronological_age - (alpha * chronological_age + beta))

# Apply the model to the test set and apply bias correction
test_loss = 0.0
test_mae = 0.0
corrected_predictions = []
corrected_real_ages = []

with torch.no_grad():
    for inputs, age_dist, age_real, subject_ids in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        x = outputs[0].cpu().view(-1, 1, 40)

        preds = []
        for i in range(inputs.size(0)):
            x_sample = x[i].detach().numpy().reshape(-1)
            prob = np.exp(x_sample)
            pred = prob @ bin_center_list
            preds.append(pred)
        
        preds = torch.tensor(preds)
        preds_rounded = torch.round(preds * 100) / 100
        corrected_preds = correct_predicted_age(preds_rounded, age_real, alpha, beta)
        mae = calculate_mae(corrected_preds, age_real)
        loss = my_KLDivLoss(x, age_dist)

        test_loss += loss.item()
        test_mae += mae.item()

        corrected_predictions.extend(corrected_preds.tolist())
        corrected_real_ages.extend(age_real.tolist())

test_loss /= len(test_loader)
test_mae /= len(test_loader)
print(f'Corrected Test Loss: {test_loss}, Corrected Test MAE: {test_mae}')

# Print the real age and corrected prediction for each subject
for real_age, corrected_prediction in zip(corrected_real_ages, corrected_predictions):
    print(f'Real Age: {real_age}, Corrected Predicted Age: {corrected_prediction}')
'''
# Plot the age distribution of subjects in the training dataset
train_ages = []
for key in keys_train:
    with h5py.File(h5_path, 'r') as h5_file:
        age = h5_file[key].attrs['Age']
        train_ages.append(int(age))  # Use only the integer part of the age

plt.figure(figsize=(10, 5))
plt.hist(train_ages, bins=range(42, 83), alpha=0.7, color='green', edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Number of Subjects')
plt.title('Age Distribution of Subjects in the Training Dataset')
plt.savefig(f'subjects_data/train_age_distribution.png')
plt.show()

# Verify and correct soft labels and real ages alignment
soft_labels_corrected = []
for real_age in corrected_real_ages[:5]:  # Check only the first five subjects
    age_array = np.array([real_age])
    age_dist, _ = num2vect(age_array, age_range, age_step, sigma)
    soft_labels_corrected.append(age_dist.flatten())

# Plot the soft labels and output probability distributions for the first five subjects
for i in range(5):
    plt.figure(figsize=(10, 5))
    plt.bar(bin_center_list, soft_labels_corrected[i], label='Soft Label', alpha=0.5, color='blue')
    plt.bar(bin_center_list, all_output_dists[i], label='Output Distribution', alpha=0.5, color='orange')
    plt.xlabel('Age')
    plt.ylabel('Probability')
    plt.title(f'Subject {i+1}: Real Age {corrected_real_ages[i]}, Corrected Predicted Age {corrected_predictions[i]}')
    plt.legend()
    plt.savefig(f'subjects_data/subject_{i+1}.png')
    plt.show()
'''