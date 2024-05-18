import torch
import os
import time
import h5py
import numpy as np


from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from sklearn.model_selection import train_test_split

from m_utils.data_transform import *
from model.mri_dataset import MRIDataset
from model.loss import my_KLDivLoss

class Trainer:
    def __init__(self, rank, world_size, h5_path=None, save_dir=None, bs=None):
        self.rank = rank
        self.world_size = world_size
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.h5_path = h5_path
        self.save_dir = save_dir
        self.bs = bs
        os.makedirs(self.save_dir, exist_ok=True)

        self.init_distributed_training()
        self.prepare_datasets_and_loaders(bs)

    def init_distributed_training(self):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "2222"
        init_process_group(backend="nccl", rank=self.rank, world_size=self.world_size)
        torch.cuda.set_device(self.rank)


    def prepare_datasets_and_loaders(self, bs):
        with h5py.File(self.h5_path, 'r') as h5_file:
            keys = list(h5_file.keys())
            ages = [h5_file[subject_id].attrs['Age'] for subject_id in keys]

        age_range = [42, 82]
        age_step = 1
        sigma = 1
        age_dist_list = []
        self.bin_center_list = []

        for age in ages:
            age_array = np.array([age, ])
            age_dist, bc = num2vect(age_array, age_range, age_step, sigma)
            age_dist_list.append(age_dist)
            self.bin_center_list.append(bc)

        age_dist_array = np.array(age_dist_list)

        keys_train_val, keys_test, age_dist_train_val, age_dist_test = train_test_split(keys, age_dist_array, test_size=0.2, random_state=42)
        keys_train, keys_val, age_dist_train, age_dist_val = train_test_split(keys_train_val, age_dist_train_val, test_size=0.125, random_state=42)

        self.dataset_train = MRIDataset(self.h5_path, keys_train, age_dist_train)
        self.dataset_val = MRIDataset(self.h5_path, keys_val, age_dist_val)
        self.dataset_test = MRIDataset(self.h5_path, keys_test, age_dist_test)

        train_sampler = DistributedSampler(self.dataset_train)
        val_sampler = DistributedSampler(self.dataset_val)
        test_sampler = DistributedSampler(self.dataset_test)

        self.train_loader = DataLoader(self.dataset_train, batch_size=bs, sampler=train_sampler, num_workers=4, pin_memory=True)
        self.val_loader = DataLoader(self.dataset_val, batch_size=bs, sampler=val_sampler, num_workers=4, pin_memory=True)
        self.test_loader = DataLoader(self.dataset_test, batch_size=bs, sampler=test_sampler, num_workers=4, pin_memory=True)

    def train(self, model, optimizer, train_loader, val_loader, num_epochs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        if torch.cuda.device_count() > 1 and device.type == 'cuda':
            print("Using", torch.cuda.device_count(), "GPUs for model parallelization.")
            model = DDP(model, device_ids=[self.rank])

        best_val_mae = float('inf')
        best_model_path = 'best_model.pth'

        for epoch in range(num_epochs):
            start_epoch = time.time()
            model.train()
            adjust_learning_rate(optimizer, epoch)
            running_loss = 0.0
            running_mae = 0.0

            for inputs, age_dist, age_real in train_loader:
                optimizer.zero_grad()
                inputs = inputs.to(device)
                outputs = model(inputs)
                x = outputs[0].cpu().view(-1, 1, 40)

                preds = []
                for i in range(inputs.size(0)):
                    x_sample = x[i].detach().numpy().reshape(-1)
                    prob = np.exp(x_sample)
                    pred = prob @ self.bin_center_list[i]
                    preds.append(pred)

                preds = torch.tensor(preds)
                preds_rounded = torch.round(preds * 100) / 100
                mae = calculate_mae(preds_rounded, age_real)
                loss = my_KLDivLoss(x, age_dist)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_mae += mae.item()

            train_loss = running_loss / len(train_loader)
            train_mae = running_mae / len(train_loader)

            end_epoch = time.time()
            epoch_time = end_epoch - start_epoch
            print(f'Epoch {epoch + 1}, Time: {epoch_time:.2f} seconds')
            print(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Train MAE: {train_mae}')

            model.eval()
            val_loss = 0.0
            val_mae = 0.0
            with torch.no_grad():
                for inputs, age_dist, age_real in val_loader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    x = outputs[0].cpu().view(-1, 1, 40)

                    preds = []
                    for i in range(inputs.size(0)):
                        x_sample = x[i].detach().numpy().reshape(-1)
                        prob = np.exp(x_sample)
                        pred = prob @ self.bin_center_list[i]
                        preds.append(pred)

                    preds = torch.tensor(preds)
                    preds_rounded = torch.round(preds * 100) / 100
                    mae = calculate_mae(preds_rounded, age_real)
                    loss = my_KLDivLoss(x, age_dist)

                    val_loss += loss.item()
                    val_mae += mae.item()

            val_loss /= len(val_loader)
            val_mae /= len(val_loader)
            print(f'Epoch {epoch + 1}, Val Loss: {val_loss}, Val MAE: {val_mae}')

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                torch.save(model.state_dict(), best_model_path)

        print('Training finished')