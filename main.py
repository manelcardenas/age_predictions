import torch
import wandb
import numpy as np

import torch.multiprocessing as mp


from model.model import CNNmodel
from trainer import Trainer

#init_process_group(backend='nccl', init_method='tcp://192.168.41.246:2222')
#dist.init_process_group(backend='nccl', init_method='tcp://192.168.41.246:2222', rank=0, world_size=4)
#wandb.init(project='Brain_age', entity='manelcardenas')


def main(rank: int, world_size: int):
    h5_path = '/mnt/work/datasets/UKBiobank/MN_males_data.h5'
    save_dir = 'subjects_data'
    num_epochs = 110
    batch_size = 4
    trainer = Trainer(rank, world_size=world_size, h5_path=h5_path, save_dir=save_dir, bs=batch_size)
    trainer.init_distributed_training()
    train_loader, val_loader, test_loader = trainer.prepare_datasets_and_loaders(bs=batch_size)
    model = CNNmodel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)
    trainer.train(model, optimizer, train_loader, val_loader, num_epochs=num_epochs)



if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(world_size)
    mp.spawn(main, args=(world_size,), nprocs=world_size)

