import torch
from torch.utils.data import DataLoader

from dataset.coin_text_dataset import coin_text_dataset

#create dataset
def create_dataset(dataset, config):
    if dataset=='coin_text':          
        train_dataset = coin_text_dataset(config['train_file'])
        test_dataset = coin_text_dataset(config['test_file'])                
        return train_dataset, test_dataset

#create dataloader
def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    