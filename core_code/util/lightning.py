import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader

from core_code.util.default_config import init_config_training
import core_code.util.helpers as util




class CustomDataset(Dataset):

    def __init__(self, x, y, task_activity):
        
        self.x = x
        self.y = y
        self.task_activity = torch.Tensor(task_activity)



    def __len__(self):
        return self.x.shape[0]



    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.task_activity[idx]




def DataLoaders(x, y, task_activity, num_workers=0, **kwargs): 
    task_activity = torch.Tensor(task_activity)

    allowed_keys = list(set(['batch_size', 'shuffle']).intersection(kwargs.keys()))
    dataloader_dict = {key: kwargs[key] for key in allowed_keys}
    dataloader_dict['batch_size']  = util.dict_extract(dataloader_dict, 'batch_size', 64)
    dataloader_dict['shuffle']  = util.dict_extract(dataloader_dict, 'shuffle', True)

    bool_data_task_batching = util.dict_extract(kwargs, 'data_task_batching', True) # default value

    # structure the generator by shuffle and data_task_batching
    if bool_data_task_batching:
        # determine the ratios based on given task_activity and “total” batch size
        unique_activities = torch.unique(task_activity).int()
        _total_activities = [(task_activity == i).sum() for i in unique_activities]
        _ratios = {'task_{}'.format(i): float(_total_activities[i] / sum(_total_activities)) for i in range(len(_total_activities))}
        _max_ratio = np.argmax(_ratios.values())
        _max_ratio_key = list(_ratios.keys())[_max_ratio]

        # guarantee that batch size is sufficiently large to sample according to non-zero ratios
        _min_batch_size = sum([ratio > 0 for ratio in _ratios.values()])
        if dataloader_dict['batch_size'] < _min_batch_size:
            raise ValueError("Since 'data_task_batching' is True and the task_activity indicates that {} tasks are used we need a total 'batch_size' of at least {}.".format(_min_batch_size, _min_batch_size))
        
        _batch_sizes = {key: max(1, int(_ratios[key] * dataloader_dict['batch_size'])) for key in _ratios.keys()}
        _batch_sizes_val = list(_batch_sizes.values())
        _batch_sizes[_max_ratio_key] = dataloader_dict['batch_size'] - sum(_batch_sizes_val[:_max_ratio] + _batch_sizes_val[_max_ratio+1:])
        _ind_taskdatas = [(task_activity == i) for i in unique_activities]
        _dataset_partitions = {'task_{}'.format(i): CustomDataset(x[_ind_taskdatas[i]], y[_ind_taskdatas[i]], task_activity[_ind_taskdatas[i]]) for i in unique_activities}
        if not(isinstance(dataloader_dict['shuffle'], dict)):
            _shuffle = {'task_{}'.format(i): dataloader_dict['shuffle'] for i in unique_activities}
        data_loaders = {'task_{}'.format(i): DataLoader(_dataset_partitions['task_{}'.format(i)], batch_size=_batch_sizes['task_{}'.format(i)], shuffle=_shuffle['task_{}'.format(i)], num_workers=num_workers) for i in unique_activities}
    
    else:
        dataset = CustomDataset(x, y, task_activity)
        data_loaders =  {'task_0': DataLoader(dataset, num_workers=num_workers, **dataloader_dict)}

    return CombinedLoader(data_loaders)




class DataModule(pl.LightningDataModule):

    def __init__(self, data, **config_training):
        super().__init__()
        self.data = data
        self.config_training, self.all_tasks = init_config_training(**config_training)
        self.n_workers = 1 # os.cpu_count()



    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.data_train = self.data.create_data('train') 
            self.data_val = self.data.create_data('val') 

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.data_test = self.data.create_data('test') 



    def train_dataloader(self): # create training data based on 'data_task_batching'
        data_loaders = DataLoaders(*self.data_train.values(), num_workers=self.n_workers, **self.config_training)
        return data_loaders



    def val_dataloader(self): 
        loader_config = {
            'batch_size': self.config_training['batch_size'], 
            'data_task_batching': self.config_training['data_task_batching'],
            'shuffle': False
        }
        data_loaders = DataLoaders(*self.data_val.values(), num_workers=self.n_workers, **loader_config)
        return data_loaders



    def test_dataloader(self): 
        loader_config = {
            'batch_size': self.config_training['batch_size'], 
            'data_task_batching': self.config_training['data_task_batching'],
            'shuffle': False
        }
        data_loaders = DataLoaders(*self.data_test.values(), num_workers=self.n_workers, **loader_config)
        return data_loaders