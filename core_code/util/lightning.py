import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader

import core_code.util.helpers as util




class CustomDataset(Dataset):

    def __init__(self, x, y, task_activity):
        
        self.x = x
        self.y = y
        self.task_activity = task_activity



    def __len__(self):
        return self.x.shape[0]



    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.task_activity[idx]




def DataLoaders(x, y, task_activity, num_workers=0, **kwargs): 
    task_activity = torch.Tensor(task_activity)

    allowed_keys = list(set(['batch_size', 'shuffle']).intersection(kwargs.keys()))
    dataloader_dict = {key: kwargs[key] for key in allowed_keys}
    dataloader_dict['batch_size']  = util.dict_extract(dataloader_dict, 'batch_size', None)
    dataloader_dict['shuffle']  = util.dict_extract(dataloader_dict, 'shuffle', None)

    batching_strategy = util.dict_extract(kwargs, 'batching_strategy', None) # default value

    # structure the generator by shuffle and batching_strategy
    if batching_strategy == 'deterministic':
        # determine the ratios based on given task_activity and “total” batch size
        unique_activities = torch.unique(task_activity).int()
        _total_activities = [(task_activity == i).sum() for i in unique_activities]
        _ratios = {'task_{}'.format(i): float(_total_activities[i] / sum(_total_activities)) for i in range(len(_total_activities))}
        _max_ratio = np.argmax(_ratios.values())
        _max_ratio_key = list(_ratios.keys())[_max_ratio]

        # guarantee that batch size is sufficiently large to sample according to non-zero ratios
        _min_batch_size = sum([ratio > 0 for ratio in _ratios.values()])
        if dataloader_dict['batch_size'] < _min_batch_size:
            raise ValueError("Since 'batching_strategy' is True and the task_activity indicates that {} tasks are used we need a total 'batch_size' of at least {}.".format(_min_batch_size, _min_batch_size))
        
        _batch_sizes = {key: max(1, int(_ratios[key] * dataloader_dict['batch_size'])) for key in _ratios.keys()}
        _batch_sizes_val = list(_batch_sizes.values())
        _batch_sizes[_max_ratio_key] = dataloader_dict['batch_size'] - sum(_batch_sizes_val[:_max_ratio] + _batch_sizes_val[_max_ratio+1:])
        _ind_taskdatas = [(task_activity == i) for i in unique_activities]
        _dataset_partitions = {'task_{}'.format(i): CustomDataset(x[_ind_taskdatas[i]], y[_ind_taskdatas[i]], task_activity[_ind_taskdatas[i]]) for i in unique_activities}
        if not(isinstance(dataloader_dict['shuffle'], dict)):
            _shuffle = {'task_{}'.format(i): dataloader_dict['shuffle'] for i in unique_activities}
        data_loaders = {'task_{}'.format(i): DataLoader(_dataset_partitions['task_{}'.format(i)], batch_size=_batch_sizes['task_{}'.format(i)], shuffle=_shuffle['task_{}'.format(i)], num_workers=num_workers) for i in unique_activities}
    
    elif batching_strategy == 'data_stochastic':
        dataset = CustomDataset(x, y, task_activity)
        data_loaders =  {'task_0': DataLoader(dataset, num_workers=num_workers, **dataloader_dict)}
    
    else: 
        raise ValueError("The batching strategy '{}' is not implemented. You might want to add it here, in the class DataLoaders.".format(batching_strategy))

    return CombinedLoader(data_loaders)