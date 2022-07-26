from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader

import core_code.util.helpers as util




class CustomDataset(Dataset):
    """ Custom torch Dataset. """

    def __init__(
        self, 
        x: torch.tensor, 
        y: torch.tensor, 
        task_activity: torch.tensor
    ):
        """Initialize the torch Dataset given torch tensors for the x, y values and their respective task activity for each (x, y) - datapoint.

        Args:
            x (torch.tensor): x values with shape[0] = n.
            y (torch.tensor): y values with shape[0] = n.
            task_activity (torch.tensor): Associated task activty values with shape = (n).
        """
        
        self.x = x
        self.y = y
        self.task_activity = task_activity



    def __len__(self):
        """ Necessary __len__ method for custom torch Dataset. (See https://pytorch.org/tutorials/beginner/data_loading_tutorial.html) """

        return self.x.shape[0]



    def __getitem__(self, idx):
        """ Necessary __getitem__ method for custom torch Dataset. (See https://pytorch.org/tutorials/beginner/data_loading_tutorial.html) """

        return self.x[idx], self.y[idx], self.task_activity[idx]




def DataLoaders(
    x: torch.tensor, 
    y: torch.tensor, 
    task_activity: torch.tensor, 
    num_workers: int = 0, 
    callback: Callable[[torch.tensor, torch.tensor, torch.tensor, int, dict], list[DataLoader]] = None,
    **kwargs
) -> CombinedLoader: 
    """Create the dataloaders for the given task. The construction is dependent on the optiona arguments. 
    Note: Additional batching strategies can be added by 

    Args:
        x (torch.tensor): x values with shape[0] = n.
        y (torch.tensor): y values with shape[0] = n.
        task_activity (torch.tensor): Associated task activty values with shape = (n).
        num_workers (int, optional): Number of workers in the torch.DataLoaders objects. Defaults to 0.
        callback (Callable, optional): Callback to insert custom batching strategy (see source code). This is only relevant if 
        batching_strategy=='custom'.
    
    Optional arguments:
        batch_size (str): Total batch size.
        shuffle (bool): Determines whether the given data is shuffled (x, y, task_activity) is shuffled before we train.
        batching_strategy (str): Determines the type of batching strategy that we want to employ (which might be task specific).

    Raises:
        ValueError: Raised if the value of 'batching_strategy' is not known. See the source code for reference of viable 
        'batching_strategy' values.

    Returns:
        CombinedLoader: List of data loaders which is passed into the pytorch_lightning.trainer.supporters.CombinedLoader object to
        be used throughout the PyTorch Lightning training.
    """

    task_activity = torch.Tensor(task_activity)

    allowed_keys = list(set(['batch_size', 'shuffle']).intersection(kwargs.keys()))
    dataloader_dict = {key: kwargs[key] for key in allowed_keys}
    dataloader_dict['batch_size']  = util.dict_extract(dataloader_dict, 'batch_size', None)
    dataloader_dict['shuffle']  = util.dict_extract(dataloader_dict, 'shuffle', None)

    batching_strategy = util.dict_extract(kwargs, 'batching_strategy', None) # default value

    # structure the generator by shuffle and batching_strategy
    if batching_strategy == 'data_deterministic':

        data_loaders = _batching_data_deterministic(x, y, task_activity, num_workers, dataloader_dict)
        
    elif batching_strategy == 'data_stochastic':

        data_loaders = _batching_data_stochastic(x, y, task_activity, num_workers, dataloader_dict)

    elif batching_strategy == 'custom':

        data_loaders = callback(x, y, task_activity, num_workers, dataloader_dict)

    else: 
        raise ValueError("The batching strategy '{}' is not implemented. You might want to add it here, in the class DataLoaders.".format(batching_strategy))

    return CombinedLoader(data_loaders)


def _batching_data_deterministic(
    x: torch.tensor, 
    y: torch.tensor, 
    task_activity: torch.tensor, 
    num_workers: int,
    dataloader_dict: dict
) -> list[DataLoader]:
    """Implements the deterministic batching strategy where the task ratio of the data implies the ratio within one batch.

    Args:
        x (torch.tensor): x values with shape[0] = n.
        y (torch.tensor): y values with shape[0] = n.
        task_activity (torch.tensor): Associated task activty values with shape = (n).
        num_workers (int): Number of workers in the torch.DataLoaders objects.
        dataloader_dict (dict): Dataloader configurations. (Currently allowed keys: batch_size (str), shuffle (str),
        batching_strategy (str))

    Raises:
        ValueError: Checking the minimum batch sizes in relation to the number of tasks.

    Returns:
        list[DataLoader]: List of dataloaders.
    """
    # determine the ratios based on given task_activity and “total” batch size
    unique_activities = torch.unique(task_activity).int()
    _total_activities = [(task_activity == i).sum() for i in unique_activities]
    _ratios = {'task_{}'.format(i): float(_total_activities[i] / sum(_total_activities)) for i in range(len(_total_activities))}
    _max_ratio = np.argmax(_ratios.values())
    _max_ratio_key = list(_ratios.keys())[_max_ratio]

    # guarantee that batch size is sufficiently large to sample according to non-zero ratios
    _min_batch_size = sum([ratio > 0 for ratio in _ratios.values()])
    if dataloader_dict['batch_size'] < _min_batch_size:
        raise ValueError("Since 'batching_strategy' is batching_data_deterministic and the task_activity indicates that {} tasks are used we need a total 'batch_size' of at least {}. \
Alternatively, set 'batching_strategy' to batching_data_stochastic.".format(_min_batch_size, _min_batch_size))
    
    _batch_sizes = {key: max(1, int(_ratios[key] * dataloader_dict['batch_size'])) for key in _ratios.keys()}
    _batch_sizes_val = list(_batch_sizes.values())
    _batch_sizes[_max_ratio_key] = dataloader_dict['batch_size'] - sum(_batch_sizes_val[:_max_ratio] + _batch_sizes_val[_max_ratio+1:])
    _ind_taskdatas = [(task_activity == i) for i in unique_activities]
    _dataset_partitions = {'task_{}'.format(i): CustomDataset(x[_ind_taskdatas[i]], y[_ind_taskdatas[i]], task_activity[_ind_taskdatas[i]]) for i in unique_activities}
    if not(isinstance(dataloader_dict['shuffle'], dict)):
        _shuffle = {'task_{}'.format(i): dataloader_dict['shuffle'] for i in unique_activities}
    data_loaders = {'task_{}'.format(i): DataLoader(_dataset_partitions['task_{}'.format(i)], batch_size=_batch_sizes['task_{}'.format(i)], shuffle=_shuffle['task_{}'.format(i)], num_workers=num_workers) for i in unique_activities}
    
    return data_loaders



def _batching_data_stochastic(
    x: torch.tensor, 
    y: torch.tensor, 
    task_activity: torch.tensor, 
    num_workers: int,
    dataloader_dict: dict
) -> list[DataLoader]:
    """Implements the stochastic batching strategy where the task ratio of the data implies the ratio within one batch.

    Args:
        x (torch.tensor): x values with shape[0] = n.
        y (torch.tensor): y values with shape[0] = n.
        task_activity (torch.tensor): Associated task activty values with shape = (n).
        num_workers (int): Number of workers in the torch.DataLoaders objects.
        dataloader_dict (dict): Dataloader configurations. (Currently allowed keys: batch_size (str), shuffle (str),
        batching_strategy (str))

    Returns:
        list[DataLoader]: List of dataloaders.
    """

    dataset = CustomDataset(x, y, task_activity)
    data_loaders =  {'task_0': DataLoader(dataset, num_workers=num_workers, **dataloader_dict)}

    return data_loaders