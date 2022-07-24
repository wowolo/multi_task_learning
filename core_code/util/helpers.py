import json
from multiprocessing.sharedctypes import Value
import torch



def dict_extract(kwargs, key, default=None):
    
    if key in kwargs.keys():
        key_val = kwargs[key]
    else:
        key_val = default

    return key_val



def to_tensor(x):
    return torch.Tensor(x)



def make_jsonable(x):
    try:
        json.dumps(x)
        return x
    except:
        return str(x)



def create_config(kwargs, default_extraction_strings):

    config = {string: None for string in default_extraction_strings}

    for string in default_extraction_strings:
        
        if string in kwargs.keys():
            item = kwargs[string]
        else:
            item = default_extraction_strings[string]
        
        config[string] = item
    
    return config



def check_config(**kwargs):
    all_tasks = set()
    # check values either non dict or dict with task_#num format
    # + check that values
    for val in kwargs.values():
        temp_set = set()
        if isinstance(val, dict):
            for key in val.keys():
                sep_key = key.split('_')
                if (len(sep_key) != 2) or (sep_key[0] != 'task') or (sep_key[1] != str(int(sep_key[1]))):
                    raise ValueError('The format of the configuration is invalid.')
                num_task = int(sep_key[1])
                temp_set.update({num_task})
        if len(all_tasks) == 0:
            all_tasks = temp_set
        elif len(temp_set) > 0:
            if all_tasks != temp_set:
                raise ValueError('The task specific configurations are invalid. All values that have task-specific inputs using a dictionary have to assing values for all task keys via their respective dictionaries.')

    if len(all_tasks) == 0:
        all_tasks = None

    return all_tasks



def extract_taskconfig(config, task_num):
        task_config = dict.fromkeys(config.keys())
        task_key = 'task_' + str(int(task_num))

        for key in task_config.keys():
            value = config[key]
            if isinstance(value, dict):
                value = value[task_key]
            task_config[key] = value

        return task_config



def dict_to_file(dict, file_path, format='v'):
    # format: 'v' or 'h'
    with open(file_path, 'w') as file:
        if format == 'v':
            for key, val in dict.items():
                file.write('{}: {}\n'.format(key, val))
        else:
            json_dict = {key: make_jsonable(dict[key]) for key in dict.keys()}
            file.write(json.dumps(json_dict))



def dictvals_to_list(dict):
    for key, val in dict.items():
        if not(isinstance(val, list)):
            dict[key] = [val] 

    return dict




class dimred_MSELoss(torch.nn.Module):

    def __init__(self, dimension_activity):

        super(dimred_MSELoss,self).__init__()
        self.dimension_activity = dimension_activity
    

    
    def forward(self, output, target):

        if output.shape[0] == 0: return 0

        dimred_output = output[:, self.dimension_activity]
        dimred_target = target[:, self.dimension_activity]

        return torch.sum((dimred_output - dimred_target)**2)