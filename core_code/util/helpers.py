# import json
from typing import Union, Sequence
import torch



def dict_extract(
    kwargs: dict, 
    key: str, 
    default=None
    ):
    """Extract a key specific value of a dictionary.

    Args:
        kwargs (dict): Dictionary from which we want to extract the value.
        key (str): Key of the wanted value.
        default (_type_, optional): Default value in case that the key is not contained in the given dictionary. 
        Defaults to None.

    Returns:
        _type_: _description_
    """
    
    if key in kwargs.keys():
        key_val = kwargs[key]
    else:
        key_val = default

    return key_val



def to_tensor(x: Sequence) -> torch.tensor:
    """Transform a suitable input to a torch tensor.

    Args:
        x (Sequence): Input argument (suitable for tensor transformation by torch.tensor).

    Returns:
        torch.tensor: Output tensor.
    """

    return torch.tensor(x)



# def make_jsonable(x):
#     try:
#         json.dumps(x)
#         return x
#     except:
#         return str(x)



def create_config(
    kwargs: dict, 
    default_extraction_strings: dict
) -> dict:
    """Create a configuration dictionary based on optional inputs in kwargs and a default dictionary supplying the allowed keys for the 
    resulting configuration and its default values.

    Args:
        kwargs (dict): Optional input arguments given as dictionary.
        default_extraction_strings (dict): Dictionary with allowed keys and its default values.

    Returns:
        dict: Generated dictionary.
    """

    config = {string: None for string in default_extraction_strings}

    for string in default_extraction_strings:
        
        if string in kwargs.keys():
            item = kwargs[string]
        else:
            item = default_extraction_strings[string]
        
        config[string] = item
    
    return config



def check_config(**kwargs) -> Union[set, None]:
    """Check the input configurations regarding the allowd task specific evaluations and return the set of tasks.

    Raises:
        ValueError: Raised in the case of a detected format layout of the task dictionaries supplied by the user as values of the
        input dictionary.
        ValueError: The items of the input dictionary's task specific values do not specifically designate the same tasks.

    Returns:
        set: The configuration induced set of tasks. None is returned in the second argument no task specification has been detected.
    """

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



def extract_taskconfig(
    config: dict, 
    task_num: int
) -> dict:
    """Given a configuration generate the induced configuration for the specified task.

    Args:
        config (dict): Given configuration.
        task_num (int): Integer specifying the task.

    Returns:
        dict: Task specific configuration.
    """

    task_config = dict.fromkeys(config.keys())
    task_key = 'task_' + str(int(task_num))

    for key in task_config.keys():
        value = config[key]
        if isinstance(value, dict):
            value = value[task_key]
        task_config[key] = value

    return task_config



# def dict_to_file(dict, file_path, format='v'):
#     # format: 'v' or 'h'
#     with open(file_path, 'w') as file:
#         if format == 'v':
#             for key, val in dict.items():
#                 file.write('{}: {}\n'.format(key, val))
#         else:
#             json_dict = {key: make_jsonable(dict[key]) for key in dict.keys()}
#             file.write(json.dumps(json_dict))



def dictvals_to_list(dict: dict) -> dict:
    """Put all dictionary values, which are not already a list, into a list.

    Args:
        dict (dict): Input dictionary whose values should be guaranteed lists.

    Returns:
        dict: Resulting dictionary.
    """
    for key, val in dict.items():
        if not(isinstance(val, list)):
            dict[key] = [val] 

    return dict