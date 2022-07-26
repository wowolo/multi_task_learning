import importlib
import torch


def CreateModel(**config_architecture) -> torch.nn.Module:
    """Function that retrieves and intializes the neural network model class based on the 
    config_architecture configuration. Note that NNModel is retrieved from the
    core_code/model_lib/'config_architecture['architecture_key']'.py file.

    Returns:
        torch.nn.Module: Instance of the NNModel class from 
        core_code/model_lib/'config_architecture['architecture_key']'.py.
    """

    # import architecture based on architecture_key in config_architecture
    models_path = 'core_code.model_lib'
    model_lib = importlib.import_module(models_path + '.' + config_architecture['architecture_key'])
    model = model_lib.NNModel(**config_architecture)

    return model