import importlib


def CreateModel(**config_architecture):

    # import architecture based on config_architecture
    # note: default architecture configurations are set in the respective model files (/core_code/nn_models/)
    models_path = 'core_code.model_lib'
    models = importlib.import_module(models_path + '.' + config_architecture['architecture_key'])
    model = models.NNModel(**config_architecture)

    return model