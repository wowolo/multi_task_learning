from experiments.compositeSine.manager import Manager
from experiments.compositeSine.configs import configs_data, configs_architecture, configs_training, configs_custom


manager = Manager(
    configs_data, 
    configs_architecture, 
    configs_training, 
    {}, 
    configs_custom
)

num_configs = len(manager.grid_config_lists(
    configs_data, 
    configs_architecture, 
    configs_training,
    {},
    configs_custom
)[0])

exit(num_configs)