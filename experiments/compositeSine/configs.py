import numpy as np

# task are enumerated starting with 0

config_function = {
    'd_in': 1, # one d_in value across all tasks, due to NN architecture
    'd_out': 7, # one d_out value across all tasks, due to NN architecture
    'f_true': 'compositeSine',
}

# configs for model architecture
configs_architecture = {
    'architecture_key': ['abcMLP'],
    'depth': 6,
    'width': 2048, 
    'bottleneck_width': 2048, # for Stack
    'variable_width': 2048, # for Stack
    'linear_skip_conn': False, # for Stack
    'linear_skip_conn_width': 64, # for Stack
    'skip_conn': False, # for Stack
    'hidden_bottleneck_activation': ['ReLU'], 
    'hidden_layer_activation': 'ReLU', 
    # for abcMLP
    # 'list_a': [[-.5] + [0 for i in range(4)] + [.5]], # default: mup 
    # 'list_b': [[.5 for i in range(6)]], # default: mup 
    'list_a': [[0] + [.5 for i in range(5)]], # NTK 
    'list_b': [[0 for i in range(6)]], # NTK 
    # 'list_a': [[0] + [0.5 for i in range(5)]], 
    # 'list_b': [[0 for i in range(6)]], 
    'c': 0, 
}
configs_architecture.update(config_function)

# configs for data creation
configs_data = {
    #### (potentially) task specific ####
    'n_train': [{'task_0': 256, 'task_1': 512}],
    'noise_scale': .2,
    'x_min_train': -2,
    'x_max_train': {'task_0': 0, 'task_1': 2},
    'n_val': 2048,
    'x_min_val': -2,
    'x_max_val': 2,
    'n_test': {'task_0': 0, 'task_1': 0},
    'x_min_test': -2,
    'x_max_test': 2,
}
configs_data.update(config_function)

# configs for model training
configs_training = {
    'batch_size': 64, 
    'batching_strategy': 'data_deterministic',
    'num_workers': 1,
    #### (potentially) task specific ####
    'criterion': {'task_0': ('dimred_MSELoss', [0]), 'task_1': ('dimred_MSELoss', list(np.arange(1, 7)))}, 
    'update_rule': 'SGD',
    'learning_rate': [0.005],
    'regularization_alpha': [0],
    'regularization_ord': 2,
    'shuffle': True,
}


# configs customised for the specific experiment setup
configs_custom = {
    'seed': 77,
    'workers': True,
    'logging_epoch_interval': 50 # >= 2
}