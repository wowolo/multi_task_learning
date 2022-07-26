import numpy as np

import core_code.util.helpers as util
from core_code.util.default_config import init_config_data
from core_code.util.config_extractions import _f_true_fm



class CreateData():


    @staticmethod
    def _equi_data(n_samples, x_min_i, x_max_i):
        return np.linspace(x_min_i, x_max_i , n_samples)



    @staticmethod
    def _uniform_data(n_samples, x_min_i, x_max_i):
        return np.random.rand(n_samples) * (x_max_i - x_min_i) + x_min_i



    @staticmethod
    def _noise_data(n_samples, x_min_i, x_max_i):
        return np.random.normal(size=n_samples) % (x_max_i - x_min_i) + (x_max_i + x_min_i) / 2



    @staticmethod
    def _periodic_data(n_samples, x_min_i, x_max_i):
        samples_1 = int(n_samples/2)
        per_1 = np.sin(1.5 * np.pi * np.linspace(-1, 1, int(n_samples/2))) * (x_max_i - x_min_i) + x_min_i
        rest_samples = n_samples - samples_1
        per_2 = - np.sin(1.5 * np.pi * np.linspace(-1, 1, rest_samples)) * (x_max_i - x_min_i) + x_min_i
        data = np.concatenate((per_1, per_2))
        return data


    def __init__(self, **config_data):

        self.config_data, self.all_tasks = init_config_data(**config_data)
        if isinstance(self.all_tasks, type(None)):
            self.all_tasks = {0}


    def create_data(self, type):

        x = np.empty((0,self.config_data['d_in']))
        y = np.empty((0,self.config_data['d_out']))
        task_activity = np.empty(0, dtype=int)

        for task_num in self.all_tasks:
            
            task_config = util.extract_taskconfig(self.config_data, task_num)
            task_config = self._clean_1dbounds(task_config)
            _x, _y, _task_activity = self.task_data_creator(type, task_config, task_num)
            x = np.concatenate([x, _x], axis=0)
            y = np.concatenate([y, _y], axis=0)
            task_activity = np.concatenate([task_activity, _task_activity], axis=0, dtype=int)


        data_dict = {
            'x': util.to_tensor(x), 
            'y': util.to_tensor(y), 
            'task_activity': util.to_tensor(task_activity)
        }
        
        return data_dict



    def task_data_creator(self, type, task_config, task_num):

        
        d_in = task_config['d_in']

        n_samples = task_config['n_{}'.format(type)]
        x_min = task_config['x_min_{}'.format(type)]
        x_max = task_config['x_max_{}'.format(type)]
        data_generators = task_config['data_generators_{}'.format(type)]

        if type == 'train':    
            noise_scale = self.config_data['noise_scale']
        else:
            noise_scale = 0

            
        task_x = np.empty((n_samples, d_in))

        if not(isinstance(data_generators, list)):
            data_generators = [data_generators]
        data_generators = data_generators[:d_in]

        while len(data_generators) < d_in: # default: fill missing data generators with last entry
            data_generators.append(data_generators[-1])
        

        temp_func_dict = {
            'equi': self._equi_data,
            'uniform': self._uniform_data,
            'periodic': self._periodic_data,
            'noise': self._noise_data
        }

        for d in range(d_in): # create data according to data generator in each dimension

            task_x[:, d] = temp_func_dict[data_generators[d]](n_samples, x_min[d], x_max[d])
    
        f_true = _f_true_fm(task_config['f_true'], task_config['f_true_callback'])

        task_y = f_true(task_x) + np.random.normal(scale=1, size=(n_samples, task_config['d_out'])) * noise_scale

        _task_activity = np.ones(task_x.shape[0], dtype=int) * task_num

        return task_x, task_y, _task_activity
    


    @staticmethod
    def _clean_1dbounds(config_data):
        for bound_key in ['x_min_train', 'x_max_train', 'x_min_val', 'x_max_val', 'x_min_test', 'x_max_test']:
            if isinstance(config_data[bound_key], int):
                config_data[bound_key] = [config_data[bound_key] for i in range(config_data['d_in'])]
        return config_data