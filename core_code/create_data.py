import numpy as np

import core_code.util.helpers as util
from core_code.util.default_config import init_config_data
from core_code.util.config_extractions import _f_true_fm



class CreateData():


    @staticmethod
    def _equi_data(
        n_samples: int, 
        x_min_i: float, 
        x_max_i: float
    ) -> np.array:
        """Helper function to generate equi distant data points between x_min_i and x_max_i.

        Args:
            n_samples (int): Number of data points.
            x_min_i (float): Lower bound of data interval.
            x_max_i (float): Upper bound of data interval.

        Returns:
            np.array: Generated data array.
        """

        return np.linspace(x_min_i, x_max_i , n_samples)



    @staticmethod
    def _uniform_data(
        n_samples: int, 
        x_min_i: float, 
        x_max_i: float
    ) -> np.array:
        """Helper function to generate equi distant data points between x_min_i and x_max_i.

        Args:
            n_samples (int): Number of data points.
            x_min_i (float): Lower bound of data interval.
            x_max_i (float): Upper bound of data interval.

        Returns:
            np.array: Generated data array.
        """

        return np.random.rand(n_samples) * (x_max_i - x_min_i) + x_min_i



    @staticmethod
    def _noise_data(
        n_samples: int, 
        x_min_i: float, 
        x_max_i: float
    ) -> np.array:
        """Helper function to generate noisy data points between x_min_i and x_max_i.

        Args:
            n_samples (int): Number of data points.
            x_min_i (float): Lower bound of data interval.
            x_max_i (float): Upper bound of data interval.

        Returns:
            np.array: Generated data array.
        """

        return np.random.normal(size=n_samples) % (x_max_i - x_min_i) + (x_max_i + x_min_i) / 2



    @staticmethod
    def _periodic_data(
        n_samples: int, 
        x_min_i: float, 
        x_max_i: float
    ) -> np.array:
        """Helper function to generate periodic data points between x_min_i and x_max_i.

        Args:
            n_samples (int): Number of data points.
            x_min_i (float): Lower bound of data interval.
            x_max_i (float): Upper bound of data interval.

        Returns:
            np.array: Generated data array.
        """

        samples_1 = int(n_samples/2)
        per_1 = np.sin(1.5 * np.pi * np.linspace(-1, 1, int(n_samples/2))) * (x_max_i - x_min_i) + x_min_i
        rest_samples = n_samples - samples_1
        per_2 = - np.sin(1.5 * np.pi * np.linspace(-1, 1, rest_samples)) * (x_max_i - x_min_i) + x_min_i
        data = np.concatenate((per_1, per_2))
        return data



    def __init__(self, **config_data):
        """Initialize the data object which generates data based on the optional configuration inputs and
        the implied configurations via the function init_config_data from core_code/util/default_config.py.
        """

        self.config_data, self.all_tasks = init_config_data(**config_data)
        if isinstance(self.all_tasks, type(None)):
            self.all_tasks = {0}


    def create_data(self, type: str) -> dict:
        """Data generating method based on the configuration in the config_data attribute creating the data
        set based on the tasks deduced from the data configuration.

        Args:
            type (str): String determining whether we want train, validation or test data.

        Returns:
            dict: Resulting data dictionary with keys 'x', 'y' and 'task_activity'.
        """

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



    def task_data_creator(
        self, 
        type: str, 
        task_config: dict, 
        task_num: int
    ) -> list[np.array, np.array, np.array]:
        """Data creator for each task.

        Args:
            type (str): String determining whether we want train, validation or test data
            task_config (dict): Task specific data configuration.
            task_num (int): Integer assoicated to the task.

        Returns:
            list[np.array, np.array, np.array]: List of resulting array (x, y, task_activity).
        """
        
        d_in = task_config['d_in']

        n_samples = task_config['n_{}'.format(type)]
        x_min = task_config['x_min_{}'.format(type)]
        x_max = task_config['x_max_{}'.format(type)]
        data_generators = task_config['data_generators_{}'.format(type)]

        if type == 'train':    
            noise_scale = self.config_data['noise_scale']
        else:
            noise_scale = 0

        task_key = 'task_{}'.format(task_num)
            
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
    


    def _clean_1dbounds(self, task_config: dict) -> dict:
        """Helper functions to expand any interval related functions which have only an integer values
        to a list of these integer values with lenght corresponding to the input dimension.

        Args:
            task_config (dict): Task configuration.

        Returns:
            dict: 'Cleaned' task configuration
        """

        for bound_key in ['x_min_train', 'x_max_train', 'x_min_val', 'x_max_val', 'x_min_test', 'x_max_test']:

            value = self._clean_int_or_list(
                task_config[bound_key], 
                task_config['d_in']
            )
            
            task_config[bound_key] = value

        return task_config
    


    @staticmethod
    def _clean_int_or_list(value, d_in):        

        if isinstance(value, int):
            value = [value]
        if isinstance(value, list):
            while len(value) < d_in:
                value.append(value[-1])
        
        return value