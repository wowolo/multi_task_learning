import os
import time
import pytorch_lightning as pl

import wandb
from pytorch_lightning.loggers import WandbLogger

from experiments.compositeSine.logging_callback import LoggingCallback 

from core_code.create_data import CreateData
from core_code.create_model import CreateModel
from core_code.lightning_multitask import DataModule, LightningMultitask

from experiments.util import BasicManager
from core_code.util.default_config import _make_init_config

import torch



class Manager(BasicManager):

    def __init__(self, configs_data, configs_architecture, configs_traininig, config_trainer, configs_custom):

        # self.set_randomness(configs_custom['torch_seed'], configs_custom['numpy_seed'])

        # save configs
        self.configs_data = configs_data 
        self.configs_architecture = configs_architecture
        self.configs_traininig = configs_traininig
        self.configs_trainer = config_trainer
        self.configs_custom = self.init_configs_custom(**configs_custom) # guarantee that it has the key determined by 'default_extraction_strings'

        self.configs_data_list, \
        self.configs_architecture_list, \
        self.configs_training_list, \
        self.configs_trainer_list, \
        self.configs_custom_list \
            = self._create_grid()    



    @staticmethod
    def init_configs_custom(**kwargs):

        default_extraction_strings = {
            'seed': 77,
            'workers': True,
            'logging_epoch_interval': 50
        }

        return _make_init_config(default_extraction_strings, **kwargs)[0]



    def run(
        self, 
        experimentbatch_name=None, 
        ind_configs=None, 
    ):

        self.configs_data_list, \
        self.configs_architecture_list, \
        self.configs_training_list, \
        self.configs_trainer_list, \
        self.configs_custom_list \
            = self._create_grid()

        if isinstance(ind_configs, type(None)):
            ind_configs = [i for i in range(self.num_experiments)]
        elif isinstance(ind_configs, int): # or isinstance(ind_configs, float):
            ind_configs = [int(ind_configs)]
        
        for i in ind_configs:

            print('num_config: {}'.format(i))
            print('len_config: {}'.format(len(self.configs_data_list)))
            print([torch.cuda.device(i) for i in range(torch.cuda.device_count())])
            
            # manage the grid configurations
            config_data = self.configs_data_list[i]
            config_architecture = self.configs_architecture_list[i]
            config_training = self.configs_training_list[i]
            config_trainer = self.configs_trainer_list[i]
            config_custom = self.configs_custom_list[i]


            # check the configs to avoid error later on
            _config_architecture = config_architecture.copy()
            del _config_architecture['d_in']
            del _config_architecture['d_out']
            del _config_architecture['f_true']
            from core_code.util.helpers import check_config
            all_configs = dict( 
                **config_data, 
                **_config_architecture,
                **config_training, 
                **config_trainer,
                **config_custom
            )
            check_config(**all_configs)


            # initialize the core objects
            pl.seed_everything(config_custom['seed'], config_custom['workers'])
            data = CreateData(**config_data)
            torch_model = CreateModel(**config_architecture)
            data_module = DataModule(data, **config_training)

            model = LightningMultitask(torch_model, **config_training)

            logging_callback = LoggingCallback(
                *data.create_data('train').values(), 
                data.config_data, 
                logging_epoch_interval=config_custom['logging_epoch_interval']
            )

            config_experiment = {}

            project_name = 'logging_' + os.path.split(os.path.dirname(os.path.realpath(__file__)))[-1]

            config_experiment.update(data.config_data)
            config_experiment.update(torch_model.config_architecture)
            config_experiment.update(model.config_training) # alternatively: data_module.config_training
            config_experiment.update(config_custom)
            config_experiment.update(config_trainer)

            if model.config_training['regularization_alpha'] > 0:
                reg_tag = 'weight decay'
            else:
                reg_tag = 'no weight decay'

            tags = [
                torch_model.config_architecture['architecture_key'], 
                model.config_training['update_rule'],
                reg_tag
            ]
            
            wandb.login()
            wandb.require(experiment="service")
            
            logger = WandbLogger(
                name = experimentbatch_name + f'_config{i}',
                project = project_name,
                log_model = True,
                group = torch_model.config_architecture['architecture_key'],
                config = config_experiment,
                tags = tags,
                # settings = wandb.Settings(start_method='fork')
            )

            trainer = model.fit(
                data_module,
                logger=logger,
                callbacks=[
                    logging_callback,
                ],
                **config_trainer
            )

            # trainer.logger.experiment.config.update(data.config_data)
            # trainer.logger.experiment.config.update(torch_model.config_architecture)
            # trainer.logger.experiment.config.update(model.config_training) # alternatively: data_module.config_training
            # trainer.logger.experiment.config.update(config_custom)
            # trainer.logger.experiment.config.update(config_trainer)

            wandb.finish()