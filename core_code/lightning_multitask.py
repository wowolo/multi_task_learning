import torch
import pytorch_lightning as pl


import core_code.util.helpers as util
from core_code.util.default_config import init_config_training, init_config_trainer
from core_code.util.config_extractions import _criterion_fm, _update_rule_fm

from core_code.util.lightning import DataLoaders





class DataModule(pl.LightningDataModule):

    def __init__(self, data, **config_training):
        super().__init__()
        self._data = data
        self.all_tasks = data.all_tasks
        self.config_training = init_config_training(**config_training)
        self.n_workers = 1 # os.cpu_count()



    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.data_train = self._data.create_data('train') 
            self.data_val = self._data.create_data('val') 

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.data_test = self._data.create_data('test') 



    def train_dataloader(self): # create training data based on 'batching_strategy'
        data_loaders = DataLoaders(*self.data_train.values(), num_workers=self.n_workers, **self.config_training)
        return data_loaders



    def val_dataloader(self): 
        loader_config = {
            'batch_size': self.config_training['batch_size'], 
            'batching_strategy': self.config_training['batching_strategy'],
            'shuffle': False
        }
        data_loaders = DataLoaders(*self.data_val.values(), num_workers=self.n_workers, **loader_config)
        return data_loaders



    def test_dataloader(self): 
        loader_config = {
            'batch_size': self.config_training['batch_size'], 
            'batching_strategy': self.config_training['batching_strategy'],
            'shuffle': False
        }
        data_loaders = DataLoaders(*self.data_test.values(), num_workers=self.n_workers, **loader_config)
        return data_loaders





class LightningMultitask(pl.LightningModule):


    def __init__(self, models, **config_training):
        super(LightningMultitask, self).__init__()
        
        self.config_training = init_config_training(**config_training)
        self._config = dict(
            model = models,
            **self.config_training
        )
        self.all_config_tasks = util.check_config(**self._config)

        # provide pl with the parameters by defining the models as attributes of the class
        if not isinstance(models, dict):
            self._model = models
        else:
            for task in models.keys():
                setattr(self, '_model' + task, models[task])

        self.automatic_optimization = False
       


    def configure_optimizers(self):        
        # single optimizer if update rule and learning rate are task unspecific
        if (not isinstance(self._config['update_rule'], dict)) and (not isinstance(self._config['learning_rate'], dict)):
            update = _update_rule_fm(self._config['update_rule'], self._config['update_rule_callback'])
            optimizer = update(self._config['model'].parameters(), lr=self._config['learning_rate'])
            self._optimizer_keys = []

            return optimizer

        else:
            optimizers = {}
            
            for task_num in self.all_config_tasks:

                task_config = util.extract_taskconfig(self._config, task_num)
                update = _update_rule_fm(task_config['update_rule'], task_config['update_rule_callback'])
                optimizer = update(task_config['model'].parameters(), lr=task_config['learning_rate'])
                optimizers['task_{}'.format(task_num)] = optimizer
            
            self._optimizer_keys = list(optimizers.keys()) 

            return list(optimizers.values())



    def training_step(self, batch, batch_idx):
        outputs, all_tasks = self._compute_tasklosses(batch)
        losses = outputs['losses']

        val_optimizers = self.optimizers()
        if len(self._optimizer_keys) == 0:
            optimizer = val_optimizers
            losses = sum(losses.values())
            # optimization step
            optimizer.zero_grad()
            self.manual_backward(losses)
            optimizer.step()
        else:    
            optimizers = {self._optimizer_keys[i]: val for i, val in enumerate(val_optimizers)}

            for task_num in all_tasks:
                task_key = 'task_{}'.format(task_num)
                optimizer = optimizers[task_key]
            
                # optimization step
                optimizer.zero_grad()
                self.manual_backward(losses[task_key])
                optimizer.step()

        return outputs
    


    def validation_step(self, batch, batch_idx): # criterion without regularization on validation set
        outputs = self._compute_tasklosses(batch, bool_training=False)

        return outputs



    def test_step(self, batch, batch_idx): # criterion without regularization on test set
        outputs = self._compute_tasklosses(batch, bool_training=False)

        return outputs

    

    def fit(
        self, 
        data_module: DataModule,
        logger = None,
        callbacks: list = [],
        **config_trainer: dict
    ): 
        config_trainer = init_config_trainer(**config_trainer)

        # check configs
        if isinstance(self.all_config_tasks, type(None)):
            pass
        elif data_module.all_tasks != self.all_config_tasks:
            raise ValueError('The DataModule implies that we have the tasks {} while the LightningMultitask model implies that we have the tasks {}.'.format(data_module.all_tasks, self.all_config_tasks))

        trainer = pl.Trainer(
            logger=logger,
            callbacks=callbacks,
            **config_trainer
        )
        
        trainer.fit(self, data_module)

        return trainer

    

    def _compute_tasklosses(self, batch, bool_training=True): # convenience function used in the step methods
        
        x, y, task_activity = self._retrieve_batch_data(batch)
        all_tasks = torch.unique(task_activity).int()

        preds = torch.empty_like(y, device=self.device)
        losses = {}
        for task_num in all_tasks:
            task_config = util.extract_taskconfig(self._config, task_num)
            _ind = (task_activity == task_num)
            _x = x[_ind]
            preds[_ind] = task_config['model'].forward(_x)

            if bool_training:
                criterion = _criterion_fm(task_config['criterion'], task_config['criterion_callback'])
                _loss = criterion(preds[_ind], y[_ind])

                # compute weight decay
                reg_alpha = task_config['regularization_alpha']
                if reg_alpha != 0:
                    reg = torch.tensor(0., device=self.device)
                    for param in task_config['model'].parameters():
                        reg = reg + torch.linalg.vector_norm(param.flatten(), ord=task_config['regularization_ord'])**2
                
                    _loss = _loss + reg_alpha * reg

                losses['task_{}'.format(task_num)] = _loss

        if not bool_training:
            return {'preds': preds}
        else:
            return {'preds': preds, 'losses': losses}, all_tasks


    
    def _retrieve_batch_data(self, batch):
        x = []
        y = []
        task_activity = []

        batch_keys = list(batch.keys())

        for task_key in batch_keys:

            _x, _y, _task_activity = batch[task_key]

            # concatenate the data to use in outputs
            x.append(_x)
            y.append(_y)
            task_activity.append(_task_activity)

        x = torch.concat(x)
        y = torch.concat(y)
        task_activity = torch.concat(task_activity)
        
        return x, y, task_activity