from typing import Union, Callable
import torch
import pytorch_lightning as pl

from core_code.create_data import CreateData
import core_code.util.helpers as util
from core_code.util.default_config import init_config_training, init_config_trainer
from core_code.util.config_extractions import _criterion_fm, _update_rule_fm

from core_code.util.lightning import DataLoaders
from pytorch_lightning.trainer.supporters import CombinedLoader






class DataModule(pl.LightningDataModule):
    """ Data module wrapping the data into an appropriate class for training with PyTorch Lightning. """

    def __init__(
        self, 
        data: CreateData, 
        **config_training
    ):
        """ Initiallization of the data module. See https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html 
        for reference.

        Args:
            data (CreateData): Data object which needs to supply train, validation (and test) data.
        """

        super().__init__()
        self._data = data
        self.all_tasks = data.all_tasks
        self.config_training = init_config_training(**config_training)



    def setup(self, stage: str = None):
        """Setup the necessary data objects.

        Args:
            stage (str, optional): String to determine what kind of data we need to setup. Defaults 
            to None.
        """

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.data_train = self._data.create_data('train') 
            self.data_val = self._data.create_data('val') 

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.data_test = self._data.create_data('test') 



    def train_dataloader(self) -> CombinedLoader: # create training data based on 'batching_strategy'
        """Initialize the PyTorch train dataloaders based on the config_training configurations and using
        the DataLoaders function in core_code/util/lightning.py.

        Returns:
            CombinedLoader: Resulting data loaders.
        """

        x = self.data_train['x']
        y = self.data_train['y']
        task_activity = self.data_train['task_activity']

        data_loaders = DataLoaders(x, y, task_activity, **self.config_training)
        
        return data_loaders



    def val_dataloader(self) -> CombinedLoader: # create training data based on 'batching_strategy'
        """Initialize the PyTorch validation dataloaders based on the config_training configurations and using
        the DataLoaders function in core_code/util/lightning.py.

        Returns:
            CombinedLoader: Resulting data loaders.
        """

        x = self.data_val['x']
        y = self.data_val['y']
        task_activity = self.data_val['task_activity']
        loader_config = self.config_training
        loader_config['shuffle'] = False

        data_loaders = DataLoaders(x, y, task_activity, **loader_config)
        
        return data_loaders



    def test_dataloader(self) -> CombinedLoader: # create training data based on 'batching_strategy'
        """Initialize the PyTorch test dataloaders based on the config_training configurations and using
        the DataLoaders function in core_code/util/lightning.py.

        Returns:
            CombinedLoader: Resulting data loaders.
        """

        x = self.data_test['x']
        y = self.data_test['y']
        task_activity = self.data_test['task_activity']
        loader_config = self.config_training
        loader_config['shuffle'] = False

        data_loaders = DataLoaders(x, y, task_activity, **loader_config)
        
        return data_loaders





class LightningMultitask(pl.LightningModule):
    """ Lightning Module incorporating the model in order to apply native PyTorch Lightning training routines
    in combination with a model agnostic data module. """

    def __init__(
        self, 
        models: Union[torch.nn.Module, dict[torch.nn.Module]], 
        **config_training
    ):
        """Initialize the multi task class with (task-specific) model(s).

        Args:
            models (Union[torch.nn.Module, dict[torch.nn.Module]]): (Task-specific) PyTorch model(s), i.e., either one PyTorch models or 
            various PyTorch models in a dictionary as values with their associated tasks as keys (in the format 'task_{i}', i-th task).
        """

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
       


    def configure_optimizers(self) -> list[Callable]: 
        """Configure the (task-specific) optimizers.

        Returns:
            list[Callable]: List of optimizers which are configured the _config attribute. Use 
            config_training['update_rule'] == 'custom' for (temporary) customization. See also the source code
            of the function _update_rule_fm in 'core_code/util/config_extraction.py'.
        """

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



    def training_step(
        self, 
        batch: dict[list], 
        batch_idx: int
    ) -> dict[dict]:
        """Formulate one training step of the (task-specific) model(s) with task specifications based on the 
        configuration of the optimizers in the method 'configure_optimizers'.

        Args:
            batch (dict[list]): Task dictionary which has task keys and values of the format (x, y, task_activity).
            batch_idx (int): Id of the current batch (currently unused).

        Returns:
            dict[dict]: Dictionary with keys 'preds' and 'losses' which have task specific dictionaries as
            respective values.
        """
        torch.autograd.set_detect_anomaly(True) ########temp
        outputs, all_tasks = self._step_computation(batch)
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
            
                # compute optimizers steps without executing step
                optimizer.zero_grad()
                self.manual_backward(losses[task_key], retain_graph=True)
                
            for task_num in all_tasks: # repeat loop for optimizer steps
                task_key = 'task_{}'.format(task_num)
                optimizer = optimizers[task_key]
                optimizer.step()
        
        with torch.no_grad():
            return outputs
    


    def validation_step( 
        self, 
        batch: dict[list], 
        batch_idx: int
    ) -> dict[dict]:
        """Formulate one validation step of the (task-specific) model(s).

        Args:
            batch (dict[list]): Task dictionary which has task keys and values of the format (x, y, task_activity).
            batch_idx (int): Id of the current batch (currently unused).

        Returns:
            dict[dict]: Dictionary with key 'preds' which has a task specific dictionary as
            respective value.
        """

        outputs = self._step_computation(batch, bool_training=False)

        return outputs



    def test_step( 
        self, 
        batch: dict[list], 
        batch_idx: int
    ) -> dict[dict]:
        """Formulate one validation step of the (task-specific) model(s).

        Args:
            batch (dict[list]): Task dictionary which has task keys and values of the format (x, y, task_activity).
            batch_idx (int): Id of the current batch (currently unused).

        Returns:
            dict[dict]: Dictionary with key 'preds' which has a task specific dictionary as
            respective value.
        """

        outputs = self._step_computation(batch, bool_training=False)

        return outputs

    

    def fit(
        self, 
        data_module: DataModule,
        logger = None,
        callbacks: list = [],
        **config_trainer: dict
    ) -> pl.Trainer: 
        """_summary_

        Args:
            data_module (DataModule): Pytorch Lightning Datamodule (see 
            https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html).
            logger (pytorch_lightning.logger, optional): Pytorch lightning embedded logger object. Refer to
            pytorch_lightning.Trainer documentation. Defaults to None.
            callbacks (list, optional): List of PyTorch Lightning callbacks (see 
            https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html). Defaults to [].

        Optional arguments:
            [All the optional arguments are optional arguments for pytorch_lightning.Trainer. Refer to its
            documentation for more information.]

        Raises:
            ValueError: Raise an error if the [data] induced tasks and the [training, model] induced tasks
            are not identical. 

        Returns:
            pl.Trainer: pl.Trainer object which has been used for the training of the model(s).
        """
        
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

    

    def _step_computation(
        self, 
        batch: dict[list], 
        bool_training: bool = True
    ) -> dict[dict]: 
        """Step computation which differs based on whether we are in a training step or not.

        Args:
            batch (dict[list]): Task dictionary which has task keys and values of the format (x, y, task_activity).
            bool_training (bool, optional): Boolean to determine whether we are. Defaults to True.

        Returns:
            dict[dict]: Dictionary with key 'preds' keys 'preds' and 'losses' (based in bool_training==True)
            which have task specific dictionaries as respective values.
        """
        
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


    
    def _retrieve_batch_data( 
        self, 
        batch: dict[list], 
    ) -> list[torch.tensor]:
        """Retrieve the data from the batch (data).

        Args:
            batch (dict[list]): Task dictionary which has task keys and values of the format (x, y, task_activity).

        Returns:
            list[torch.tensor]: List of torch tensors of the form (x, y, task_activity) retrieved from the batch. 
        """

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