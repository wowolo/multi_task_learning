import torch
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger


import core_code.util.helpers as util
from core_code.util.default_config import init_config_training, init_config_trainer
from core_code.util.config_extractions import _criterion_fm, _update_rule_fm

from core_code.util.lightning import DataModule



class LightningModel(pl.LightningModule):


    def __init__(self, model, **config_training):
        super(LightningModel, self).__init__()
        
        self.model = model # create model with model_selector
        
        self.config_training, self.all_tasks = init_config_training(**config_training)
        # self.register_buffer("config_training", tmp_config_training)
        # self.register_buffer("all_tasks", tmp_all_tasks)



    def forward(self, x):
        return self.model(x).type_as(x)



    def configure_optimizers(self):
        update = _update_rule_fm(self.config_training['update_rule'])
        optimizer = update(self.model.parameters(), lr=self.config_training['learning_rate'])
        return optimizer



    def training_step(self, batch, batch_idx):
        outputs = self._compute_combined_taskloss(batch)
        
        # add regularization terms to loss
        loss = outputs['loss']

        reg_alpha = self.config_training['regularization_alpha']
        if reg_alpha != 0:

            reg = torch.tensor(0., requires_grad=True).type_as(loss)
            for param in self.model.parameters():
                reg = reg + torch.linalg.vector_norm(param.flatten(), ord=self.config_training['regularization_ord'])**2
        
            loss = loss + reg_alpha * reg
        
        outputs['loss'] =  loss

        return outputs
    


    def validation_step(self, batch, batch_idx): # criterion without regularization on validation set
        outputs = self._compute_combined_taskloss(batch, bool_training=False)

        return outputs



    def test_step(self, batch, batch_idx): # criterion without regularization on test set
        outputs = self._compute_combined_taskloss(batch, bool_training=False)

        return outputs

    

    def fit(
        self, 
        data_module: DataModule,
        project: str = None,
        name: str = None,
        callbacks: list = [],
        **config_trainer: dict
    ): 
        # TODO setup method such that it can be used with flags given via bash sript - flags determined by trainer
        config_trainer = init_config_trainer(**config_trainer)

        wandb.login()
        logger = WandbLogger(
            project = project,
            name = name, 
            log_model=True
        )

        trainer = pl.Trainer(
            logger=logger,
            callbacks=callbacks,
            **config_trainer
        )
        
        # log the configurations
        if trainer.global_rank == 0:
            logger.experiment.config.update(data_module.data.config_data)
            logger.experiment.config.update(self.model.config_architecture)
            logger.experiment.config.update(self.config_training)
            logger.experiment.config.update(config_trainer)
        
        trainer.fit(self, data_module)

    

    def _compute_combined_taskloss(self, batch, bool_training=True): # convenience function used in the step methods
        
        x, y, task_activity = self._retrieve_batch_data(batch)
        preds = self.forward(x)
        
        if not bool_training:
            return {'preds': preds}

        else:
            # compute loss based on task configurations 
            loss = torch.zeros((1), requires_grad=True).type_as(x)
            unique_activities = torch.unique(task_activity).int()

            for task_num in unique_activities:

                task_config = util.extract_taskconfig(self.config_training, task_num)

                _ind = (task_activity == task_num)
                criterion = _criterion_fm(task_config['criterion'])

                loss = loss + criterion(preds[_ind], y[_ind])

            return {'preds': preds, 'loss': loss}


    
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