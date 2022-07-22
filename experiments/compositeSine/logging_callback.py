# handle all the logging via one callback class
from typing import Sequence
import tempfile
import os

import numpy as np
import torch
from pytorch_lightning import Callback, Trainer, LightningModule
import wandb

from matplotlib import pyplot as plt
from matplotlib import figure as Figure

import core_code.util.helpers as util
from core_code.util.default_config import init_config_data
from core_code.util.config_extractions import _criterion_fm



grayscale_list = ['black', 'dimgray', 'dimgrey', 'gray', 'grey', 'darkgray', 'darkgrey', 'silver', 'lightgray', 'lightgrey', 'gainsboro']



class LoggingCallback(Callback):  
    

    def __init__(
        self,
        x_train,
        y_train,
        task_activity_train,
        config_data: dict = {},
        logging_epoch_interval: int = 5,
    ):
        super().__init__()
        self.x_train = x_train.cpu()
        self.y_train = y_train.cpu()
        self.task_activity_train = task_activity_train
        self.config_data, self.all_losses = init_config_data(**config_data)
        self.logging_epoch_interval = logging_epoch_interval
        self.state_train = self._empty_state()
        self.state_val = self._empty_state()

    

    @staticmethod
    def _empty_state():
        state = { 
            'x': [],
            'y': [],
            'task_activity': [],
            'preds': [],
            'loss': [],
            'data_len': []
        } 
        return state



    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
    ) -> None:
        with torch.no_grad():
            _x, _y, _task_activity = pl_module._retrieve_batch_data(batch)
            data_len = _x.shape[0]
            
            self.state_train['data_len'].append(data_len)
            self.state_train['loss'].append(outputs['loss'])



    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        with torch.no_grad():
            _x, _y, _task_activity = pl_module._retrieve_batch_data(batch)
            data_len = _x.shape[0]
            
            self.state_val['data_len'].append(data_len)
    
            _preds = outputs['preds']

            # log validation loss
            loss = torch.zeros((1), requires_grad=False).type_as(_x)

            for task_num in torch.unique(_task_activity):

                task_config = util.extract_taskconfig(pl_module.config_training, task_num)
                _ind = (_task_activity == task_num)
                criterion = _criterion_fm(task_config['criterion'])
                loss = loss + criterion(_preds[_ind], _y[_ind])

            self.state_val['loss'].append(loss)

            if trainer.current_epoch % self.logging_epoch_interval == 1 or (trainer.current_epoch == (trainer.max_epochs-1)):
                self.state_val['x'].append(_x)
                self.state_val['y'].append(_y)
                self.state_val['task_activity'].append(_task_activity)
                self.state_val['preds'].append(_preds)
        
        

    def on_train_epoch_end(
        self, 
        trainer: Trainer, 
        pl_module: LightningModule
    ) -> None:
        # (data) weighted epoch loss
        data_len = self.state_train['data_len']
        epoch_loss = torch.sum(torch.Tensor([self.state_train['loss'][i] * data_len[i] for i in range(len(data_len))])) / torch.sum(torch.Tensor(data_len))
        
        trainer.logger.experiment.log({'train/loss': epoch_loss, 'epoch': trainer.current_epoch, 'global_step': trainer.global_step})

        if torch.isnan(epoch_loss):
            trainer.should_stop = True

        self.state_train = self._empty_state()



    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule
    ) -> None: 
        # (data) weighted epoch loss
        data_len = self.state_val['data_len']
        epoch_loss = torch.sum(torch.Tensor([self.state_val['loss'][i] * data_len[i] for i in range(len(data_len))])) / torch.sum(torch.Tensor(data_len))
        
        trainer.logger.experiment.log({'validation/loss': epoch_loss, 'epoch': trainer.current_epoch, 'global_step': trainer.global_step})
        
        if trainer.global_rank == 0:
            if (trainer.current_epoch % self.logging_epoch_interval == 1) or (trainer.current_epoch == (trainer.max_epochs-1)): # plot the images based on the states collected over the epoch
                
                del self.state_val['data_len']
                self.state_val = {key: torch.concat(self.state_val[key]).cpu() for key in self.state_val.keys()}

                # determine the plots we want to log and log them with self._log_plot
                unique_activities = torch.unique(self.state_val['task_activity']).int()
                for task_num in unique_activities:
                    task_num = int(task_num)
                    active_dimensions = util.extract_taskconfig(pl_module.config_training, task_num)['criterion'][1]
                    for d in active_dimensions:
                        self._log_plot(
                            trainer,
                            task_num,
                            d,
                        )

        self.state_val = self._empty_state() # important to keep emptying the chached data in state_val when not needed!


            
    def _log_plot(
        self,
        trainer,
        task_num,
        d
    ) -> Figure:

        fig = plt.figure()

        # task_config = util.extract_taskconfig(self.config_data, task_num) 

        _ind = (self.task_activity_train == task_num)
        task_x_train = self.x_train[_ind]
        task_y_train = self.y_train[_ind]

        _ind = (self.state_val['task_activity'] == task_num)
        task_x_val = self.state_val['x'][_ind]
        task_y_val = self.state_val['y'][_ind]

        task_preds_val = self.state_val['preds'][_ind]
    
        # determine relevant x, y, preds data for task via task activity
        set_counter = 0
        
        # plot the used training data
        plt.plot(
            task_x_train, 
            task_y_train[:,d], 
            'o', 
            color=grayscale_list[set_counter], 
            markersize=3, 
            label='Training data - task {}'.format(task_num)
        )
        set_counter = (set_counter + 1) % len(grayscale_list)

        # plot the true function on the validation data
        
        plt.plot(
            task_x_val,
            task_y_val[:,d], 
            'k-', 
            label='True function'
        )
        
        # plot the NN prediction on the validation data
        plt.plot(task_x_val.cpu(), task_preds_val[:,d].cpu(), 'r-', label='Neural network')
        
        plt.title('Task {} - Output dimension {}'.format(task_num, d))
            
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=3)
        
        # log the plots - jpg
        # os.environ["TMPDIR"] = os.getcwd()


        _dir = os.getenv('HOME')
        with tempfile.TemporaryDirectory(dir=_dir) as tmp_dir:
            filename = tmp_dir + '/' + 'tmp.jpg'
            fig.savefig(filename)
            tmp_img = plt.imread(filename)

            wandb.log(
                {
                    'validation/plot_task{}_dim{}_jpg'.format(task_num, d): [wandb.Image(tmp_img)], 
                    'epoch': trainer.current_epoch
                }
            )

        # io_buf = io.BytesIO()
        # fig.savefig(io_buf, format='png')
        # io_buf.seek(0)
        # tmp_img = plt.imread(io_buf)
        # io_buf.close()

        # wandb.log(
        #     {
        #         'validation/plot_task{}_dim{}_jpg'.format(task_num, d): [wandb.Image(tmp_img)], 
        #         'epoch': trainer.current_epoch
        #     }
        # )

        plt.close(fig)