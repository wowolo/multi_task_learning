import torch
from core_code.util.function_lib import function_library


class dimred_MSELoss(torch.nn.Module):

    def __init__(self, dimension_activity):

        super(dimred_MSELoss,self).__init__()
        self.dimension_activity = dimension_activity
    

    
    def forward(self, output, target):

        if output.shape[0] == 0: return 0

        dimred_output = output[:, self.dimension_activity]
        dimred_target = target[:, self.dimension_activity]

        return torch.sum((dimred_output - dimred_target)**2)



def _f_true_fm(value):
    f_true = function_library(value)
    return f_true



def _criterion_fm(value):
    if isinstance(value, str):
        return {
            'MSELoss': torch.nn.MSELoss(),
        }[value]
    elif isinstance(value, tuple):
        args = value[1:]
        string = value[0]
        return {
            'dimred_MSELoss': dimred_MSELoss(*args),
        }[string]
    else:
        raise ReferenceError('The function keyword is not yet implemented.')



def _update_rule_fm(value):
    return {
        'Adam': torch.optim.Adam,
        'SGD': torch.optim.SGD,
    }[value]



class identity_activation(torch.nn.Module): # currently implemented as Identity
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x


def _hidden_bottleneck_activation_fm(string):
    return {
        'Identity': identity_activation,
        'ReLU': torch.nn.ReLU,
    }[string]



def _hidden_layer_activation_fm(string):
    return {
        'Identity': identity_activation,
        'ReLU': torch.nn.ReLU,
    }[string]