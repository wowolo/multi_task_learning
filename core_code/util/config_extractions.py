from subprocess import call
from typing import Callable
import numpy as np
import torch
from core_code.util.function_lib import function_library

    

class dimred_MSELoss(torch.nn.Module):
    """ Mean squared error loss which only considers a subset of the tensor dimensions to which it is applied. """

    def __init__(
        self, 
        dimension_activity: list
    ):
        """Mean sqaured error loss which is active on input defined dimensions only (instead of all tensor dimensions).

        Args:
            dimension_activity (list): Dimensions to which the input tensors are reduced and the MSE loss applied, i.e., 
            tensor = tensor[:, dimension_activity].
        """

        super(dimred_MSELoss,self).__init__()
        self.dimension_activity = dimension_activity
    

    
    def forward(
        self, 
        output: torch.tensor, 
        target: torch.tensor
    ) -> torch.tensor:
        """Forward pass of the dimension modified MSE loss.

        Args:
            output (torch.tensor): Output tensor which is reduced according to dimension_activity.
            target (torch.tensor): Target tensor which is reduced according to dimension_activity.

        Returns:
            torch.tensor: Modified MSE loss as torch tensor (with underlying computation graph for autograd).
        """

        if output.shape[0] == 0: return 0

        dimred_output = output[:, self.dimension_activity]
        dimred_target = target[:, self.dimension_activity]

        return torch.sum((dimred_output - dimred_target)**2)



def _f_true_fm(
        value: str,
        callback: Callable[[np.array], np.array] = None
    ) -> Callable[[np.array], np.array]:
    """'Function maker' to extract the f_true function from the function_library in 'core_code/util/function_lib.py'.

    Args:
        value (str): String to designate which function should be extracted via function_library.
        callback (Callable[[np.array], np.array], optional): Callback that inserts f_true given by callback 
        if the value string is set to 'custom'. Defaults to None.

    Returns:
        Callable[[np.array], np.array]: Extracted function.
    """
    
    if value != 'custom':
        return function_library(value)
    else:
        return callback



def _criterion_fm(
        value: str,
        callback: Callable[[torch.tensor], torch.tensor] = None
    ) -> Callable[[torch.tensor], torch.tensor]:
    """'Function maker' to extract the criterion based on the deposited objects in its the function's source code.

    Args:
        value (str): String to designate which criterion should be extracted.
        callback (Callable[[torch.tensor], torch.tensor], optional): Callback that inserts the criterion given by callback
        if the value string is set to 'custom'. Defaults to None.

    Raises:
        ReferenceError: Raised in the case of the extraction not being implemented based on the given string in value.

    Returns:
        Callable[[torch.tensor], torch.tensor]: Extracted criterion.
    """

    if value != 'custom':
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
    
    else: 
        return callback



def _update_rule_fm(
        value: str,
        callback: Callable = None
    ) -> Callable:
    """'Function maker' to extract the update rule based on the deposited objects in its the function's source code.

    Args:
        value (str): String to designate which update rule should be extracted.
        callback (Callable, optional): Callback that inserts the update rule given by callback
        if the value string is set to 'custom'. Defaults to None.

    Returns:
        Callable: Extracted update rule.
    """
    
    if value != 'custom':
        return {
            'Adam': torch.optim.Adam,
            'SGD': torch.optim.SGD,
        }[value]
    else:
        return callback



class identity_activation(torch.nn.Module): # currently implemented as Identity
    """ Helper torch.nn.Module to define the identity on tensors. """
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, 
        x: torch.tensor
    ) -> torch.tensor:
        """Forward pass of the identity.

        Args:
            x (torch.tensor): Input tensor.

        Returns:
            torch.tensor: Output tensor (x).
        """

        return x


def _hidden_bottleneck_activation_fm(
        value: str,
        callback: Callable[[torch.tensor], torch.tensor] = None
    ) -> Callable[[torch.tensor], torch.tensor]:
    """'Function maker' to extract the bottleneck activation based on the deposited objects in its the function's source code.

    Args:
        value (str): String to designate which bottleneck activation should be extracted.
        callback (Callable[[torch.tensor], torch.tensor], optional): Callback that inserts the hidden
        bottleneck activation given by callback if the value string is set to 'custom'. Defaults to None.

    Returns:
        Callable[[torch.tensor], torch.tensor]: Extracted bottleneck activation.
    """

    if value != 'custom':
        return {
            'Identity': identity_activation,
            'ReLU': torch.nn.ReLU,
        }[value]
    else:
        return callback



def _hidden_layer_activation_fm(
        value: str,
        callback: Callable[[torch.tensor], torch.tensor] = None
    ) -> Callable[[torch.tensor], torch.tensor]:
    """'Function maker' to extract the hidden layer activation based on the deposited objects in its the function's source code.

    Args:
        value (str): String to designate which hidden layer activation should be extracted.
        callback (Callable[[torch.tensor], torch.tensor], optional): Callback that inserts the hidden layer activation 
        given by callback if the value string is set to 'custom'. Defaults to None.

    Returns:
        Callable[[torch.tensor], torch.tensor]: Extracted hidden layer activation.
    """

    if value != 'custom':
        return {
            'Identity': identity_activation,
            'ReLU': torch.nn.ReLU,
        }[value]
    else:
        return callback