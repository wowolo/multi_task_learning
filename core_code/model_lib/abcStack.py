from typing import Callable
import torch
import core_code.util.config_extractions as util
from core_code.util.default_config import init_config_abcStack



class abc_CoreStack(torch.nn.Module):
    """ A single abc-parametrized layer. """

    def __init__(
        self, 
        input_width: int, 
        variable_width: int,
        output_width: int, 
        hidden_layer_activation: str,
        list_a: float, 
        list_b: float, 
        bias: bool = True, 
        bias_tune: float = 1,
        hidden_layer_activation_callback: Callable[[torch.tensor], torch.tensor] = None
    ):
        """Initialize the abc layer.

        Args:
            input_width (int): Input width.
            variable_width (int): Variable width.
            output_width (int): Output width.
            list_a (list_a (list): The list has to be of length 2. Aligned with the notation of the paper "Feature Learning of Infinite Width 
            Neural Networks" by Greg Yang et. al..
            list_b (list_a (list): The list has to be of length 2. Aligned with the notation of the paper "Feature Learning of Infinite Width 
            Neural Networks" by Greg Yang et. al..
            hidden_layer_activation (str): String to designate which hidden layer activation should be extracted.
            bias (bool, optional): Determines whether to have a Gaussian initialized bias parameter. Defaults to True.
            bias_tune (float, optional): Standard deviation of the initialization distribution of the bias parameter (if bias True). Defaults to 1.
            hidden_layer_activation_callback (Callable[[torch.tensor], torch.tensor], optional): Callback that inserts the hidden layer activation 
            given by callback if the value string is set to 'custom'. Defaults to None.
        """
        
        super().__init__()

        if (len(list_a) != 2) or (len(list_b) != 2):
            raise ValueError('The input lists for a and b parameters have to be of lenght 2 but len(list_a) = {} and len(list_b) = {}.'.format(len(list_a), len(list_b)))

        self.input_width = input_width
        self.variabel_width = variable_width
        self.output_width = output_width
        self.hidden_layer_activation = hidden_layer_activation
        self.list_a = list_a
        self.list_b = list_b
        self.bias_bool = bias
        self.bias_tune = bias_tune
        self.hidden_layer_activation_callback = hidden_layer_activation_callback

        self.initialize_parameters()



    def initialize_parameters(self):
        """Initialize all the parameters of the layer.
        """

        self.A_1 = torch.nn.Parameter(torch.randn(self.input_width, self.variabel_width) * self.input_width**(-self.list_b[0]))
        self.A_2 = torch.nn.Parameter(torch.randn(self.variabel_width, self.output_width) * self.variabel_width**(-self.list_b[1]))

        if self.bias_bool:
            self.bias_1 = torch.nn.Parameter(torch.randn(self.variabel_width))
            self.bias_2 = torch.nn.Parameter(torch.randn(self.output_width))
            self.bias_tune = torch.nn.Parameter(torch.ones(1) * self.bias_tune, requires_grad=False)
        else:
            self.bias_1 = torch.zeros(self.variabel_width)
            self.bias_2 = torch.zeros(self.output_width)



    def forward(
        self, 
        x: torch.tensor
    ) -> torch.tensor:
        """Forward pass of the layer.

        Args:
            x (torch.tensor): Input tensor.

        Returns:
            torch.tensor: Output tensor.
        """
        activation = util._hidden_bottleneck_activation_fm(self.hidden_layer_activation, self.hidden_layer_activation_callback)

        hidden = x @ self.A_1 * self.input_width**(-self.list_a[0]) + self.bias_1 * self.bias_tune
        hidden = activation()(hidden)
        out = hidden @ self.A_2 * self.variabel_width**(-self.list_a[1]) + self.bias_2 * self.bias_tune

        return out

    

    def reset_layer_parameters(self):
        """Reset the layer parameters by re-initializing them.
        """

        self.initialize_parameters()



class NNModel(torch.nn.Module):
    """ Neural network model indentified by the architecture key 'abcMLP' consisting of abc layers. """

    def __init__(
        self, 
        **config_architecture
    ):
        """Initialize the the neural network with optional parameters. Refer to the class source code and the 'core_code/util/default_config.py' file as reference.
        """
        super(NNModel, self).__init__()

        self.config_architecture = init_config_abcStack(**config_architecture)   
        
        self.layers = self.init_architecture()



    def init_architecture(self) -> torch.nn.ModuleList:
        """Initialize the architecture relevant layers.

        Returns:
            torch.nn.ModuleList: Listing of all the layers relevant for the network's architecture.
        """
        # extract necessary hyperparameters
        mod_list = []
        d_in = self.config_architecture['d_in']
        d_out = self.config_architecture['d_out']
        depth = self.config_architecture['depth']
        variable_width = self.config_architecture['variable_width']
        bottleneck_width = self.config_architecture['bottleneck_width']
        hidden_layer_activation = self.config_architecture['hidden_layer_activation']
        list_a = self.config_architecture['list_a']
        list_b = self.config_architecture['list_b']
        hidden_layer_activation_callback = self.config_architecture['hidden_layer_activation_callback']

        for i in range(depth):
            _list_a = list_a[2*i: 2*i+2]
            _list_b = list_b[2*i: 2*i+2]
            if depth == 1:
                mod_list.append(abc_CoreStack(d_in, variable_width, d_out, hidden_layer_activation, _list_a, _list_b, hidden_layer_activation_callback=hidden_layer_activation_callback))
            else:
                if i == 0:
                    mod_list.append(abc_CoreStack(d_in, variable_width, bottleneck_width, hidden_layer_activation, _list_a, _list_b, hidden_layer_activation_callback=hidden_layer_activation_callback))
                elif i < depth - 1:
                    mod_list.append(abc_CoreStack(bottleneck_width, variable_width, bottleneck_width, hidden_layer_activation, _list_a, _list_b, hidden_layer_activation_callback=hidden_layer_activation_callback))
                else:
                    mod_list.append(abc_CoreStack(bottleneck_width, variable_width, d_out, hidden_layer_activation, _list_a, _list_b, hidden_layer_activation_callback=hidden_layer_activation_callback))

        layers = torch.nn.ModuleList(mod_list)

        return layers


    
    def forward(
        self, 
        x: torch.tensor
    ) -> torch.tensor:
        """Forward pass of the neural network.

        Args:
            x (torch.tensor): Input tensor of dimension config_architecture['d_in'].

        Returns:
            torch.tensor: Output tensor of dimension config_architecture['d_out'].
        """

        activation = util._hidden_bottleneck_activation_fm(self.config_architecture['hidden_bottleneck_activation'], self.config_architecture['hidden_bottleneck_activation_callback'])
        for layer in self.layers[:-1]:
            x = activation()(layer(x))
        
        last_layer = self.layers[-1]
        x = last_layer(x)
        
        return x
    


    def reset_parameters(self):
        """Reset the parameters of the network by re-initialization.
        """
        for layer in self.layers:
            layer.reset_layer_parameters()