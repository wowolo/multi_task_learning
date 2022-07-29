import torch
import core_code.util.config_extractions as util
from core_code.util.default_config import init_config_abcMLP



class abc_Layer(torch.nn.Module):
    """ A single abc-parametrized layer. """

    def __init__(
        self, 
        input_width: int, 
        output_width: int, 
        a: float, 
        b: float, 
        bias: bool = True, 
        bias_tune: float = 1
    ):
        """Initialize the abc layer.

        Args:
            input_width (int): Input width.
            output_width (int): Output width.
            a (float): Aligned with the notation of the paper "Feature Learning of Infinite Width Neural Networks" by Greg Yang et. al..
            b (float): Aligned with the notation of the paper "Feature Learning of Infinite Width Neural Networks" by Greg Yang et. al..
            bias (bool, optional): Determines whether to have a Gaussian initialized bias parameter. Defaults to True.
            bias_tune (float, optional): Standard deviation of the initialization distribution of the bias parameter (if bias True). Defaults to 1.
        """
        
        super().__init__()

        self.input_width = input_width
        self.output_width = output_width
        self.a = a
        self.b = b
        self.bias_bool = bias
        self.bias_tune = bias_tune


        self.initialize_parameters()



    def initialize_parameters(self):
        """Initialize all the parameters of the layer.
        """

        self.A = torch.nn.Parameter(torch.randn(self.input_width, self.output_width) * self.input_width**(-self.b))

        if self.bias_bool:
            self.bias = torch.nn.Parameter(torch.randn(self.output_width))
            self.bias_tune = torch.nn.Parameter(torch.ones(1) * self.bias_tune, requires_grad=False)
        else:
            self.bias = torch.zeros(self.output_width)
    


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

        return x @ self.A * self.input_width**(-self.a) + self.bias * self.bias_tune

    

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

        self.config_architecture = init_config_abcMLP(**config_architecture)   
        
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
        width = self.config_architecture['width']
        list_a = self.config_architecture['list_a']
        list_b = self.config_architecture['list_b']

        for i in range(depth):
            if depth == 1:
                mod_list.append(abc_Layer(d_in, d_out, list_a[i], list_b[i]))
            else:
                if i == 0:
                    mod_list.append(abc_Layer(d_in, width, list_a[i], list_b[i]))
                elif i < depth - 1:
                    mod_list.append(abc_Layer(width, width, list_a[i], list_b[i]))
                else:
                    mod_list.append(abc_Layer(width, d_out, list_a[i], list_b[i]))

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

        activation = util._hidden_layer_activation_fm(self.config_architecture['hidden_layer_activation'], self.config_architecture['hidden_layer_activation_callback'])
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