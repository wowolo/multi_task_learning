from typing import Callable
import torch
import core_code.util.config_extractions as util

from core_code.util.default_config import init_config_Stack




class Stack_Core(torch.nn.Module):
    """A single Stack core layer. """

    def __init__(
        self, 
        input_width: int, 
        output_width: int, 
        variable_width: int, 
        hidden_layer_activation: str, 
        skip_conn: bool, 
        linear_skip_conn: bool, 
        linear_skip_conn_width: int,
        hidden_layer_activation_callback: Callable[[torch.tensor], torch.tensor] = None
    ):
        """Initialize the Stack core layer.

        Args:
            input_width (int): Input width.
            output_width (int): Output width.
            variable_width (int): Variable width, i.e., hidden layer width of the Stach core layer.
            hidden_layer_activation (str): String to determine the hidden layer activation.
            skip_conn (bool): Whether to allow skip connections.
            linear_skip_conn (bool): Whether to allow skip connections with a linear embedding in between.
            linear_skip_conn_width (int): Width of the linear embedding used by the linear skip connection.
            hidden_layer_activation_callback (Callable[[torch.tensor], torch.tensor]): Callback for the hidden layer activation if the 
            hidden_layer_activation string equals 'custom'. Default is None.
        """
        
        super().__init__()

        self.input_width = input_width
        self.output_width = output_width
        self.variable_width = variable_width
        self.hidden_layer_activation = hidden_layer_activation
        self.hidden_layer_activation_callback = hidden_layer_activation_callback
        

        self.skip_conn = skip_conn
        self.linear_skip_conn = linear_skip_conn
        self.linear_skip_conn_width = linear_skip_conn_width

        self.hidden_input_width = variable_width
        if self.skip_conn:
            self.hidden_input_width += input_width
        if self.linear_skip_conn:
            self.hidden_input_width += self.linear_skip_conn_width


        self.initialize_parameters()



    def initialize_parameters(self):
        """Initialize all the parameters of the layer.
        """

        self.linear_1 = torch.nn.Linear(self.input_width, self.variable_width)
        
        if self.linear_skip_conn:
            self.linear_skip = torch.nn.Linear(self.input_width, self.linear_skip_conn_width)

        self.linear_2 = torch.nn.Linear(self.hidden_input_width, self.output_width)
    


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

        activation = util._hidden_layer_activation_fm(self.hidden_layer_activation, self.hidden_layer_activation_callback)
        hidden_input = activation()(self.linear_1(x))

        if self.skip_conn:
            hidden_input = torch.cat((hidden_input, x), 1)

        if self.linear_skip_conn:
            linear_input = self.linear_skip(x)
            hidden_input = torch.cat((hidden_input, linear_input), 1)

        y = self.linear_2(hidden_input)

        return y

    

    def reset_layer_parameters(self):
        """Reset the layer parameters by re-initializing them.
        """

        self.initialize_parameters()




class NNModel(torch.nn.Module):    
    """ Neural network model indentified by the architecture key 'Stack' consisting of Stack core layers. """

    def __init__(
        self, 
        **config_architecture
    ):
        """Initialize the the neural network with optional parameters. Refer to the class source code and the 'core_code/util/default_config.py' file as reference.
        """
        super(NNModel, self).__init__()

        self.config_architecture = init_config_Stack(**config_architecture)    
        self.layers = self.init_architecture()



    def init_architecture(self) -> torch.nn.ModuleList:
        """Initialize the architecture relevant layers.

        Returns:
            torch.nn.ModuleList: Listing of all the layers relevant for the network's architecture.
        """
        mod_list = []
        depth = self.config_architecture['depth']

        standard_input_wo_dim = [
            self.config_architecture['variable_width'],
            self.config_architecture['hidden_layer_activation'],
            self.config_architecture['skip_conn'],
            self.config_architecture['linear_skip_conn'],
            self.config_architecture['linear_skip_conn_width']
        ]

        for i in range(depth):
            if depth == 1:
                mod_list.append(Stack_Core(
                            self.config_architecture['d_in'], 
                            self.config_architecture['d_out'],
                            *standard_input_wo_dim
                        )
                    )
            else:
                if i == 0:
                    mod_list.append(Stack_Core(
                            self.config_architecture['d_in'], 
                            self.config_architecture['bottleneck_width'],
                            *standard_input_wo_dim
                        )
                    )
                elif i < depth - 1:
                    mod_list.append(Stack_Core(
                            self.config_architecture['bottleneck_width'], 
                            self.config_architecture['bottleneck_width'],
                            *standard_input_wo_dim
                        )
                    )
                else:
                    mod_list.append(Stack_Core(
                            self.config_architecture['bottleneck_width'], 
                            self.config_architecture['d_out'],
                            *standard_input_wo_dim
                        )
                    )

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

        for layer in self.layers[:-1]:
            activation = util._hidden_bottleneck_activation_fm(
                self.config_architecture['hidden_bottleneck_activation'], 
                self.config_architecture['hidden_bottleneck_activation_callback']
            )
            x = activation()(layer(x))
        
        last_layer = self.layers[-1]
        x = last_layer(x)
        
        return x



    def reset_parameters(self):
        """Reset the parameters of the network by re-initialization.
        """
        for layer in self.layers:
            layer.reset_layer_parameters()