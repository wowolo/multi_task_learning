import torch
import core_code.util.config_extractions as util

from core_code.util.default_config import init_config_Stack




class Stack_Core(torch.nn.Module):

    def __init__(self, input_width, output_width, variable_width, hidden_layer_activation, skip_conn, linear_skip_conn, linear_skip_conn_width):
        
        super().__init__()

        self.input_width = input_width
        self.output_width = output_width
        self.variable_width = variable_width
        self.hidden_layer_activation = hidden_layer_activation
        

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

        self.linear_1 = torch.nn.Linear(self.input_width, self.variable_width)
        
        if self.linear_skip_conn:
            self.linear_skip = torch.nn.Linear(self.input_width, self.linear_skip_conn_width)

        self.linear_2 = torch.nn.Linear(self.hidden_input_width, self.output_width)
    


    def forward(self, x):

        activation = util._hidden_layer_activation_fm(self.hidden_layer_activation)
        hidden_input = activation()(self.linear_1(x))

        if self.skip_conn:
            hidden_input = torch.cat((hidden_input, x), 1)

        if self.linear_skip_conn:
            linear_input = self.linear_skip(x)
            hidden_input = torch.cat((hidden_input, linear_input), 1)

        y = self.linear_2(hidden_input)

        return y

    

    def reset_layer_parameters(self):
        self.initialize_parameters()




class NNModel(torch.nn.Module):    


    def __init__(self, **config_architecture):
        super(NNModel, self).__init__()

        self.config_architecture = init_config_Stack(**config_architecture)    
        self.layers = self.init_architecture()



    def init_architecture(self):
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
    


    def forward(self, x):

        for layer in self.layers[:-1]:
            activation = util._hidden_bottleneck_activation_fm(self.config_architecture['hidden_bottleneck_activation'])
            x = activation()(layer(x))
        
        last_layer = self.layers[-1]
        x = last_layer(x)
        
        return x