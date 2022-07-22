import torch
import core_code.util.config_extractions as util
from core_code.util.default_config import init_config_abcMLP



class abc_Layer(torch.nn.Module):

    def __init__(self, input_width, output_width, a, b, bias=True, bias_tune=1):
        
        super().__init__()

        self.input_width = input_width
        self.output_width = output_width
        self.a = a
        self.b = b
        self.bias_bool = bias
        self.bias_tune = bias_tune


        self.initialize_parameters()



    def initialize_parameters(self):

        self.A = torch.nn.Parameter(torch.randn(self.input_width, self.output_width) * self.input_width**(-self.b))

        if self.bias_bool:
            self.bias = torch.nn.Parameter(torch.randn(self.output_width))
            self.bias_tune = torch.nn.Parameter(torch.ones(1) * self.bias_tune, requires_grad=False)
        else:
            self.bias = torch.zeros(self.output_width)
    


    def forward(self, x):
        return x @ self.A * self.input_width**(-self.a) + self.bias * self.bias_tune

    

    def reset_layer_parameters(self):
        self.initialize_parameters()



class NNModel(torch.nn.Module):


    def __init__(self, **config_architecture):
        super(NNModel, self).__init__()

        self.config_architecture = init_config_abcMLP(**config_architecture)   
        if isinstance(self.config_architecture['list_a'], int):
            self.config_architecture['list_a'] = [self.config_architecture['list_a'] for i in range(self.config_architecture['depth'])]
        if isinstance(self.config_architecture['list_b'], int):
            self.config_architecture['list_b'] = [self.config_architecture['list_b'] for i in range(self.config_architecture['depth'])] 
        
        self.layers = self.init_architecture()



    def init_architecture(self):
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


    
    def forward(self, x):

        activation = util._hidden_layer_activation_fm(self.config_architecture['hidden_layer_activation'])
        for layer in self.layers[:-1]:
            x = activation()(layer(x))
        
        last_layer = self.layers[-1]
        x = last_layer(x)
        
        return x
    


    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_layer_parameters()