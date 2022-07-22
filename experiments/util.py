from sklearn.model_selection import ParameterGrid



class BasicManager():

    @staticmethod
    def dictvals_to_list(dict):

        for key, val in dict.items():
            if not(isinstance(val, list)):
                dict[key] = [val] 

        return dict



    def grid_config_lists(self, *args): 

        configs = {}
        for config in args:
            configs.update(config)

        configs = self.dictvals_to_list(configs)

        grid = ParameterGrid(configs)

        args_list_of_configs = [[]] * len(args)

        for new_config in grid:

            for i in range(len(args)):

                temp_dict = {key: new_config[key] for key in args[i].keys()}
                args_list_of_configs[i] = args_list_of_configs[i] + [temp_dict]

        return args_list_of_configs
    


    def _create_grid(self): 
        # list of configs
        list_of_lists = self.grid_config_lists(
            self.configs_data, 
            self.configs_architecture, 
            self.configs_traininig,
            self.configs_trainer, 
            self.configs_custom
        )

        self.num_experiments = len(list_of_lists[0])

        return list_of_lists