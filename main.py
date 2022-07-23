import argparse
from experiments.compositeSine.manager import Manager
from experiments.compositeSine.configs import configs_data, configs_architecture, configs_training, configs_custom



class StoreDict(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        
        kv = {}
        if not isinstance(values, (list,)):
            values = (values,)
        for value in values:
            n, *_l = value.split(':')
            v = _l[0]
            for i in range(1, len(_l)):
                v += (':' + _l[i])

            v = self._convert_v(v)

            kv[n] = v

        setattr(namespace, self.dest, kv)
    

    @staticmethod
    def _convert_v(v):
        # allow for int/float/bool conversion
        converted = False
        if not converted:
            try:
                v = int(v)
                converted = True
            except ValueError:
                pass
        if not converted:
            try:
                v = float(v)
                converted = True
            except ValueError:
                pass
        if v == 'True':
            v = True
            converted = True
        if v == 'False':
            v = False
            converted = True
        
        return v



parser = argparse.ArgumentParser()
parser.add_argument('--num_config', type=int)
parser.add_argument('--experimentbatch_name', type=str)
parser.add_argument('--config_trainer', action=StoreDict, nargs='*')



def main():

    args = parser.parse_args()
    
    num_config = args.num_config
    if not isinstance(num_config, type(None)):
        num_config -= 1
    
    experimentbatch_name = args.experimentbatch_name
    if isinstance(experimentbatch_name, type(None)):
        experimentbatch_name = 'default_name'
    
    config_trainer = args.config_trainer
    if isinstance(config_trainer, type(None)):
        config_trainer = {}


    # num_config = args.JOB_INDEX
    # if isinstance(num_config, type(None)):
    #     num_config = 1
    # num_config -= 1


    # config_trainer['fast_dev_run'] = True ######tmp

    manager = Manager(
        configs_data, 
        configs_architecture, 
        configs_training, 
        config_trainer, 
        configs_custom
    )
    
    manager.run(experimentbatch_name, ind_configs=num_config)



if __name__ == '__main__':
    main()