


def report_config(config):

    print('The following configuration has been used for the construction of the {} \
architecture:'.format(config['architecture_key']))

    for paramkey in config.keys():
        print('{}: {}'.format(paramkey, config[paramkey]))