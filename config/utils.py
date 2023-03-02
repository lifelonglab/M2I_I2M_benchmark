import os

import yaml

from paths import ROOT_PATH


def load_config(config_name):
    config_path = f'{ROOT_PATH}/config'
    with open(os.path.join(config_path, config_name)) as file:
        config = yaml.safe_load(file)

    return config
