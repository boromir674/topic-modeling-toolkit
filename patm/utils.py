from configparser import ConfigParser
from collections import OrderedDict

_section2encoder = {
        'learning': int,
        'regularizers': str,
        'scores': str
    }


def cfg2model_settings(cfg_file):
    config = ConfigParser()
    config.read(cfg_file)
    return OrderedDict([(section.encode('utf-8'), OrderedDict([(setting_name.encode('utf-8'), _section2encoder[section](value)) for setting_name, value in config.items(section) if value])) for section in config.sections()])
