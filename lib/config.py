import yaml
import os
from easydict import EasyDict


def load_config(net_name='default'):
    filepath = os.path.join('./data/cfgs/', net_name + '.yml')
    filepath = os.path.abspath(filepath)
    with open(filepath, mode='r') as f:
        cfg = yaml.load(f.read())
        cfg = EasyDict(cfg)

    for k, v in cfg.items():
        print('%s: %s' % (k, v))
    return cfg


cfg = load_config()
