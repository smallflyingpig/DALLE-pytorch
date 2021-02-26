from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
from easydict import EasyDict as edict


__C = edict()
cfg = __C

__C.EPOCHS = 20
__C.BATCH_SIZE = 8
__C.LEARNING_RATE = 1e-3
__C.LR_DECAY_RATE = 0.98
__C.NUM_TOKENS = 8192
__C.NUM_LAYERS = 2
__C.NUM_RESNET_BLOCKS = 2
__C.SMOOTH_L1_LOSS = False
__C.EMB_DIM = 512
__C.HID_DIM = 256
__C.KL_LOSS_WEIGHT = 0
__C.STARTING_TEMP = 1.
__C.TEMP_MIN = 0.5
__C.ANNEAL_RATE = 1e-6
__C.NUM_IMAGES_SAVE = 4



def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)
