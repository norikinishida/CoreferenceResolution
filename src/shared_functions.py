import random

import numpy as np
import torch

import utils


####################################
# Debug
####################################


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    utils.writelog("Set random seed: %d" % seed)

