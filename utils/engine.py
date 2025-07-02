import torch
import numpy as np
import random


def set_random_seed(seed=0):
    """
    fix the random seed

    :seed: the seed want to set
    """
    
    print('set random seed to {}'.format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
