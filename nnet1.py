import torch
import torch.nn as nn
import torch.nn.functional as F

class nnet1(nn.module):
    """
    Defining layers of for the first attempt at solving embedding corrector problem.
    """
    def __init__(self):
    """
    Constructor for nnet1 class.
    """
        super(nnet1, self).__init__()
        self.layer1 = nn.Linear(512, 8192)
        self.layer2 = nn.Linear(8192, 4096)
        self.layer3 = nn.Linear(4096, 2048)
        self.layer4 = nn.Linear(2048,1024)
        self.layer5 = nn.Linear(1024, 512)
