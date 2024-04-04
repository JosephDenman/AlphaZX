import torch
import torch.nn.functional as torch_func

ETYPE_COUNT = 2

B_ETYPE_INDEX = 0
B_ETYPE_NAME = 'base'
B_ETYPE_ONE_HOT = torch_func.one_hot(torch.tensor(B_ETYPE_INDEX), ETYPE_COUNT)

I_ETYPE_INDEX = 1
I_ETYPE_NAME = 'inclusion'
I_ETYPE_ONE_HOT = torch_func.one_hot(torch.tensor(I_ETYPE_INDEX), ETYPE_COUNT)
