import dgl
import torch

hg = dgl.heterograph({
    ('user', 'develops', 'activity'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
    ('developer', 'has', 'game'): (torch.tensor([0, 1]), torch.tensor([0, 1]))
})

hg.add_edges(0, 1, etype=('user', 'develops', 'activity'))
hg.add_edges(0, 1, etype=('user', 'develops', 'activity'))
print(hg)
print(hg.edges[[0, 1]]('develops'))