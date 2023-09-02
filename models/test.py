import torch
from game.zx_game import ZXGame
from models.model import Model

game = ZXGame(100, 100)
hdata = game.reset()
print('hdata = ', hdata)

model = Model(hdata.metadata(), hidden_channels=64, out_channels=4, num_heads=2, num_layers=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hdata, model = hdata.to(device.type), model.to(device)

with torch.no_grad():  # Initialize lazy modules.
    out = model(hdata.collect('x'), hdata.collect('edge_index'))
    print('out = ', out)
