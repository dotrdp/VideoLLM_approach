import torch.nn as nn
from umap import UMAP
import torch
'''
PENDING FOR REVISION, BUT UMAP DOES THE TRICK JUST FINE...

class ProjectionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(768, 512) 
'''

class ProjectionLayer:
    def __init__(self):
        self.umap = UMAP(n_components=30)
    def forward(self, x):
        x = x.cpu().detach().numpy()
        x = self.umap.fit_transform(x)
        return torch.tensor(x).to(torch.float32).cuda(device=0)