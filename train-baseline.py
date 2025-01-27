import pandas as pd
import torch
import torch.nn as nn

from tqdm import tqdm

import plotly.express as px
px.defaults.template = 'plotly_dark'

from sklearn.model_selection import train_test_split
from IPython import display

ITERS = 200
LR = 1e-2
LATENT_DIM = 4
LAMBDA = 1e-5
SEED = 1
torch.manual_seed(SEED)

class MFBaseline(nn.Module):
    def __init__(self, n_users, n_items, n_latent):
        super().__init__()
        self.emb_u = nn.Embedding(n_users, n_latent)
        self.emb_i = nn.Embedding(n_items, n_latent)
        
        nn.init.orthogonal_(self.emb_u.weight)
        nn.init.orthogonal_(self.emb_i.weight)

    def fit(self, r, u, i, u_val, i_val, epochs=100, loss_fn=None, lr=1e-2, weight_lambda=1e-5):
        optim = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}.get(loss_fn, nn.MSELoss())
        
        losses = {'train': [], 'val': []}
        for _ in tqdm(range(epochs), desc='Training MF Baseline'):
            v = self(u, i)
            loss = loss_fn(v, r[u, i])
            loss_reg = loss + weight_lambda * self.emb_u.weight.norm(p=2) + weight_lambda * self.emb_i.weight.norm(p=2)
            optim.zero_grad()
            loss_reg.backward()
            optim.step()
            
            losses['train'].append(loss.item())
            
            with torch.no_grad():
                v = self(u_val, i_val)
                loss_val = loss_fn(v, r[u_val, i_val])
                losses['val'].append(loss_val.item())

        return losses

    def forward(self, u, i):
        e_u = self.emb_u(u)
        e_i = self.emb_i(i)
        return (e_u * e_i).sum(1)


# Load data
df = pd.read_csv('../ml-latest-small/ratings.csv')

unique_users = df.userId.unique()
unique_movies = df.movieId.unique()

# Create user-item matrix
r_ui = torch.zeros(len(unique_users), len(unique_movies)) * float('nan')
r_ui = r_ui.cuda()
for i, j, r in zip(df.userId.factorize()[0], df.movieId.factorize()[0], df.rating):
    r_ui[i, j] = r

# Split data
rows, cols = r_ui.isfinite().nonzero().split(1, dim=1)

idx = range(len(rows))
train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=SEED)

mf_baseline = MFBaseline(len(unique_users), len(unique_movies), LATENT_DIM).cuda()
u = rows[train_idx].squeeze().cuda()
i = cols[train_idx].squeeze().cuda()

u_val = rows[test_idx].squeeze().cuda()
i_val = cols[test_idx].squeeze().cuda()

l1 = nn.L1Loss()

losses = mf_baseline.fit(r_ui, u, i, u_val, i_val, epochs=ITERS, loss_fn='mae', lr=LR, weight_lambda=LAMBDA)

with torch.no_grad():
    u = rows[test_idx].squeeze().cuda()
    i = cols[test_idx].squeeze().cuda()

    mf_loss_test = (mf_baseline(u, i) - r_ui[u, i]).cpu()

print(f'Baseline test loss: {mf_loss_test.abs().mean().item()}')

jit_baseline = torch.jit.script(mf_baseline.cpu())
torch.jit.save(jit_baseline, './baseline.pt')
print('Baseline model saved!')