
import pdb
import os
import sys
import argparse

import math
import torch
import numpy as np
from utils import  accuracy
import random
os.environ["PYTHONHASHSEED"] = str(2025)
random.seed(2025)
np.random.seed(2025)
torch.manual_seed(2025)
torch.cuda.manual_seed(2025)
torch.cuda.manual_seed_all(2025) 
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
try:
    import torch_geometric
    torch_geometric.seed_everything(2025)
except:
    pass



def run_epoch(model, optimizer, adata,args, is_training):
    if is_training:
        model.train()
    else:
        model.eval()

    if args.Model_name =='Simple':
        feat, Losses, dae_loss, bce_loss, Concept_loss, Contrastive_loss = model(adata,args.Network,args.device)
    else:
        feat, Losses, dae_loss, bce_loss, Concept_loss, Contrastive_loss= model(adata)

    if is_training:
        optimizer.zero_grad()
        Losses.backward()
        optimizer.step()
    return feat, Losses, dae_loss, bce_loss, Concept_loss, Contrastive_loss
