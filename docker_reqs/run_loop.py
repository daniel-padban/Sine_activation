#!/bin/bash
n_runs=10
import torch
torch.manual_seed(100)
torch.cuda.manual_seed(100)
torch.cuda.manual_seed_all(100)

for i in range(n_runs):
    import ka_train

