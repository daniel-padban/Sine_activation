#!/bin/bash
n_runs=100

import subprocess
import torch
torch.manual_seed(100)
torch.cuda.manual_seed(100)
torch.cuda.manual_seed_all(100)

for i in range(n_runs):
   subprocess.run(['python','docker_reqs/ka_train.py'])

