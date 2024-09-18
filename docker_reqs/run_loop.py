#!/bin/bash
n_runs=100

import subprocess

for i in range(n_runs):
   current_seed = i*100 + i

   subprocess.run(['python','docker_reqs/ka_train.py','--seed',str(current_seed)])

