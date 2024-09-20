#!/bin/bash
n_runs=100
start = 75
import subprocess

for i in range(start,n_runs):
   current_seed = i*100 + i

   subprocess.run(['python','docker_reqs/ka_train.py','--seed',str(current_seed)])

