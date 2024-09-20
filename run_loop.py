#!/bin/bash
n_runs=100
start = 0
import subprocess

for i in range(start,n_runs):
   current_seed = i*100 + i

   subprocess.run(['python','ka_train.py','--seed',str(current_seed)])

