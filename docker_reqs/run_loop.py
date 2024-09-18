#!/bin/bash
n_runs=100

import subprocess

for i in range(n_runs):
   subprocess.run(['python','docker_reqs/ka_train.py','--seed',str(i)])

