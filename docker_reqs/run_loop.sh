
#!/bin/bash
n_runs=10

for i in $(seq 1 $n_runs); do
    echo "Run $i of $n_runs"
    python docker_reqs/ka_train.py
done

