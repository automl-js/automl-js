#!/usr/bin/env bash
rm -f tmp.csv
python3 benchmark.py run --model=DecisionTreeClassifier --params='{"max_depth":4}' --name=tree
