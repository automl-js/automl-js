#!/usr/bin/env bash

# Linear models
python3 benchmark.py run --model=SGDRegressor --params='{"max_iter":10,"penalty":"elasticnet","l1_ratio":0.0}'
python3 benchmark.py run --model=SGDRegressor --params='{"max_iter":100,"penalty":"elasticnet","l1_ratio":0.0}'
python3 benchmark.py run --model=SGDRegressor --params='{"max_iter":1000,"penalty":"elasticnet","l1_ratio":0.0}'

python3 benchmark.py run --model=SGDRegressor --params='{"max_iter":10,"penalty":"elasticnet","l1_ratio":1.0}'
python3 benchmark.py run --model=SGDRegressor --params='{"max_iter":100,"penalty":"elasticnet","l1_ratio":1.0}'
python3 benchmark.py run --model=SGDRegressor --params='{"max_iter":1000,"penalty":"elasticnet","l1_ratio":1.0}'

python3 benchmark.py run --model=SGDRegressor --params='{"max_iter":250,"penalty":"elasticnet","alpha":1e-0}'
python3 benchmark.py run --model=SGDRegressor --params='{"max_iter":250,"penalty":"elasticnet","alpha":1e-1}'
python3 benchmark.py run --model=SGDRegressor --params='{"max_iter":250,"penalty":"elasticnet","alpha":1e-2}'

python3 benchmark.py run --model=SGDClassifier --params='{"max_iter":10,"penalty":"elasticnet","l1_ratio":0.0}'
python3 benchmark.py run --model=SGDClassifier --params='{"max_iter":100,"penalty":"elasticnet","l1_ratio":0.0}'
python3 benchmark.py run --model=SGDClassifier --params='{"max_iter":1000,"penalty":"elasticnet","l1_ratio":0.0}'

python3 benchmark.py run --model=SGDClassifier --params='{"max_iter":10,"penalty":"elasticnet","l1_ratio":1.0}'
python3 benchmark.py run --model=SGDClassifier --params='{"max_iter":100,"penalty":"elasticnet","l1_ratio":1.0}'
python3 benchmark.py run --model=SGDClassifier --params='{"max_iter":1000,"penalty":"elasticnet","l1_ratio":1.0}'

python3 benchmark.py run --model=SGDClassifier --params='{"max_iter":250,"penalty":"elasticnet","alpha":1e-0}'
python3 benchmark.py run --model=SGDClassifier --params='{"max_iter":250,"penalty":"elasticnet","alpha":1e-1}'
python3 benchmark.py run --model=SGDClassifier --params='{"max_iter":250,"penalty":"elasticnet","alpha":1e-2}'

# Decision trees