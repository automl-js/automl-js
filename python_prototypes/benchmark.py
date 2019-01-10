"""
A script to benchmark javascript implementations against
those in sklearn. As performance metrics, test scores of
models are taken, as well as their runtime.

Usage:
  benchmark.py run --model=<class> --params=<json> [--name=<name>] [--every=<Nth>] [--max_size=<size>]
  benchmark.py score --name=<name>
  benchmark.py -h | --help
  benchmark.py --version

Options:
  -h --help             Show this screen.
  --version             Show version.
  --model=<class>       Name of the model class to test.
  --params=<json>       Parameter configuration to test. Should
                        be a valid json string, importantly, with
                        no spaces.
  --name=<name>         Name of the experiment. [default: results]
  --every=<Nth>         Use only every n-th dataset for testing.
                        Useful during development, to speed up the
                        tests. [default: 1]
  --max_size=<size>     Maximum number of samples to use for training
                        of the models. Is used to speed up training during
                        development of algorithms. [default: 1000]
"""

from subprocess import call, DEVNULL
import numpy as np
import pmlb
import json
from time import time
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.pipeline import make_pipeline
from tqdm import tqdm
from itertools import product
from pprint import pprint
from joblib import Parallel, delayed
import pandas as pd


def rnd_name():
    return ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(10))


def compare_estimator(X, y, X_test, y_test, estimator, params):
    rnd = rnd_name()
    result_name = 'results_' + rnd + '.json'
    script_name = 'jstrain_' + rnd + '.js'
    datajs_name = 'dataset_' + rnd + '.json'

    node_code = """
const ai = require('../src/aitable')
const fs = require('fs')
var data = JSON.parse(fs.readFileSync('%s', 'utf8'));

async function main(){
    var X = data['X'];
    var y = data['y'];
    var X_test = data['X_test'];
    var y_test = data['y_test'];

    // read estimator from the serialization module
    var model = new ai.io.base_estimators['%s'](%s)

    var fit_start = process.hrtime();
    await model.fit(X, y)
    var elapsed = process.hrtime(fit_start)[1] / 1000000; // divide by a million to get nano to milli

    var res = {
        'y_pred': await model.predict(X_test), 'runtime': elapsed
    }
    await fs.writeFile('%s', JSON.stringify(res), 'utf8', function(){ })
}

main()
    """ % (
        datajs_name,
        estimator.__class__.__name__,
        json.dumps(params),
        result_name
    )

    with open(script_name, 'w') as s:
        s.write(node_code)

    with open(datajs_name, 'w') as d:
        json.dump({
            'X': X.tolist(),
            'y': y.tolist(),
            'X_test': X_test.tolist(),
            'y_test': y_test.tolist(),
        }, d)

    call(['node '+script_name], shell=True, stdout=DEVNULL, stderr=DEVNULL)

    with open(result_name, 'r') as js:
        javascript = json.load(js)

    estimator.set_params(**params)

    start = time()
    estimator.fit(X, y)
    elapsed = (time() - start) * 1000.0  # miliseconds

    # clean up
    os.remove(script_name)
    os.remove(result_name)
    os.remove(datajs_name)

    from sklearn.metrics import accuracy_score, r2_score

    metric = accuracy_score if estimator._estimator_type == "classifier" else r2_score

    return {
        'python_score': estimator.score(X_test, y_test),
        'python_runtime': elapsed,
        'javascript_score': metric(y_test, javascript['y_pred']),
        'javascript_runtime': javascript['runtime']
    }

def test_on_dataset(config, dataset, max_size=1000):
    cname = config['--model']
    params = config['--params']
    params = json.loads(params)

    X, y = pmlb.fetch_data(dataset, True)
    print(dataset, X.shape)

    if len(y) > max_size:
        X = X[:max_size]
        y = y[:max_size]

    if (len(set(y)) < 2):
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    feats = make_pipeline(
        Imputer(),
        StandardScaler()
    )
    X_train = feats.fit_transform(X_train)
    X_test = feats.transform(X_test)

    estimator = globals()[cname]()

    result = compare_estimator(X_train, y_train, X_test, y_test, estimator, params)
    result['dataset'] = dataset
    result['dataset_shape'] = str(X.shape)
    result['params'] = config['--params']
    result['model'] = cname

    return result


def score_bench(args):
    fpath = args['--name'] + '.csv'
    df = pd.read_csv(fpath)

    df = df.set_index('dataset')

    # avoid outliers, and weird results like 0.1 / -0.1
    touse = (df['javascript_score'] > 0.1) & (df['python_score'] > 0.1)
    sc = df[touse]

    print('JS / PY score ratio: %2.3f' % (sc['javascript_score'] / sc['python_score']).mean())
    print('Average scores:')
    print((df[['javascript_score', 'python_score']]).mean())
    print('JS / PY runtime ratio: %2.2f' % (df['javascript_runtime'] / df['python_runtime']).mean())

    # find most different scores
    print('Datasets with most different scores, js_score - py_score:')
    diff = df['javascript_score'] - df['python_score']
    diff = diff.sort_values()

    print(diff[:3])
    print(diff[-3:])



def run_bench(args):
    every = int(args['--every'])
    max_size = int(args['--max_size'])

    if 'Classifier' in args['--model']:
        datasets = pmlb.classification_dataset_names
    else:
        datasets = pmlb.regression_dataset_names

    datasets = datasets[::every]

    jobs = [delayed(test_on_dataset)(args, d, max_size) for d in datasets]
    results = Parallel(n_jobs=-1)(jobs)
    results = [r for r in results if r is not None]

    df = pd.DataFrame(results)

    # append to existing results
    fpath = args['--name'] + '.csv'
    df.to_csv(fpath, mode='a', header=not os.path.exists(fpath), index=False)

    score_bench(args)


if __name__ == '__main__':
    from docopt import docopt
    args = docopt(__doc__, version='1.0')

    if args['run']:
        run_bench(args)
    if args['score']:
        score_bench(args)