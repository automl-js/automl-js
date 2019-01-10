# automl-js

Automated Machine Learning, done locally in browser or on a server with nodejs. Ground up implementation of ML algorithms for both regression and classification, such as Decision Trees, Linear Models and Gradient Boosting with Decision Trees. The implementation is benchmarked against excellent `scikit-learn` library to give quite close, albeit somewhat smaller (at most 1 percent of classification accuracy on average) score.

# Installation

The code should be used in browser using standard script import:

```html
<script src="./dist/automljs.js"></script>
```

This creates a global `aml` object, that can be used to instantiate the models, do data splitting and preprocessing, etc. If you wish to run it in nodejs, install the package with npm and import using `require`.

# Docs and examples

Below this section are two examples for how to use code in automl-js. Beyond this, see docs at [https://automl-js.github.io](https://automl-js.github.io) for description of objects and functions of automljs, and `tests` folder for example usage of functionality.

# Example automl estimator

```javascript
// automl-js uses asynchronous functionality of JavaScript
async function main(){
    // Each row is an observation, each column is a feature (similar to numpy)
    // Mixed types: categorical and numerical, missing values are handled automatically
    var X = [
        ['a', 0.6],
        ['b', -1.3],
        ['a', 1.1],
        ['b', -2],
        ['a', 0.5],
        ['b', ""],  // missing value
        ['a', 0.4],
        ['', 1.1],  // missing value
        ['b', -0.8],
        ['a', "1e-1"]  // wrong type
    ]

    // Outputs are a vector / 1d array
    var y = ['pos', 'neg', 'pos', 'neg', 'neg', 'pos', 'neg', 'pos', 'neg', 'pos']

    // Create a model instance
    var model = new aml.automl.AutoMLModel({
        'max_iter': 7  // specifies how many iterations you are willing to wait for result
    })

    // Does internally data splitting into training and testing
    // Tries a bunch of ML models to see which one fits best
    await model.fit(X, y)

    // Evaluate the best found model
    var score = await model.score(X, y)

    // Get estimations by the model; Interface same as sklearn
    var y_pred = await model.predict(X)
}

// run the async function
main()
```

# Example learning with estimator

The code should be run in browser. If you wish to run it in nodejs, install the package with npm and import using `require`.

```html
<script src="./dist/automljs.js"></script>

<script>
    async function main(){
        // Data needs to be of numeric type for regular estimators
        var X = [
            [1, 0],
            [0, 1],
            [1, 1],
            [0, 0]
        ]

        // Outputs are a vector / 1d array
        var y = [1, 2, 2, 0]

        // Create a model instance; Names of parameters are mostly similar to sklearn
        var model = new aml.ensemble.GradientBoostingRegressor({'n_estimators':10, 'learning_rate': 1.0, 'loss': 'ls'})

        // Fit the model
        await model.fit(X, y)

        // Evaluate the model
        var score = await model.score(X, y)

        // Get estimations by the model; Interface same as sklearn
        var y_pred = await model.predict(X)
    }

    main()
</script>
```

