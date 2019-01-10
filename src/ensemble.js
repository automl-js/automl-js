const base = require('./base')
const RegressorMixin = base.RegressorMixin
const BaseEstimator = base.BaseEstimator
const utils = require('./utils')
const tf = require('@tensorflow/tfjs')
const prep = require('./preprocessing')
const tree = require('./tree')

/**
 * Gradient boosting model, for regression problems.
 * As a weak learner, decision tree is used. Inspired by:
 * https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
 */
class GradientBoostingRegressor extends RegressorMixin(BaseEstimator){
    /**
     * Instantiates GradientBoostingRegressor.
     * @param {Object} params Dictionary with parameters for
     * the model.
     * @param {String} [params.loss] Type of loss to use. Currently
     * available is 'squared_loss'.
     * @param {Number} [param.learning_rate] How much does a single
     * weak learner affect the final ensemble. Smaller values might
     * require larger number of estimators for satisfactory behavior,
     * but usually lead to better generalization.
     * @param {Number} [param.n_estimators] How many weak learners
     * to use in final ensemble.
     * @param {Number} [param.min_samples_split] Hyperparameter of
     * the underlying weak learner. Minimum number of samples in the
     * decision tree to form a split. Can either be a value in range
     * [0.0, 1.0], which indicates the fraction of overall dataset.
     * Alternatively, it could be a value greater equal 2, in which 
     * case specific number of samples is specified.
     * @param {Number} [param.max_depth] Maximum depth of the decision
     * tree weak learner.
     * @param {Number} [param.min_impurity_decrease] Minimal decrease
     * of impurity value, for which to form a new split. Used in the
     * training heuristic of the tree weak learner. Recommended to leave
     * as 0.0.
     */
    constructor(params){
        super(params, {
            'loss': 'squared_loss',
            'learning_rate': 0.1,
            'n_estimators': 100,
            'min_samples_split': 2,
            'max_depth': 3,
            'min_impurity_decrease': 0.0
        })
    }

    async fit(X, y){
        var [X, y] = utils.check_2dXy(X, y, false)

        var loss = this.params['loss']
        if(loss === 'ls'){
            loss = 'squared_loss'
        }
        var loss_id = {
            'squared_loss': 0,
            'hinge': 1
        }[loss]
        var learning_rate = this.params['learning_rate']
        var n_estimators = this.params['n_estimators']
        var min_samples_split = this.params['min_samples_split']
        var max_depth = this.params['max_depth']
        var min_impurity_decrease = this.params['min_impurity_decrease']

        // calculate initial value
        var init_ = 0.0
        for(var i=0; i<y.length; i++){
            init_ += y[i]
        }
        init_ /= y.length

        // predictions of the model, and residuals
        var y_pred = []
        var rm = []
        for(var i=0; i<y.length; i++){
            y_pred.push(init_)
            rm.push(0.0)
        }

        var estimators = []

        var aux0 = 0.0

        // main fitting loop
        for(var i=0; i<n_estimators; i++){
            // calculate the residuals - negative gradients of loss
            // w.r.t. the estimator score.
            for(var j=0; j<y_pred.length; j++){
                if(loss_id == 0){ // sq. loss
                    rm[j] = -2*(y[j] - y_pred[j])
                }else if(loss_id == 1){ // hinge
                    aux0 = 1.0 - y[j] * y_pred[j]
                    rm[j] = aux0 > 0.0? -y[j] : 0.0
                }
                rm[j] = -rm[j]
            }

            // fit the regression tree
            var estimator = new tree.DecisionTreeRegressor({
                'max_depth': max_depth,
                'min_impurity_decrease': min_impurity_decrease,
                'min_samples_split': min_samples_split
            })

            // learn the gradients
            await estimator.fit(X, rm)

            // estimate the gradients
            var h = await estimator.predict(X)
            var gamma_base = 0.0
            var step = 1.0

            var total_iter = 64

            // find optimal gamma; assumes convex loss w.r.t. gamma!
            while(total_iter > 0){   
                // calculate the total gradient w.r.t. gamma
                var grad = 0.0
                var gamma = gamma_base + step
                for(var j=0; j<h.length; j++){
                    // should calculate losss of L(y, f(x) + gamma*h(x)) w.r.t gamma
                    if(loss_id === 0){ // sq. loss
                        grad += -2*h[j]*(y[j] - y_pred[j] - gamma*h[j])
                    }else if(loss_id === 1){ // hinge loss
                        // d max(0, 1-y(f+h)) / dh = I(1-y(f+gh)) * (-yh)
                        aux0 = 1.0 - y[j]*(y_pred[j] + gamma*h[j]) 
                        grad += aux0 > 0.0 ? -y[j]*h[j] : 0.0
                    }
                }
                
                // find the smallest globally optimal gamma
                // prior: if equivalent, select smaller gamma
                // expectation of stability: smaller learning 
                // rates lead to more stable gradient descent
                if(grad >= 0.0){
                    step /= 2.0
                }else{
                    gamma_base += step
                }
                total_iter -= 1
            }

            var gamma = gamma_base + 0.5*step

            // update the predictions
            for(var j=0; j<y_pred.length; j++){
                y_pred[j] = y_pred[j] + learning_rate * gamma * h[j]
            }

            // add estimator
            estimators.push([estimator, gamma])

        }

        this.state['estimators_'] = estimators
        this.state['init_'] = init_
        
        return this // sklearn convention
    }

    async predict(X){
        utils.check_is_fitted(this, ['estimators_', 'init_'])
        var X = utils.check_2dX(X, false)

        var estimators = this.state['estimators_']
        var init_ = this.state['init_']
        var learning_rate = this.params['learning_rate']

        // initialize with init_
        var y_pred = []
        for(var x of X){
            y_pred.push(init_)
        }

        for(var eg of estimators){
            var [estim, gamma] = eg
            var h = await estim.predict(X)

            for(var j=0; j<y_pred.length; j++){
                y_pred[j] += learning_rate * gamma * h[j]
            }
        }

        return y_pred
    }
}

module.exports.GradientBoostingRegressor = GradientBoostingRegressor


class GradientBoostingClassifier extends base.ClassifierMixin(BaseEstimator){
    /**
     * Instantiates GradientBoostingClassifier.
     * @param {Object} params Dictionary with parameters for
     * the model.
     * @param {String} [params.loss] Type of loss to use. Currently
     * available is 'hinge'.
     * @param {Number} [param.learning_rate] How much does a single
     * weak learner affect the final ensemble. Smaller values might
     * require larger number of estimators for satisfactory behavior,
     * but usually lead to better generalization.
     * @param {Number} [param.n_estimators] How many weak learners
     * to use in final ensemble.
     * @param {Number} [param.min_samples_split] Hyperparameter of
     * the underlying weak learner. Minimum number of samples in the
     * decision tree to form a split. Can either be a value in range
     * [0.0, 1.0], which indicates the fraction of overall dataset.
     * Alternatively, it could be a value greater equal 2, in which 
     * case specific number of samples is specified.
     * @param {Number} [param.max_depth] Maximum depth of the decision
     * tree weak learner.
     * @param {Number} [param.min_impurity_decrease] Minimal decrease
     * of impurity value, for which to form a new split. Used in the
     * training heuristic of the tree weak learner. Recommended to leave
     * as 0.0.
     */
    constructor(params){
        super(params, {
            'loss': 'hinge',
            'learning_rate': 0.1,
            'n_estimators': 100,
            'min_samples_split': 2,
            'max_depth': 3,
            'min_impurity_decrease': 0.0
        })
    }

    async fit(X, y){
        var [X, y] = utils.check_2dXy(X, y, false)
        var binarizer = new prep.LabelBinarizer()
        binarizer.fit(y)
        var y = binarizer.transform(y).mul(2).add(-1)

        var n_models = y.shape[1]

        var models = []
        for(var yi=0; yi<n_models; yi++){
            var y_model = y.slice([0, yi], [y.shape[0], 1])
            y_model = Array.from(await y_model.data())
            var model = new GradientBoostingRegressor(this.params)
            await model.fit(X, y_model)
            models.push(model)
        }

        this.state['models'] = models
        this.state['binarizer'] = binarizer
        return this
    }

    async predict(X){
        utils.check_is_fitted(this, ['models', 'binarizer'])
        var X = utils.check_2dX(X, false)

        var y_models = []
        for(var model of this.state['models']){
            y_models.push(await model.predict(X))
        }

        var y_pred = []
        var inverse = this.state['binarizer'].state['inverse']

        for(var i=0; i<X.length; i++){
            var max_score = Number.NEGATIVE_INFINITY;
            var max_j = 0
            for(var j=0; j<y_models.length; j++){
                if(y_models[j][i] > max_score){
                    max_score = y_models[j][i]
                    max_j = j
                }
            }
            y_pred.push(inverse[max_j])
        }

        return y_pred
    }

}

module.exports.GradientBoostingClassifier = GradientBoostingClassifier