const io = require('./io')
const utils = require('./utils')
const metrics = require('./metrics')

/**
 * Base class for all transformers, estimators etc.
 */
class BaseEstimator{

    /**
     * Construct an estimator.
     * @param {Object} params parameters of estimator
     * @param {Object} defaults default values of the
     * parameters
     */
    constructor(params, defaults){
        this.params = utils.set_defaults(params, defaults)
        this.state = {}
    }

    /**
     * Return the parameters of the estimator.
     * @param {Boolean} deep Whether to create an independent
     * copy of the parameters.
     */
    async get_params(deep=true){
        var result = this.params
        if(deep){
            result = io.clone(result)
        }
        return result
    }

    /**
     * Set parameters of the estimator.
     * @param {dictionary} params parameter values with names specified
     * as keys.
     */
    async set_params(params){
        for(var key in params){
            this.params[key] = params[key]
        }
        return this
    }
}

module.exports.BaseEstimator = BaseEstimator

/**
 * Mixin class for all classifiers.
 * @param {class} superclass Class to extend with functionality
 */
module.exports.ClassifierMixin = (superclass)=>class extends superclass{
    /**
     * Returns the mean accuracy on the given test data and labels.
     * In multi-label classification, this is the subset accuracy
     * which is a harsh metric since you require for each sample that
     *  each label set be correctly predicted.
     * @param {Array} X Test samples.
     * @param {Array} y Test labels for X.
     * @returns {Number} Mean accuracy of this.predict(X) wrt. y.
     */
    async score(X, y){
        var y_pred = await this.predict(X)
        return metrics.accuracy_score(y, y_pred)
    }
}

/**
 * Mixin class for all regressors.
 * @param {class} superclass Class to extend with functionality
 */
module.exports.RegressorMixin = (superclass)=>class extends superclass{
    /**
     * Returns the coefficient of determination R^2 of the prediction.
     * The coefficient R^2 is defined as (1 - u/v), where u is the residual
     * sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
     * sum of squares ((y_true - y_true.mean()) ** 2).sum().
     * The best possible score is 1.0 and it can be negative (because the
     * model can be arbitrarily worse). A constant model that always
     * predicts the expected value of y, disregarding the input features,
     * would get a R^2 score of 0.0.
     * @param {Array} X Test samples.
     * @param {Array} y Test outputs for X.
     * @returns {Number} R^2  of this.predict(X) wrt. y.
     */
    async score(X, y){
        var y_pred = await this.predict(X)
        return metrics.r2_score(y, y_pred)
    }
}