const tf = require('@tensorflow/tfjs');
const utils = require('./utils')

/**
 * R^2 (coefficient of determination) regression score function.
 * Best possible score is 1.0 and it can be negative (because the
 * model can be arbitrarily worse). A constant model that always
 * predicts the expected value of y, disregarding the input features,
 * would get a R^2 score of 0.0.
 * @param {Array} y_true array-like of shape = (n_samples); Ground truth (correct) target values. 
 * @param {Array} y_pred array-like of shape = (n_samples) or (n_samples, n_outputs); Estimated target values.
 * @param {Array} sample_weight array-like of shape = (n_samples), optional; Sample weights.
 */
function r2_score(y_true, y_pred){
    var y_pred = utils.t1d(y_pred)
    var y_true = utils.t1d(y_true)
    
    var mean = tf.mean(y_true) // can be seen as predictions of a constant model
    
    // calculate errors of trivial model
    var base_errors = tf.sub(y_true, mean)
    base_errors = tf.sum(tf.pow(base_errors, 2))
    base_errors = base_errors.get()

    var model_errors = tf.sub(y_true, y_pred)
    model_errors = tf.sum(tf.pow(model_errors, 2))
    model_errors = model_errors.get()

    if(base_errors === 0.0){
        return 0.0
    }

    return 1.0 - model_errors / base_errors
}

module.exports.r2_score = r2_score

/**
 * In multilabel classification, this function computes subset accuracy:
 * the set of labels predicted for a sample must *exactly* match the
 * corresponding set of labels in y_true.
 * @param {Array} y_true 1d array-like, or label indicator array; Ground truth (correct) labels
 * @param {Array} y_pred 1d array-like, or label indicator array; Predicted labels, as returned by a classifier.
 */
function accuracy_score(y_true, y_pred){
    var total_accurate = 0
    for(var i = 0; i<y_true.length; i++){
        if(y_true[i] === y_pred[i]){
            total_accurate += 1
        }
    }
    var fraction = total_accurate / y_pred.length
    return fraction
}

module.exports.accuracy_score = accuracy_score
