const tf = require('@tensorflow/tfjs');

/**
 * Converts some TypedArray into 1d tfjs array.
 * @param {Array} data data to be converted to 1d tfjs array
 */
function t1d(data){
    if(data instanceof tf.Tensor){
        if(data.shape.length == 1){
            return data
        }
        if(data.shape.length == 2){
            return tf.reshape(data, [data.shape[0]])
        }
    }
    return tf.tensor1d(data)
}

module.exports.t1d = t1d

/**
 * Converts some TypedArray into 2d tfjs array.
 * @param {Array} data data to be converted to 2d tfjs array
 */
function t2d(data){
    if(data instanceof tf.Tensor){
        if(data.shape.length == 2){
            return data
        }
    }
    return tf.tensor2d(data)
}

module.exports.t2d = t2d

/**
 * Checks if the estimator is fitted by verifying the presence of 
 * “all_or_any” of the passed attributes and raises an Error
 *  with the given message.
 * @param {Object} estimator estimator to check to be fitted.
 * @param {Array} attributes list of attributes to check presence of.
 * @param {String} all_or_any Specify whether all or any of the 
 * given attributes must exist.
 */
function check_is_fitted(estimator, attributes, all_or_any='all'){
    var check = []
    for(var k of attributes){
        check.push(k in estimator.state)
    }
    var is_fitted = all_or_any == 'all'
    for(var c of check){
        if(all_or_any == 'all'){
            is_fitted = is_fitted && c
        }else{
            is_fitted = is_fitted || c
        }
    }

    if(!is_fitted){
        throw Error(
            "Estimator " + estimator.constructor.name + " is not fitted. Please " +
            "call .fit(X, y) method, with proper inputs X and outputs y."
        )
    }

    return true
}

module.exports.check_is_fitted = check_is_fitted

/**
 * Sets default parameter values for the model classes.
 * @param {Dictionary} params Parameters supplied by user
 * @param {Dictionary} defaults Default values of parameters
 */
function set_defaults(params=null, defaults=null){
    var result = {}
    defaults = defaults || {}
    for(var key in defaults){
        result[key] = defaults[key]
    }
    params = params || {}
    for(var key in params){
        result[key] = params[key]
    }
    return result
}

module.exports.set_defaults = set_defaults

/**
 * Check dimensions of the array.
 * @param {Array} array Array to check, and optionally convert to tf tensor.
 * @param {Number} min_nd minimum number of the dimensions in the array.
 * @param {Number} max_nd maximum number of dimensions.
 * @param {Array} min_shape array of maximal sizes for dimensions of the tensor.
 * @param {Array} max_shape maximal sizes of dimensions in array.
 * @param {Boolean} make_tf whether to convert the input array to tf tensor.
 */
function check_array(array, min_nd=2, max_nd=2, min_shape=[1, 1], max_shape=null, make_tf=true){
    var shape = null
    if(array.constructor.name === "Tensor"){
        shape = array.shape
    }else{
        // recursively get the size of array
        // ToDo: check the consistent size of all feature vectors
        var ap = array
        var shape = []
        while(ap instanceof Array){
            shape.push(ap.length)
            ap = ap[0]
        }
    }

    var n_dim = shape.length

    if(max_nd !== null && n_dim > max_nd){
        throw Error(n_dim + ' is too many dimensions in the input data: ' + array)
    }

    if(min_nd !== null && n_dim < min_nd){
        throw Error(n_dim + ' is too little dimensions in the input data: ' + array)
    }
    
    for(var i=0; i<n_dim; i++){
        if(min_shape !== null && i<min_shape.length && shape[i] < min_shape[i]){
            throw Error(shape[i] + ' is too small size of dimension ' + i + ' of input ' + array)
        }
        if(max_shape !== null && i<max_shape.length && shape[i] > max_shape[i]){
            throw Error(shape[i] + ' is too large size of dimension ' + i + ' of input ' + array)
        }
    }

    if(make_tf){
        return tf.tensor(array)
    }
    return array
}

module.exports.check_array = check_array

/**
 * Check if inputs X are of proper format.
 * @param {Array} X Input samples.
 * @param {Boolean} make_tf Whether to convert input samples to tf.tensor
 */
function check_2dX(X, make_tf=true){
    return check_array(X, 2, 2, [1, 1], null, make_tf)
}

module.exports.check_2dX = check_2dX

/**
 * Check if outputs y are of proper format.
 * @param {Array} y Outputs.
 * @param {Boolean} make_tf Whether to convert y to tf.tensor
 */
function check_1dy(y, make_tf=true){
    return check_array(y, 1, 1, [1], null, make_tf)
}

/**
 * Check whether a dataset is consistent, and whether inputs
 * are a matrix and outputs are a vector.
 * @param {Array} X Array of input observations.
 * @param {Array} y Outputs.
 * @param {Boolean} make_tf Whether to convert y to tf.tensor
 */
function check_2dXy(X, y, make_tf=true){
    var Nx = null
    var Ny = null
    // check consistent size of arrays
    if(X.constructor.name === 'Tensor'){
        Nx = X.shape[0]
    }else{
        Nx = X.length
    }
    if(y.constructor.name === 'Tensor'){
        Ny = y.shape[0]
    }else{
        Ny = y.length
    }

    if(Nx != Ny){
        throw Error('Inconsistent size of samples provided')
    }
    
    X = check_2dX(X, make_tf)
    y = check_1dy(y, make_tf)
    
    return [X, y]
}

module.exports.check_2dXy = check_2dXy

/**
 * Shuffles array in place.
 * @param {Array} a items An array containing the items.
 */
function shuffle(a) {
    var j, x, i;
    for (i = a.length - 1; i > 0; i--) {
        j = Math.floor(Math.random() * (i + 1));
        x = a[i];
        a[i] = a[j];
        a[j] = x;
    }
    return a;
}

/**
 * Generate index set for partitioning of array of size N into
 * a number of partitions. The size of partitions can be specified
 * with `partitions` argument. The function ensures that the minimal
 * size of every partition is at least one. This is important for
 * small sized data partitioning, for example, or in case of leave one
 * out cross - validation.
 * @param {Integer} N Size of array to be partitioned
 * @param {Array} weights Weights that are assigned to partitions. 
 * With higher weight, the partition is more likely to receive elements.
 * @returns {Array} Every element in the output array is a set of indicies
 * of the elements that belong to the i-th partition.
 */
function random_partitioning(N, weights){
    if(N < weights.length){
        throw new Error('Cannot split the array in desired manner. N is smaller than partitions.')
    }

    // make up random indicies
    var I = []
    for(var i=0; i<N; i++){
        I.push(i)
    }
    shuffle(I)

    var ix = 0
    // make up random partitions, necessary non - empty
    var P = []
    for(var w of weights){
        P.push([I[ix++]])
    }

    // divide whats left according to the weights provided
    N_left = N - weights.length

    // ensure that everything sums to 1.0
    var psum = weights.reduce((s, c)=>{return s+c}, 0.0)
    weights = weights.map((v)=>{return v / psum})

    // distribute elements that are left among all of the partitions
    for(var i=0; i<weights.length; i++){
        var p = P[i]
        var N_part = Math.round(weights[i] * N_left)
        
        if(i === weights.length-1){
            N_part = N  // use up all the elements in the last section
        }

        for(var j=0; j<N_part; j++){
            if(ix >= I.length){
                break
            }
            p.push(I[ix++])
        }
    }

    return P
}

module.exports.random_partitioning = random_partitioning