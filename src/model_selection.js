const utils = require('./utils')
const base = require('./base')
const prep = require('./preprocessing')
const io = require('./io')
const opt = require('optimization-js')

/**
 * Splits the data into training and testing partitions for model 
 * fitting. The split is done at random.
 * @param {Array} arrays List of arrays to be split into training
 * and testing parts.
 * @param {Number} test_split Value in the range of [0; 1]. Specifies
 * how much of the data will be used for test partition.
 */
function train_test_split(arrays, test_split=0.25){
    var N = arrays[0].length
    var partition = utils.random_partitioning(N, [1.0 - test_split, test_split])
    function train_test_split(arrays, test_split=0.25){
        var N = arrays[0].length
        var partition = utils.random_partitioning(N, [1.0 - test_split, test_split])
    
        var result = []
    
        // for arrays = [X, y], the function should return
        // [X_train, X_test, y_train, y_test]
        for(var arr of arrays){
            for(var I of partition){
                var part = I.map((ix)=>arr[ix])
                result.push(part)
            }
        }
    
        return result
    }
    
    module.exports.train_test_split = train_test_split
    
    var result = []

    // for arrays = [X, y], the function should return
    // [X_train, X_test, y_train, y_test]
    for(var arr of arrays){
        for(var I of partition){
            var part = I.map((ix)=>arr[ix])
            result.push(part)
        }
    }

    return result
}

module.exports.train_test_split = train_test_split

/**
 * Evaluate estimator performance via cross - validation.
 * Interface similar to `cross_val_score` function of sklearn.
 * @param {BaseEstimator} estimator Model to be scored.
 * @param {Array} X Input values to be used for sorting.
 * @param {Array} y Output values to be used for scoring
 * @param {Object} groups Ignored.
 * @param {Object} scoring Ignored for the time being.
 * @param {Number} cv Number of folds used for cross - validation.
 * @returns {Array} Scores on various partitions.
 */
async function cross_val_score(estimator, X, y=null, groups=null, scoring=null, cv=3){
    var length = []

    for(var i=0; i<cv; i++){
        length.push(1.0) // all partitions are made equal at this point
    }

    var partitions = utils.random_partitioning(X.length, length)
    var scores = []

    for(var fold=0; fold<cv; fold++){
        var p = partitions[fold]
        var test_set = {}
        
        // setify fold
        for(var i=0; i<p.length; i++){
            test_set[p[i]] = true
        }

        // sort into training and testing
        var [X_train, y_train, X_test, y_test] = [[], [], [], []]

        for(var i=0; i<X.length; i++){
            if(i in test_set){
                X_test.push(X[i])
                y_test.push(y[i])
            }else{
                X_train.push(X[i])
                y_train.push(y[i])
            }
        }

        // evaluate on the model
        var fold_estim = await io.clone(estimator, false)
        await fold_estim.fit(X_train, y_train)
        var score = await fold_estim.score(X_test, y_test)
        scores.push(score)
    }

    return scores
}

module.exports.cross_val_score = cross_val_score


/**
 * Optimizes hyperparameters for supplied estimator.
 * A very simple version of genetic algorithm is used
 * for this otpimization. See more in optimization-js,
 * in particular in OMGOptimizer docs.
 */
class OMGSearchCV extends base.BaseEstimator{
    /**
     * Creates an instance of OMGSearchCV.
     * @param {Object} params Parameters of the hyperparameter
     * optimization algorithm.
     * @param {BaseEstimator} [params.estimator] Estimator instance,
     * whose hyperparameters are optimized.
     * @param {Object} [params.param_grid] Specification of the search
     * space, over which to optimize the estimator parameters. The 
     * specification should be given in the following format:
     * name_of_parameter: optimization-js dimension object instance
     * Here dimension object should be one of Integer, Real or 
     * Categorical class instances from optimization-js package.
     * @param {Number} [params.max_iter] Maximum number of iterations, 
     * allowed for the algorithm  to execute. The algorithm could
     * be terminated prematurely via callbacks.
     * @param {Number} [params.cv] Number of cross validation folds 
     * that are used for evaluation of particular hyperparameter 
     * configuration.
     * @param {Boolean} [params.refit_on_improvement] Whether to fit
     * the final model on the whole dataset as soon as improvement in 
     * hyperparameter configuration is obtained, w.r.t. CV score.
     * @param {Boolean} [params.refit] Whether to fit the final model
     * after the hyperparameter search has concluded.
     */
    constructor(params){
        super(params, {
            'estimator': null,
            'param_grid': null,
            'max_iter': 100,
            'cv': 3,
            'refit_on_improvement': false,
            'refit': true
        })
    }

    /**
     * Gets the vector of dimensions that represent
     * a search space flattened to vector.
     */
    get_dim_vector(){
        var grid = this.params['param_grid']
        var grid_keys = []
        for(var key in grid){
            grid_keys.push(key)
        }
        grid_keys.sort()

        var dimensions = []

        for(var key of grid_keys){
            dimensions.push(grid[key])
        }

        return [dimensions, grid_keys]
    }

    /**
     * Initialize the search algorithm.
     */
    async init(X, y){
        var [dimensions, grid_keys] = this.get_dim_vector()

        // get dimensions from grid
        var optimizer = new opt.OMGOptimizer(dimensions)
        
        // temporary values
        this.grid_keys = grid_keys
        this.optimizer = optimizer
        this.X = X
        this.y = y
        this.improved = false

        this.state['best_params_'] = null
        this.state['best_score_'] = null
        this.state['best_estimator_'] = null
    }

    /**
     * Updates the ranges of hyperparameters to search over.
     * This is useful in case first the models with small
     * complexity are intended to be tried. Then in few initial
     * iterations, smaller parameters that control complexity 
     * should be given.
     * Note: only the ranges should be updated for dimensions.
     * Adding new dimensions will likely lead to unexpected behavior.
     * @param {Object} values Dictionary with keys being
     * names of the parameters, and values being dimensions
     * of the optimization-js type. 
     */
    update_ranges(values){
        for(var key in values){
            this.params['param_grid'][key] = values[key]
        }

        // create dimensions with updated ranges
        var [dimensions, grid_keys] = this.get_dim_vector()

        // create a new space
        var space = new opt.Space(dimensions)
        
        // update space in optimizer
        this.optimizer.space = space
    }

    /**
     * Run a single step of search algorithm.
     * @returns {Object} object that indicate result of a step.
     * Parameters of the object are as follows:
     * - {Boolean} improved whether there was an improvement in
     * score.
     */
    async step(){
        var optimizer = this.optimizer
        var grid_keys = this.grid_keys

        var p = optimizer.ask()
        var estim_params = {}

        // mix with parameters
        for(var i=0; i<grid_keys.length; i++){
            estim_params[grid_keys[i]] = p[i]
        }

        var estimator = this.params['estimator']
        var cv = this.params['cv']

        // set the current parameters
        await estimator.set_params(estim_params)
        
        // calculate CV score
        var cv_scores = await cross_val_score(estimator, this.X, this.y, null, null, cv)
        var avg_score = cv_scores.reduce((p, c)=>p+c) / cv
        var best_score_ = this.state['best_score_']

        if(best_score_ === null || best_score_ < avg_score){
            this.state['best_score_'] = avg_score
            this.state['best_params_'] = estim_params

            if(this.params['refit_on_improvement']){
                await this.fit_final(this.X, this.y)
            }

            return {"improved": true}
        }else{
            return {"improved": false}
        }
    }

    /**
     * Fits the model with set best parameters to the data.
     * @param {Array} X input samples
     * @param {Array} y output samples
     */
    async fit_final(X, y){
        var estimator = this.params['estimator']
        var best_params_ = this.params['best_params_']

        var estim = await io.clone(estimator)
        await estim.set_params(best_params_)
        await estim.fit(X, y)
        this.state['best_estimator_'] = estim
    }

    /**
     * Clear the temporarily stored variables
     * for the grid search.
     */
    async finish_search(){
        // fit the final model
        if(this.params['refit']){
            await this.fit_final(this.X, this.y)
        }

        // release the pointers to objectz
        this.X = null
        this.y = null
        this.optimizer = null
        this.grid_keys = null
    }

    async fit(X, y){
        await this.init(X, y)

        var n_iter = this.params['max_iter']
        for(var i=0; i<n_iter; i++){
            await this.step()
        }
        
        await this.finish_search()
    }

    async predict(X){
        utils.check_is_fitted(this, ['best_estimator_'])
        var estimator = this.state['best_estimator_']

        var y_pred = await estimator.predict(X)
        return y_pred
    }

    async score(X, y){
        utils.check_is_fitted(this, ['best_estimator_'])
        var estimator = this.state['best_estimator_']

        var score = await estimator.score(X, y)
        return score
    }
}

module.exports.OMGSearchCV = OMGSearchCV