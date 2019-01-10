
class OMGSearchCV extends base.BaseEstimator{
    constructor(params){
        this.params = utils.set_defaults(params, {
            'estimator': null,
            'param_grid': null
        })
    }

    fit(X, y){

    }
}


