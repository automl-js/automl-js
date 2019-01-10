const expect = require('chai').expect;
const ai = require('../src/automljs')
const tu = require('./testutils')

describe('ensemble', async function(){
    it('fits, scores, clones GradientBoostingRegressor', async function(){
        await tu.test_estimator_behavior(
            ai.ensemble.GradientBoostingRegressor,
            {'n_estimators':10, 'learning_rate': 1.0, 'loss': 'ls'},
            'model',
            0.7
        )
    })
    it('fits, scores, clones GradientBoostingClassifier', async function(){
        await tu.test_estimator_behavior(
            ai.ensemble.GradientBoostingClassifier,
            {'n_estimators':3, 'learning_rate': 1.0},
            'model',
            0.7
        )
    })
})