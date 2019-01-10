const expect = require('chai').expect;
const ai = require('../src/automljs')
const tu = require('./testutils')

describe('linear_model', async function(){
    it('fits, scores, clones SGDRegressor', async function(){
        await tu.test_estimator_behavior(
            ai.linear_model.SGDRegressor,
            {'max_iter':100, 'eta0':0.1},
            'model',
            0.7
        )

        var model = new ai.linear_model.SGDRegressor()
        await model.fit([
            [0, 1, 2],
            [2, 1, 0],
        ], [0, 1])
        model.to_tabulator()
        model.to_tabulator([
            {name: 'A', type: 'number'},
            {name: 'B', type: 'boolean'},
            {name: 'C', type: 'number'},
        ])
    })
    it('fits, scores, clones SGDClassifier', async function(){
        await tu.test_estimator_behavior(
            ai.linear_model.SGDClassifier,
            {'max_iter':100, 'eta0':0.1},
            'model',
            0.7
        )
    })
})