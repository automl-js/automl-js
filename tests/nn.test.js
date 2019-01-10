const expect = require('chai').expect;
const ai = require('../src/automljs')
const tu = require('./testutils')

describe('nn', async function(){
    it('fits, scores, clones MLPRegressor', async function(){
        var X = [
            [1, 0],
            [0, 1],
            [1, 1],
            [0, 0]
        ]
        var y = [1, 1, 2, 0]
        var model = new ai.nn.MLPRegressor({
            'n_neurons': 10,
            'epochs': 10,
            'lr': 0.1
        })
        await model.fit(X, y)
        var score = await model.score(X, y)
        // prevent catastrophic failure
        expect(score).to.be.greaterThan(0.5)

        await tu.test_serialization_estim(model, X, y, 'model')
    })
    it('fits, scores, clones MLPClassifier', async function(){
        var X = [
            [1, 0],
            [0, 1],
            [1, 1],
            [0, 0]
        ]
        var y = ["1", "1", "0", "0"]
        var model = new ai.nn.MLPClassifier({
            'n_neurons': 10,
            'epochs': 10,
            'lr': 0.1
        })
        await model.fit(X, y)
        var score = await model.score(X, y)
        // prevent catastrophic failure
        expect(score).to.be.greaterThan(0.5)
        await tu.test_serialization_estim(model, X, y, 'model')
    })
    
})