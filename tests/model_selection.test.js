var expect = require('chai').expect;
var ai = require('../src/automljs')
var opt = require('optimization-js')
const utils = require('./testutils')

describe('model_selection', async function(){
    it('splits data into training and testing', function(){
        var X = [[1], [2], [3], [4]]
        var y = [1, 2, 3, 4]

        var [Xt, Xe, yt, ye] = ai.model_selection.train_test_split([X, y])

        expect(Xt.length).to.be.equal(3)
        expect(Xe.length).to.be.equal(1)
        expect(yt.length).to.be.equal(3)
        expect(ye.length).to.be.equal(1)

        var [Xt, Xe, yt, ye] = ai.model_selection.train_test_split([X, y], 0.5)

        expect(Xt.length).to.be.equal(2)
        expect(Xe.length).to.be.equal(2)
        expect(yt.length).to.be.equal(2)
        expect(ye.length).to.be.equal(2)
    })
    it('cross - validates the estimators', async function (){
        var model = new ai.tree.DecisionTreeClassifier()
        var X = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ]
        var y = [0, 1, 0, 1, 0, 1, 0, 1]

        var scores = await ai.model_selection.cross_val_score(model, X, y, null, null, 3)

        expect(scores.length).to.be.equal(3)
        expect(scores).deep.equal([1.0, 1.0, 1.0])

        
    })
    it('OMGSearchCV works', async function (){
        var model = new ai.model_selection.OMGSearchCV({
            'estimator': new ai.tree.DecisionTreeClassifier(),
            'max_iter': 10,
            'refit_on_improvement':true,
            'param_grid': {
                'max_depth': new opt.Integer(1, 5),
                'min_impurity_decrease': new opt.Categorical([0.0, 0.0001]),
                'min_samples_split': new opt.Real(0.0001, 0.01)
            }
        })
        var X = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ]
        var y = [0, 1, 0, 1, 0, 1, 0, 1]
    
        await model.fit(X, y)
        
        // because it overfits, train. score will be 1.0
        expect(await model.score(X, y)).to.be.equal(1.0)
        expect(await model.predict(X)).deep.equal(y)

        await utils.test_serialization_estim(model, X, y)
    })
    it('OMGSearchCV works in manual mode', async function (){
        var model = new ai.model_selection.OMGSearchCV({
            'estimator': new ai.tree.DecisionTreeClassifier(),
            'max_iter': 10,
            'param_grid': {
                'max_depth': new opt.Integer(1, 5)
            }
        })
        var X = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ]
        var y = [0, 1, 0, 1, 0, 1, 0, 1]

        // try doing things manually
        await model.init(X, y)

        // first small trees, then big trees
        for(var max_depth of [3, 6]){
            model.update_ranges({
                'max_depth': new opt.Integer(1, max_depth)
            })

            for(var i of [1, 2, 3, 4]){
                await model.step()
            }
        }

        await model.finish_search()
        
        // because it overfits, train. score will be 1.0
        expect(await model.score(X, y)).to.be.equal(1.0)
        expect(await model.predict(X)).deep.equal(y)
    })
})