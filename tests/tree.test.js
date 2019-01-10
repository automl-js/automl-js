var expect = require('chai').expect;
var ai = require('../src/automljs')
var tu = require('./testutils')

var close = function(a, b){
    expect(Math.abs(a - b) < 1e-6).to.be.equal(true)
}

async function test_eq(X, y, X_test, y_sklearn, params=null, regression=false){
    var model = null

    if(regression){
        model = new ai.tree.DecisionTreeRegressor(params)
    }else{
        model = new ai.tree.DecisionTreeClassifier(params)
    }

    await model.fit(X, y)
    var y_pred = await model.predict(X_test)
    if(regression){
        for(var yi=0; yi<y.length; yi++){
            close(y_pred[yi], y_sklearn[yi])
        }
    }else{
        expect(y_pred).deep.equal(y_sklearn)
    }
}

describe('tree', function(){
    it('computes impurities', async function(){
        // test calculation of splitting criterions
        var gini = new ai.tree.GiniCriterion()
        expect(gini.set_impurity([0,1,0,1])).to.be.equal(0.5)
        expect(gini.set_impurity([1,1,1,1])).to.be.equal(0.0)

        var mse = new ai.tree.MSECriterion()
        expect(mse.set_impurity([1,1,1,1])).to.be.equal(0.0)
        expect(mse.set_impurity([0,1,1,2])).to.be.equal(0.5)

        gini.split_init([0, 0, 1, 1])
        gini.split_move_after(1)
        gini.split_move_after(1)

        var [l, r] = gini.split_impurities()
        expect(l+r).to.be.equal(0.0)

        gini.split_init([0, 0, 0, 1, 1, 1, 2, 2, 2])
        gini.split_move_after(0)
        gini.split_move_after(1)
        gini.split_move_after(2)

        var [l, r] = gini.split_impurities()
        close(l, 6.0/9.0)
        close(r, 6.0/9.0)

        mse.split_init([0, 1, 3, 1])
        mse.split_move_after(0)
        mse.split_move_after(1)

        var [l, r] = mse.split_impurities()
        close(l, 0.25)
        close(r, 1.0)

    })
    it('does node splits', async function(){
        // Node split, trivial case
        var X = [[0, 1],[1, 0],[2, 1],[3, 0]]
        var y = [0, 0, 1, 1]

        var splitter = new ai.tree.NodeSplitter(ai.tree.GiniCriterion)
        var sp = splitter.best_split(X, y)

        expect(sp['boundary']).to.be.lessThan(2).greaterThan(1)
        expect(sp['feature']).to.be.equal(0)

        // what happens if some points cannot be split
        var X = [[0],[1],[1],[3]]
        var y = [ 0,  0,  1,  1]

        var splitter = new ai.tree.NodeSplitter(ai.tree.GiniCriterion)
        var sp = splitter.best_split(X, y)

        // consistent with sklearn
        expect(sp['boundary']).to.be.lessThan(0.51).greaterThan(0.49)

    })
    it('fits, scores, clones DecisionTreeClassifier', async function(){
        // test whether the outputs are similar to sklearn

        // perfect classifier
        X = [[0, 0],[1, 0],[1, 2],[3, 1]]
        y = [ 0,  0,  1,  1 ]

        await test_eq(X, y, X, [0, 0, 1, 1])

        // what happens if all points cannot be split
        var X = [[1],[1],[1],[1]]
        var y = [ 0,  0,  0,  1]

        await test_eq(X, y, X, [0, 0, 0, 0])

        // what happens if some points cannot be split
        var X = [[0],[1],[1],[3]]
        var y = [ 0,  0,  1,  1]

        await test_eq(X, y, X, [0, 0, 0, 1])

        // stopping criteria test
        await test_eq(X, y, X, [0, 1, 1, 1], {'max_depth': 1})
        await test_eq(X, y, X, [0, 1, 1, 1], {'min_samples_split': 4})

        // a more elaborate dataset
        var X = [[ 1.62, -0.61, -0.53],
        [-1.07,  0.87, -2.3 ],
        [ 1.74, -0.76,  0.32],
        [-0.25,  1.46, -2.06],
        [-0.32, -0.38,  1.13],
        [-1.1 , -0.17, -0.88],
        [ 0.04,  0.58, -1.1 ],
        [ 1.14,  0.9 ,  0.5 ],
        [ 0.9 , -0.68, -0.12],
        [-0.94, -0.27,  0.53]]
        var y = [-1,  1, -1,  1,  0,  1,  0, -1,  0,  0]

        await test_eq(X, y, X, y)
        await test_eq(X, y, X, [-1,  0, -1,  0,  0,  0,  0, -1,  0,  0], {'max_depth': 1})
        await test_eq(X, y, X, [-1,  1, -1,  1,  0,  1,  1, -1,  0,  0], {'max_depth': 2})
        await test_eq(X, y, X, [-1.,  0., -1.,  0.,  0.,  0.,  0., -1.,  0.,  0.], {'min_samples_split': 8})
        await test_eq(X, y, X, [-1.,  1., -1.,  1.,  0.,  1.,  1., -1.,  0.,  0.], {'min_impurity_decrease': 0.16})


        var X = [[1, 1, 0, 1, 1],
        [0, 0, 1, 1, 0],
        [1, 1, 0, 0, 0],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 1, 1],
        [1, 1, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 0, 0, 0, 1],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0],
        [1, 1, 0, 0, 1],
        [1, 1, 1, 1, 0],
        [0, 1, 1, 0, 1],
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0],
        [0, 0, 1, 1, 1],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 1],
        [1, 1, 1, 0, 1],
        [1, 0, 1, 1, 1],
        [1, 0, 1, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 1, 0, 1, 0],
        [1, 0, 0, 1, 0]]

        var y =[0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0,
            1, 1, 1, 0, 0, 0, 0, 0, 0, 0]

        // test serialization, score
        var model = new ai.tree.DecisionTreeClassifier()
        await model.fit(X, y)
        var score = await model.score(X, y)
        expect(score).to.be.equal(1.0)

        await tu.test_serialization_estim(model, X, y, 'model')
    })
    it('fits, scores, clones DecisionTreeRegressor', async function(){
        var X = [[ 1.62, -0.61, -0.53],
        [-1.07,  0.87, -2.3 ],
        [ 1.74, -0.76,  0.32],
        [-0.25,  1.46, -2.06],
        [-0.32, -0.38,  1.13],
        [-1.1 , -0.17, -0.88],
        [ 0.04,  0.58, -1.1 ],
        [ 1.14,  0.9 ,  0.5 ],
        [ 0.9 , -0.68, -0.12],
        [-0.94, -0.27,  0.53]]
        var y = [-1,  1, -1,  1,  0,  1,  0, -1,  0,  0]
        // test regressor
        await test_eq(X, y, X, [-1. ,  0.42857143, -1.,  0.42857143,  0.42857143,  0.42857143,  0.42857143, -1.,  0.42857143,  0.42857143], {'max_depth': 1}, true)
        await test_eq(X, y, X, [-1. ,  0.42857143, -1.,  0.42857143,  0.42857143,  0.42857143,  0.42857143, -1.,  0.42857143,  0.42857143], {'min_samples_split': 8}, true)
        await test_eq(X, y, X, [-1.  ,  0.75, -1.  ,  0.75,  0.  ,  0.75,  0.75, -1.  ,  0.  ,0.  ], {'max_depth': 2}, true)
        await test_eq(X, y, X, [-1.  ,  0.75, -1.  ,  0.75,  0.  ,  0.75,  0.75, -1.  ,  0.  ,0.  ], {'min_impurity_decrease': 0.09}, true)

        // test serialization
        var model = new ai.tree.DecisionTreeRegressor()
        await model.fit(X, y)
        await tu.test_serialization_estim(model, X, y, 'model')
    })
    it('Renders tree to mermaid', async function(){
        var X = [
            [1, 0],
            [0, 1],
            [1, 1],
            [0, 0]
        ]
        var y = [1, 2, 2, 0]
    
        var model = new ai.tree.DecisionTreeClassifier()
        
        await model.fit(X, y)
        var tree = await model.to_mermaid()
        var tree = await model.to_mermaid([
            {'name': 'A', 'type': 'boolean'},
            {'name': 'B', 'type': 'number'}
        ])
    })
})