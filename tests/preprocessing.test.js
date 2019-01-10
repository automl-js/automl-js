var expect = require('chai').expect;
var ai = require('../src/automljs')
var tu = require('./testutils')
const tf = require('@tensorflow/tfjs');

describe('preprocessing', async function(){
    it('runs TableTransformer with numerical data only', async function(){
        var X = [
            [0, 1, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0]
        ]
        var y = [1, 2, 3, 4]
        var features = new ai.preprocessing.TableFeaturesTransformer({})
        features.fit(X, y)
        X = features.transform(X)
    })
    it('runs TableTransformer with categorical, missing and numerical string', async function(){
        var X = [
            ['a', 0],
            ['b', '1'],
            ['a', 0],
            ['b', -1],
            ['a', -1],
            ['b', '1e-1'],
            ['b', ''],
        ]
        var y = [-1, 2, -1, 0, -2, 1.1, 1]
        var y_c = y.map((v) => v < 0.0 ? 'neg' : 'pos')
        
        var X_exp = [
            [1, 0, 0],
            [0, 1, 1],
            [1, 0, 0],
            [0, 1, -1],
            [1, 0, -1],
            [0, 1, 0.1],
            [0, 1, -0.225]
        ]

        var feats = new ai.preprocessing.TableFeaturesTransformer({})
        var Xt = feats.fit(X, y, ['Feature A', 'Feature B']).transform(X, y)

        // check if types are detected correctly
        expect(Xt).deep.equal(X_exp)
        expect(feats.state['output']['type']).equal('number')

        feats.fit(X, y_c)
        expect(feats.state['output']['type']).equal('category')

        // test serialization
        tu.test_serialization_estim(feats, X, null, 'transformer')
    })
    it('executes StandardScaler similar to sklearn', async function(){
        var X = [[1.0, -1.0], [0.5, 1.0], [0.0, 0.5]]

        var sc = new ai.preprocessing.StandardScaler()
        sc.fit(X)

        // values from python implementation
        var Xe = tf.tensor2d([
            [ 1.22474487, -1.37281295],
            [ 0.        ,  0.98058068],
            [-1.22474487,  0.39223227]
        ])

        var Xr = sc.transform(X)
        
        // total error of values
        var error = tf.sum(tf.abs(tf.sub(Xe, Xr))).get()
        expect(error).to.be.lessThan(1e-6)

        // serialization test
        tu.test_serialization_estim(sc, X, null, 'transformer')
    })
})
