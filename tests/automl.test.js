const expect = require('chai').expect
const ai = require('../src/automljs')

describe('automl', async function(){
    it('fills last column, regression', async function(){
        var dataset = [
            ['a', 0.6, ''],
            ['b', -1.3, ''],
            ['a', 1, 0],
            ['b', 2, 3],
            ['a', 0.5, -0.5],
            ['b', -0.3, 0.7],
            ['a', 0.4, -0.6],
            ['b', 1.1, 2.1],
            ['', -0.8, 0.2],
            ['a', "", -0.7]
        ]
        
        function callback(data){
            var i = data['iteration']
            expect(data['dataset'].length).to.be.equal(dataset.length)
            expect(data['dataset'][0].length).to.be.equal(dataset[0].length)
            expect(i).to.be.lessThan(2)
            if(i>=1){
                return 'cancel'
            }
        }
        
        var state = await ai.automl.fill_last_column(dataset, callback, 3, 2)
    })
    it('fills last column, classification', async function(){
        var dataset = [
            ['a', 0.6, ''],
            ['b', -1.3, ''],
            ['a', 1.1, 'pos'],
            ['b', -2, 'neg'],
            ['a', 0.5, 'neg'],
            ['b', "", 'pos'],
            ['a', 0.4, 'neg'],
            ['', 1.1, 'pos'],
            ['b', -0.8, 'pos'],
            ['a', "1e-1", 'neg']
        ]
        
        function callback(data){}
        
        var state = await ai.automl.fill_last_column(dataset, callback, 2, 2)
    })
    it('runs training with AutoML estimator', async function(){
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
    
        var y = ['b', 'b', 'a', 'b', 'a', 'b', 'a', 'a']
        
        var event_fnc = function(iter_result){
            var iteration = iter_result['iteration']
            var n_iter = iter_result['n_iter']
            var model_name = iter_result['model_name']

            return iteration > 0.5*n_iter
        }

        var model = new ai.automl.AutoMLModel({
            'max_iter': 7
        })
        
        model.on_improve['event'] = event_fnc
        model.on_improve['event2'] = null // this one is simply ignored

        // determine automatically the type of learning problem
        // and find decent performing model
        await model.fit(X, y)
    
        // evaluate the performance of model
        var score_clasif = await model.score(X, y)
   
        // make estimations with the best found model
        var y_pred = await model.predict(X)


        // ########## Regression
        var yr = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]

        var model = new ai.automl.AutoMLModel({
            'max_iter': 7
        })
        
        model.on_step['event'] = event_fnc
        model.on_step['event2'] = null // this one is skipped

        // determine automatically the type of learning problem
        // and find decent performing model
        await model.fit(X, yr)
    
        // evaluate the performance of model
        var score_clasif = await model.score(X, yr)
   
        // make estimations with the best found model
        var y_pred = await model.predict(X)
    })
})
