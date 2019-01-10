var expect = require('chai').expect;
var ai = require('../src/automljs')
var tu = require('./testutils')

describe('base', function(){
    it('processes parameters', async function(){
        // test calculation of splitting criterions
        var model = new ai.tree.DecisionTreeClassifier({'max_depth': 1})
        var params = await model.get_params()
        
        expect(params['max_depth']).to.be.equal(1)

        await model.set_params({'max_depth':2})
        params = await model.get_params()

        expect(params['max_depth']).to.be.equal(2)
    })
})