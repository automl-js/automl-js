const expect = require('chai').expect
const ai = require('../src/automljs')

describe('metrics', async function(){
    it('r2 metric works similar to sklearn', async function(){
        var A = [0, 1, 1, 0, 1]
        var B = [1, 0, 1, 0, 1]
        var C = [0, 0, 0, 0, 0]
        var w = [1, 1, 2, 2, 1]
        
        expect(ai.metrics.r2_score(A, B)).to.be.greaterThan(-0.67).lessThan(-0.66)
        expect(ai.metrics.r2_score(B, A)).to.be.greaterThan(-0.67).lessThan(-0.66)

        // behavior below is consistent with scikit - learn
        expect(ai.metrics.r2_score(C, B)).to.be.equal(0.0)
        expect(ai.metrics.r2_score(C, A)).to.be.equal(0.0)
    })
    it('accuracy fraction works similar to sklearn', async function(){
        var A = [1, 0, 1, 0, 1]
        var B = [1, 1, 0, 0, 1]

        expect(ai.metrics.accuracy_score(A, A)).to.be.equal(5.0 / 5.0)
        expect(ai.metrics.accuracy_score(A, B)).to.be.equal(3.0 / 5.0)

        var C = ['a', 'ab', 'd', 'e', 'f']
        var D = ['a', 'ab', 'c', 'd', 'f']

        expect(ai.metrics.accuracy_score(C, D)).to.be.equal(3.0 / 5.0)
        expect(ai.metrics.accuracy_score(A, D)).to.be.equal(0.0 / 5.0)
    })
})
