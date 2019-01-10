const expect = require('chai').expect;
const ai = require('../src/automljs')
const tu = require('./testutils')
const tf = require('@tensorflow/tfjs');
const utils = ai.utils

describe('utils', async function(){
    it('converts 1d, 2d tfjs arrays', async function(){
        var A = tf.tensor2d([[1, 2], [3, 4]])
        var B = [[1, 2], [3, 4]]

        var Ap = ai.utils.t2d(A)
        expect(Ap.shape.length).to.be.equal(2)
        var Ap = ai.utils.t2d(B)
        expect(Ap.shape.length).to.be.equal(2)
        expect(Array.from(await Ap.data())).deep.equal([1, 2, 3, 4])

        var A = tf.tensor2d([[1], [3]])
        var Ap = ai.utils.t1d(A)
        expect(Ap.shape.length).to.be.equal(1)
        expect(Array.from(await Ap.data())).deep.equal([1, 3])

        var Ap = ai.utils.t1d(tf.tensor1d([1, 2, 3]))
        expect(Ap.shape.length).to.be.equal(1)

        var Ap = ai.utils.t1d([1, 2])
        expect(Ap.shape.length).to.be.equal(1)
    })
    it('partitions properly', function(){
        var part = ai.utils.random_partitioning

        for(var ptt of [[1, 1, 1], [10, 100, 1, 0.1], [10, 1, 3]]){
            for(var N of [4, 5, 13, 15, 127]){
                // fix partitions if necessary
                var result = ai.utils.random_partitioning(N, ptt)
                var total_len = 0
                for(var r of result){
                    expect(r.length).to.be.greaterThan(0)
                    total_len += r.length
                }
                expect(total_len).to.be.equal(N)
            }
        }

        // single partition situation
        var result = ai.utils.random_partitioning(3, [0.3])
        expect(result.length).to.be.equal(1)

        // equal partitions situation
        var result = ai.utils.random_partitioning(4, [1, 1, 1])
        var total_len = 0
        for(var r of result){
            expect(r.length).to.be.greaterThan(0)
            total_len += r.length
        }
        expect(total_len).to.be.equal(4)

        // check exectional case
        var throws = false
        try{
            part(2, [1, 1, 1])
        }catch(ex){
            throws = true
        }
        expect(throws).to.be.equal(true)
    })
    it('checks shapes of tensors', async function(){
        var result = await utils.check_array([1, 2, 3], 1, 1, null, null, false)
        expect(result.constructor.name).to.be.not.equal('Tensor')

        await tu.except(async ()=>{ // too many dimensions
            await utils.check_array([[1], [2]], 1, 1)
        })
        await tu.except(async ()=>{ // too many dims, tfjs
            await utils.check_array(tf.randomNormal([2,2]), 1, 1)
        })
        await tu.except(async ()=>{ // too less dims
            await utils.check_array(tf.randomNormal([2,2]), 3, 3)
        })
        await tu.except(async ()=>{ // too less elements in dimension
            await utils.check_array([1, 2, 3], 1, 1, [4])
        })
        await tu.except(async ()=>{ // too many elements
            await utils.check_array([1, 2, 3], 1, 1, [1], [2])
        })
        await tu.except(async ()=>{ // inconsistent dataset sizes, tfjs
            await utils.check_2dXy(tf.randomNormal([2,2]), tf.randomNormal([2,2]))
        })
    })
    it('checks whether estimator is fitted', async function(){
        var model = new ai.tree.DecisionTreeBase()
        model.state['A'] = true
        utils.check_is_fitted(model, ['A', 'B'], 'any')
    })
})
