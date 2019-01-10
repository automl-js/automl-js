const expect = require('chai').expect;
const ai = require('../src/automljs')
const tu = require('./testutils')
const tf = require('@tensorflow/tfjs');


describe('io', async function(){
    it('clones number, string', async function(){
        await tu.roundtrip_test(1)
        await tu.roundtrip_test(1.0)
        await tu.roundtrip_test('hello')
        await tu.roundtrip_test(true)
        await tu.roundtrip_test(null)
    })
    it('clones array of native values', async function(){
        await tu.roundtrip_test([])
        await tu.roundtrip_test([1])
        await tu.roundtrip_test([1, 2, ['a'], {'a': 'b'}])
    })
    it('clones dict of native values', async function(){
        await tu.roundtrip_test({})
        await tu.roundtrip_test({1: 2})
        await tu.roundtrip_test({1: true, true: 1})
        await tu.roundtrip_test({'a': {'b': {'c': 1, 'd': true}}, 'b': [[1, 2]]})
    })
    it('raises tu.exception on unsupported types', async function(){
        await tu.except(async function(){
            await ai.io.clone(()=>{return 0})
        })

        await tu.except(async function(){
            await ai.io.loadjson({'type': 'broken_type_vfr45tg', 'value': 'huh'})
        })
    })
    it('works with native type state estimator', async function (){
        var est = new ai.preprocessing.LabelBinarizer()
        y = [1, 1, 0, 1, 1, 0]
        await est.fit(y)
        await tu.test_serialization_estim(est, null, y, 'label')
    })
    it('works with Float/Int JS Array', async function (){
        await tu.roundtrip_test(new Float32Array([1, 2, 3]))
        await tu.roundtrip_test(new Float64Array([1, 2, 3]))
        await tu.roundtrip_test(new Int8Array([1, 2, 3]))
        await tu.roundtrip_test(new Int16Array([1, 2, 3]))
        await tu.roundtrip_test(new Int32Array([1, 2, 3]))
        await tu.roundtrip_test(new Uint8Array([1, 2, 3]))
        await tu.roundtrip_test(new Uint16Array([1, 2, 3]))
        await tu.roundtrip_test(new Uint32Array([1, 2, 3]))
    })
    it('works with tf.tensor', async function (){
        var arr = [1, 2, 3, 4, 5, 6, 7, 8]
        
        // test float
        await tu.roundtrip_test(tf.tensor(arr, [8]))
        await tu.roundtrip_test(tf.tensor(arr, [2, 4]))
        await tu.roundtrip_test(tf.tensor(arr, [2, 2, 2]))

        // test boolean
        await tu.roundtrip_test(tf.tensor([true, true, false, false], [2, 2], 'bool'))
    })
})