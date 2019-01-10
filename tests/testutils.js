const tf = require('@tensorflow/tfjs');
const io = require('../src/io')
const expect = require('chai').expect;

/**
 * Tests if the function with no arguments raises
 * an exception.
 * @param {function} fnc Function that contains some
 * functionality that is supposed to raise exception
 */
async function except(fnc){
    var raises = false
    try{
        await fnc()
    }catch(ex){
        raises = true
    }
    expect(raises).to.be.eq(true)
}

module.exports.except = except

/**
 * Test whether the object can be serialized and deserialized.
 * @param {Any} value object to be tested for serialization
 */
async function roundtrip_test(value){
    var clone = await io.clone(value, true)
    
    // check if types match
    expect(typeof value).deep.equal(typeof clone)

    if(typeof value === 'object' && value !== null){
        // expect class name to deeply equal
        var c1name = value.constructor.name
        var c2name = clone.constructor.name
        expect(c1name).to.be.equal(c2name)

        // special case of tf.tensor
        if(c1name === 'Tensor'){
            value = [value.shape, value.dtype, await value.data()]
            clone = [clone.shape, clone.dtype, await clone.data()]
        }
    }

    // check if values match
    expect(clone).deep.equal(value)
}

module.exports.roundtrip_test = roundtrip_test

/**
 * @param {Object} origin Estimator instance to
 * be tested for serialization
 * @param {Object} X Example inputs to process. It is
 * assumed that the model was fit already on the data,
 * if inputs are used by the model.
 * @param {Object} y Example outputs to process. Also
 * assumes that the outputs work with estimator .score
 * function, if such function is present.
 * @param {String} etype type of the estimator.
 * Could be one of: 'model', 'transformer', 'label'
 * Here 'label' means transformation on labels.
 */
async function test_serialization_estim(origin, X, y, etype){
    // these will be outputs of different models
    var out1 = null
    var out2 = null

    var clone = await io.clone(origin, true)

    if(etype === 'label'){
        out1 = [await origin.transform(y)]
        out2 = [await clone.transform(y)]
    }else if(etype === 'transformer'){
        out1 = [await origin.transform(X)]
        out2 = [await clone.transform(X)]
    }else{
        out1 = [await origin.predict(X), await origin.score(X, y)]
        out2 = [await clone.predict(X), await clone.score(X, y)]
    }

    // convert to types that can be tested
    for(var i=0; i<out1.length; i++){
        if(out1[i] instanceof tf.Tensor){
            out1[i] = await out1[i].data()
            out2[i] = await out2[i].data()
        }
    }

    // check consistency of behavior
    expect(out1).deep.eq(out2)
}

module.exports.test_serialization_estim = test_serialization_estim

async function test_estimator_behavior(estimator, params, type, base_score=0.5){
    var X = [
        [1, 0],
        [0, 1],
        [1, 1],
        [0, 0]
    ]
    var y = [1, 2, 2, 0]

    var model = new estimator(params)

    if(type == 'model'){
        await except(async ()=>{
            await model.predict(X)
        })
        await except(async ()=>{
            await model.fit(X.slice(1), y)
        })
        await except(async ()=>{
            await model.fit(X, y.slice(1))
        })
        await except(async ()=>{
            await model.fit(y, X)
        })
        await except(async ()=>{
            await model.fit(y, y)
        })
        await except(async ()=>{
            await model.fit(X, X)
        })
    }    
    
    await model.fit(X, y)
    var score = await model.score(X, y)
    // prevent catastrophic failure
    expect(score).to.be.greaterThan(base_score)

    await test_serialization_estim(model, X, y, 'model')
}

module.exports.test_estimator_behavior = test_estimator_behavior