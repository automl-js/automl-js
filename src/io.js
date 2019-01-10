const tf = require('@tensorflow/tfjs');
const base64js = require('base64-js')
const opt = require('optimization-js'
)
const base_estimators = {}

// supported serializable classes
const prep = require('./preprocessing')
base_estimators['TableFeaturesTransformer'] = prep.TableFeaturesTransformer
base_estimators['StandardScaler'] = prep.StandardScaler
base_estimators['LabelBinarizer'] = prep.LabelBinarizer

const tree = require('./tree')
base_estimators['DecisionTreeClassifier'] = tree.DecisionTreeClassifier
base_estimators['DecisionTreeRegressor'] = tree.DecisionTreeRegressor

const nn = require('./nn')
base_estimators['MLPRegressor'] = nn.MLPRegressor
base_estimators['MLPClassifier'] = nn.MLPClassifier

const linear_model = require('./linear_model')
base_estimators['SGDRegressor'] = linear_model.SGDRegressor
base_estimators['SGDClassifier'] = linear_model.SGDClassifier

const ensemble = require('./ensemble')
base_estimators['GradientBoostingRegressor'] = ensemble.GradientBoostingRegressor
base_estimators['GradientBoostingClassifier'] = ensemble.GradientBoostingClassifier

const model_selection = require('./model_selection')
base_estimators['OMGSearchCV'] = model_selection.OMGSearchCV

module.exports.base_estimators = base_estimators

// supported numerical array types for serialization
const js_array_types = {
    'Float32Array': Float32Array,
    'Float64Array': Float64Array,
    'Int8Array': Int8Array,
    'Int16Array': Int16Array,
    'Int32Array': Int32Array,
    'Uint8Array': Uint8Array,
    'Uint16Array': Uint16Array,
    'Uint32Array': Uint32Array
}

// supported optimization-js classes
const opt_js_types = {
    'Real': opt.Real,
    'Integer': opt.Integer,
    'Categorical': opt.Categorical
}

/**
 * Convert various objects to json format. Is useful for
 * serialization and deserialization of aitable objects.
 * @param {Any} obj Instance of an object to be converted
 * into a serializable json. Can be a json serializable
 * value, a class that can be serialized, or instance
 * of tensorflowjs objects.
 */
async function dumpjson(obj){
    var type = typeof obj

    if(['number', 'string', 'boolean'].includes(type) || obj === null){
        // native type, serializable
        return {
            'type': 'native',
            'value': obj
        }
    }else if(type === 'object'){
        // get class name of the object
        var cname = obj.constructor.name

        // list type
        if(cname === 'Array'){
            var serialized = []
            for(var value of obj){
                var out = await dumpjson(value)
                serialized.push(out)
            }
            return {
                'type': 'list',
                'value': serialized
            }
        }else if(cname === 'Object'){ // assumes the case of dictionary
            var serialized = [] // list of pairs of key, value
            for(var key in obj){
                serialized.push([
                    await dumpjson(key), await dumpjson(obj[key])
                ])
            }
            return {
                'type': 'dict',
                'value': serialized
            }
        }else if(cname === 'Tensor'){
            var data = await obj.data()
            var shape = obj.shape
            var dtype = obj.dtype

            return {
                'type': 'tf_tensor',
                'value': {
                    'data': await dumpjson(data),
                    'shape': shape,
                    'dtype': dtype
                }
            }
        }else if(cname === 'Sequential'){  // convert tf model
            var results = []

            var handleSave = function (artifacts){
                results.push(artifacts.modelTopology)
                results.push(artifacts.weightSpecs)
                results.push(artifacts.weightData)
            }

            await obj.save(tf.io.withSaveHandler(handleSave));

            var value = {
                modelTopology: results[0],
                weightSpecs: results[1],
                weightData: await dumpjson(new Uint8Array(results[2]))
            }  
            
            return {
                'type': 'tf_model',
                'value': value
            }
        }else if(cname in opt_js_types){
            var value = {
                'class':cname
            }
            if(cname === 'Categorical'){
                value['categories'] = obj.categories
            }else{
                value['low'] = obj.low
                value['high'] = obj.high    
            }
            return {
                'type': "optimization-js",
                'value': value
            }
        }else if(cname in js_array_types){ // convert to supported array
            var serialized = {
                'class': cname,
                'data': base64js.fromByteArray(new Uint8Array(obj.buffer))
            }
            return {
                'type': 'native_array',
                'value': serialized
            }
        }else if(cname in base_estimators){ // convert estimator to json
            var serialized = {
                'params': await dumpjson(obj.params),
                'state': await dumpjson(obj.state),
                'class': cname
            }
            return {
                'type': 'estimator',
                'value': serialized
            }
        }
    }else{
        throw Error('Unsupported object for serialization: ' + obj)
    }    
}

module.exports.dumpjson = dumpjson

/**
 * Inverse of dumpjson.
 * @param {Any} json Blueprint of the object, to be deserialized.
 */
async function loadjson(json){
    var type = json['type']
    var value = json['value']

    if(type === 'native'){
        return value
    }else if(type === 'list'){
        var result = []
        for(var v of value){
            result.push(await loadjson(v))
        }
        return result
    }else if(type === 'dict'){
        var result = {}
        for(var v of value){
            var key = await loadjson(v[0])
            var value = await loadjson(v[1])
            result[key] = value
        }
        return result
    }else if(type === 'tf_tensor'){
        var dtype = value['dtype']
        var shape = value['shape']
        var data = await loadjson(value['data'])
        var result = new tf.tensor(data, shape, dtype)
        return result
    }else if(type === 'tf_model'){
        var modelTopology = value['modelTopology']
        var weightSpecs = value['weightSpecs']
        var weightData = value['weightData']

        // load the buffer
        weightData = (await loadjson(weightData)).buffer

        // load the model
        var model = await tf.loadModel(
            tf.io.fromMemory(modelTopology, weightSpecs, weightData)
        );
        return model
    }else if(type === 'optimization-js'){
        var cname = value['class']
        var ctype = opt_js_types[cname]
        var result = null

        if(cname === 'Categorical'){
            result = new opt.Categorical(value['categories'])
        }else if(cname === 'Real'){
            result = new opt.Real(value['low'], value['high'])
        }else{
            result = new opt.Integer(value['low'], value['high'])
        }
        
        return result
    }else if(type === 'native_array'){
        var cname = value['class']
        var ctype = js_array_types[cname]
        // should return Uint8Array; Convert it to appropriate format
        var buffer = base64js.toByteArray(value['data'])
        buffer = buffer.buffer
        var result = new ctype(buffer)
        return result
    }else if(type === 'estimator'){
        var cname = value['class']
        var params = await loadjson(value['params'])
        var state = await loadjson(value['state'])
        
        // get the estimator class
        var ctype = base_estimators[cname]
        var obj = new ctype(params)
        obj.state = state

        return obj
    }else{
        throw Error('Unknown type for deserialization: ' + type + ', value: ' + value)
    }
}

module.exports.loadjson = loadjson

/**
 * Creates a copy of the object by serializing
 * the object to json, and then deserializing.
 * @param {Any} obj object to be cloned.
 * @param {Boolean} stringify whether to stringify
 * obtained blueprint. Should only be used for testing
 * of whether the serialization representation is
 * convertable to string, or in case serialized
 * object might contain pointers to nested objects
 * of the parent object, which erroneously are not
 * cloned properly.
 */
async function clone(obj, stringify=false){
    var blueprint = await dumpjson(obj)

    if(stringify){
        blueprint = JSON.stringify(blueprint)
        blueprint = JSON.parse(blueprint)
    }
    
    var result = await loadjson(blueprint)
    return result
}

module.exports.clone = clone