const utils = require('./utils')
const prep = require('./preprocessing')
const metrics = require('./metrics')
const tf = require('@tensorflow/tfjs')
const base = require('./base')

class MLPBase extends base.BaseEstimator{
    constructor(params){
        super(params, {
            'n_neurons': 32,
            'lr': 1e-4,
            'epochs': 100,
            'batch_size': 128
        })
        // to be specified by child classes
        this.outputs = 1
        this.loss = 'meanSquaredError'
        this.final_layer_activation = 'linear'
    }
    
    fit(X, y){
        X = utils.t2d(X)
        if(this.outputs == 1){
            y = utils.t1d(y)
        }else{
            y = utils.t2d(y)
        }

        var sample_shape = [X.shape[1]]

        var params = this.params
        var model = tf.sequential()
        model.add(tf.layers.dense({units: params['n_neurons'], inputShape: sample_shape}))
        model.add(tf.layers.leakyReLU({'alpha': 0.01}))
        model.add(tf.layers.dense({units: this.outputs, activation: this.final_layer_activation}))

        model.compile({
            loss: this.loss, 
            optimizer: tf.train.adam(params['lr'])
        });

        var res = model.fit(X, y, {'epochs': params['epochs'], 'batchSize': params['batch_size']})

        this.state['model'] = model

        return res
    }

    predict(X){
        var X = utils.t2d(X)
        var model = this.state['model']
        var y_pred = model.predict(X)
        return y_pred
    }
}

class MLPRegressor extends base.RegressorMixin(MLPBase) {
    constructor(params){
        super(params)
    }

    predict(X){
        var y_pred = super.predict(X)
        y_pred = tf.reshape(y_pred, [y_pred.shape[0]]) // drop last dimension
        y_pred = y_pred.dataSync(Float32Array)  //convert to js array
        return y_pred
    }
}

module.exports.MLPRegressor = MLPRegressor

class MLPClassifier extends base.ClassifierMixin(MLPBase) {
    constructor(params){
        super(params)
        this.loss = tf.losses.softmaxCrossEntropy
        this.final_layer_activation = 'softmax'
    }

    fit(X, y){
        var binarizer = new prep.LabelBinarizer()
        binarizer.fit(y)
        var y = binarizer.transform(y)

        this.outputs = binarizer.state['n_classes']
        this.state['binarizer'] = binarizer
        return super.fit(X, y)
    }

    predict(X){
        X = utils.t2d(X)
        var binarizer = this.state['binarizer']
        var y_pred = super.predict(X)
        var y_pred = binarizer.inverse_transform(y_pred)
        return y_pred
    }
}


module.exports.MLPClassifier = MLPClassifier