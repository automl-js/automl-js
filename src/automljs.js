// use cpu backend for tf
const tf = require('@tensorflow/tfjs')
tf.setBackend('cpu')

// io needs to be imported first. Otherwise issues with importing
module.exports.io = require('./io')

module.exports.automl = require('./automl')
module.exports.ensemble = require('./ensemble')
module.exports.linear_model = require('./linear_model')
module.exports.tree = require('./tree')
module.exports.nn = require('./nn')
module.exports.preprocessing = require('./preprocessing')
module.exports.metrics = require('./metrics')
module.exports.model_selection = require('./model_selection')
module.exports.utils = require('./utils')