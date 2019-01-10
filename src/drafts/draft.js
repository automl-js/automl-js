
const ai = require('../automljs')

myfunc = async function(){
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
    

    var model = new ai.automl.AutoMLModel({
        'max_iter': 5
    })
    

    // ########## Regression
    var yr = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]

    var model = new ai.automl.AutoMLModel({
        'max_iter': 5
    })
    
    // determine automatically the type of learning problem
    // and find decent performing model
    await model.fit(X, yr)

    // evaluate the performance of model
    var score_clasif = await model.score(X, yr)

    // make estimations with the best found model
    var y_pred = await model.predict(X)
}

myfunc()