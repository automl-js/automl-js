
var dataset_table = null
var weights_table = null

async function runTraining(){
    var data = dataset_table.getData()

    // get the column names
    var fields = dataset_table.getColumns()
    fields = fields.map((v)=>v._column.field)

    // make up a dataset of inputs and outputs
    var X = []
    var y = []

    var x_fields = fields.slice(0, -1)
    var y_field = fields[fields.length - 1]

    for(var row of data){
        var features = []

        for(var f of x_fields){
            features.push(row[f])
        }

        X.push(features)
        y.push(row[y_field])
    }

    // skip missing target values
    var I = y.map((v)=>v!=="")
    var X = X.filter((v, i)=>I[i])
    var y = y.filter((v, i)=>I[i])

    var [X_train, X_test, y_train, y_test] = aml.model_selection.train_test_split([X, y], test_split=0.25)

    // training part
    var feats = new aml.preprocessing.TableFeaturesTransformer()

    await feats.fit(X_train, y_train, x_fields)
    var feat_params = feats.state['feature_params']

    var X_train = await feats.transform(X_train, y_train)
    var X_test = await feats.transform(X_test, y_test)

    // if output is numerical, then parse the outputs
    if(feats.state['output']['type'] === 'number'){
        y_train = y_train.map((v)=>Number.parseFloat(v))
        y_test = y_test.map((v)=>Number.parseFloat(v))
    }

    var tree_model = null

    if(feats.state['output']['type'] === 'number'){
        tree_model = new aml.tree.DecisionTreeRegressor({'max_depth': 3})
    }else{
        tree_model = new aml.tree.DecisionTreeClassifier({'max_depth': 3})
    }

    await tree_model.fit(X_train, y_train)
    var tree_score = await tree_model.score(X_test, y_test)
    document.getElementById('tree_score').innerHTML = tree_score.toFixed(3)

    // make the tree visualization
    var tree = await tree_model.to_mermaid(feat_params)

    // update the tree
    var vis = document.getElementById('best_decision_tree')

    var insertSvg = function(svgCode, bindFunctions){
        vis.innerHTML = svgCode;
        vis.firstChild.style.height = vis.getAttribute('viewbox').split(' ')[3] + 'px'
        svgPanZoom(vis.firstChild, {
            controlIconsEnabled: true,
        })
    };

    mermaidAPI.render('best_decision_tree', tree, insertSvg);

    // train a linear model

    var lin_model = null

    // if output is numerical, then parse the outputs
    if(feats.state['output']['type'] === 'number'){
        lin_model = new aml.linear_model.SGDRegressor({'max_iter': 1000})
    }else{
        lin_model = new aml.linear_model.SGDClassifier({'max_iter': 1000})
    }
    await lin_model.fit(X_train, y_train)
    var lin_score = await lin_model.score(X_test, y_test)
    document.getElementById('lin_score').innerHTML = lin_score.toFixed(3)

    var [data, fields] = lin_model.to_tabulator(feat_params)

    // create a new table
    weights_table = new Tabulator("#best_linear_model", {
        height:400, // set height of table (in CSS or here), this enables the Virtual DOM and improves render speed dramatically (can be any valid css height value)
        data: data, //assign data to table
        layout:"fitColumns", //fit columns to width of table (optional)
        columns:fields
    });

}

function uploadCSV() {
    var form = document.getElementById('dataset_file')
    var file = form.files[0]
    
    Papa.parse(file, {
        header: true,
        complete: setSheet,
        skipEmptyLines: true
    })
}


async function setSheet(contents, file){
    var fields = contents['meta']['fields']
    var dataset = contents['data']

    // format into Tabulator format
    fields = fields.map((v)=>{ return {title: v, field: v, width: 9*(v.length + 4)}})
    dataset = dataset.map((v, i)=>{
        var result = {id: i+1}
        for(var k in v){
            result[k] = v[k]
        }
        return result
    })

    // create a new table
    dataset_table = new Tabulator("#dataset_rendering", {
        height:400, // set height of table (in CSS or here), this enables the Virtual DOM and improves render speed dramatically (can be any valid css height value)
        data: dataset, //assign data to table
        layout:"fitColumns", //fit columns to width of table (optional)
        columns:fields
   });

}