const utils = require('./utils')
const metrics = require('./metrics')
const base = require('./base')

class GiniCriterion{
    get_counts(y, norm=true){
        var labels = {}
        // count the occurances of class
        for(var v of y){
            if(!(v in labels)){
                labels[v] = {
                    'label': v,  // necessary to preserve the type of label
                    'count': 0
                }
            }
            labels[v]['count'] += 1
        }

        if(norm){
            var N = y.length
            for(var k in labels){
                labels[k]['count'] = labels[k]['count'] / N
            }
        }

        return labels
    }

    /**
     * Returns most frequent class in y
     * @param {array} y list of outputs
     */
    get_output(y){
        var c = this.get_counts(y)
        var max = 0
        var result = null

        for(var key in c){
            if(c[key]['count'] > max){
                max = c[key]['count']
                result = c[key]['label']
            }
        }

        return result
    }
    
    counts_impurity(labels){
        var N = 0;
        for(var k in labels){
            N += labels[k]['count']
        }

        // calculate impurity
        var result = 1.0
        for(var k in labels){
            var p_i = labels[k]['count'] / N
            result -= p_i * p_i
        }

        return result
    }

    set_impurity(y){
        var labels = this.get_counts(y, false)
        return this.counts_impurity(labels)
    }

    split_init(y){
        this.right = this.get_counts(y, false)
        this.left = {}

        for(var k in this.right){
            this.left[k] = {
                'label': this.right[k]['label'],
                'count': 0
            }
        }
    }

    split_move_after(y_v){
        this.right[y_v]['count'] -= 1
        this.left[y_v]['count'] += 1
    }

    split_impurities(){
        return [this.counts_impurity(this.left), this.counts_impurity(this.right)]
    }

}

module.exports.GiniCriterion = GiniCriterion

class MSECriterion{
    get_counts(y){
        // sum_i (ai - m)*(ai - m) = sum_i[ai*ai] - 2*m*sum_i[ai] + sum_i[m*m]
        var counts = {
            'v*v': 0,
            'v': 0,
            '1': 0
        }

        for(var v of y){
            counts['v*v'] += v*v
            counts['v'] += v
            counts['1'] += 1
        }

        return counts
    }

    get_output(y){
        var counts = this.get_counts(y)
        return counts['v'] / counts['1'] // return mean as estimation
    }

    counts_impurity(counts){
        var mean = counts['v'] / counts['1']
        var squared_error = counts['v*v'] - 2*mean*counts['v'] + counts['1']*mean*mean
        var mse = squared_error / counts['1']
        return mse
    }

    set_impurity(y){
        return this.counts_impurity(this.get_counts(y))
    }

    split_init(y){
        this.right = this.get_counts(y, false)
        this.left = {}

        for(var key in this.right){
            this.left[key] = 0.0
        }
    }

    split_move_after(v){
        // remove from right set
        this.right['v*v'] -= v*v
        this.right['v'] -= v
        this.right['1'] -= 1
        
        // add to the left set
        this.left['v*v'] += v*v
        this.left['v'] += v
        this.left['1'] += 1
    }

    split_impurities(){
        return [this.counts_impurity(this.left), this.counts_impurity(this.right)]
    }

}

module.exports.MSECriterion = MSECriterion

class NodeSplitter{
    constructor(criterion){
        this.criterion = criterion
    }

    enact_split(X, y, split){
        var X_l = [], y_l = [], X_r = [], y_r = []

        var feat = split['feature']
        var boundary = split['boundary']

        for(var i=0; i< X.length; i++){
            if(X[i][feat] < boundary){ // left child
                X_l.push(X[i])
                y_l.push(y[i])
            }else{ // right child
                X_r.push(X[i])
                y_r.push(y[i])
            }
        }

        return {
            'left': [X_l, y_l],
            'right': [X_r, y_r]
        }
    }

    best_split(X, y){
        // this will be the result
        var split = {
            'feature': null,
            'boundary': null, 
            'impurity': Number.POSITIVE_INFINITY
        }

        // data structures below are used for random selection
        // of 
        var checked_splits = {}
        var best_impurity = split['impurity']

        checked_splits[split['impurity']] = [split]

        var crit = new this.criterion()

        // join X and y
        var Xy = X.map((x, i)=>[x, y[i]])

        for(var feat=0; feat<X[0].length; feat++){
            var N_right = Xy.length
            var N_left = 0

            // sort the Xy array by feature f
            Xy.sort((a, b)=>{return a[0][feat] > b[0][feat] ? 1 : a[0][feat] < b[0][feat] ? -1 : 0})
            
            crit.split_init(y)

            for(var row=0; row<Xy.length-1; row++){
                N_right -= 1
                N_left += 1

                // move to next label
                crit.split_move_after(Xy[row][1])

                // determine new boundary
                var lb = Xy[row][0][feat]
                var rb = Xy[row+1][0][feat]

                // skip splitting if the points have same feature value
                if(lb === rb){
                    continue
                }

                var boundary = 0.5 * lb + 0.5 * rb
                
                // get node impurity
                var [li, ri] = crit.split_impurities()
                
                // weighted sum of impurities
                var impurity = (N_left * li + N_right * ri) / (N_right + N_left)

                if(best_impurity >= impurity){
                    split = {}
                    split['impurity'] = impurity
                    split['boundary'] = boundary
                    split['feature'] = feat
                    if(best_impurity == impurity){
                        checked_splits[best_impurity].push(split)
                    }else{
                        best_impurity = impurity
                        checked_splits[best_impurity] = [split]
                    }
                }
            }
        }

        // among all the best splits, select one at random
        var best = checked_splits[best_impurity]
        var idx = Math.floor(Math.random() * best.length); 
        split = best[idx]

        // note that default split with boundary === null is returned
        // in case no meaningful split exists
        return split
    }
}

module.exports.NodeSplitter = NodeSplitter

class BestFirstTreeBuilder{
    constructor(criterion, params){
        this.criterion = criterion
        this.params = params
    }

    make_node(X, y, splitter, criterion, id, depth, N){
        var impurity = criterion.set_impurity(y)
        var split = splitter.best_split(X, y)
        var improvement = y.length * (impurity - split['impurity']) / N

        return {
            'impurity': impurity,
            'output': criterion.get_output(y),
            'info': criterion.get_counts(y),
            'split': split, // information about split
            'X': X,
            'y': y,
            'improvement': improvement, // used to select the best node to split
            'left': null,
            'right': null,
            'id': id,
            'depth': depth
        }
    }

    clear(tree){
        var new_tree = {}

        for(var key of ['impurity', 'output', 'info', 'split', 'id']){
            new_tree[key] = tree[key]
        }

        for(var key of ['left', 'right']){
            if(tree[key] !== null){
                new_tree[key] = this.clear(tree[key])
            }else{
                new_tree[key] = null
            }
        }

        return new_tree
    }

    build(X, y){
        var splitter = new NodeSplitter(this.criterion)
        var criterion = new this.criterion()

        // calculate root statistics
        var N = y.length // total size of dataset
        var id = 0 // current id of the node
        var root = this.make_node(X, y, splitter, criterion, id, 1, N)
        var frontier = {
            0: root
        }

        if(this.params['min_samples_split'] > 1){
            this.params['min_samples_split'] = this.params['min_samples_split'] / N
        }

        while(Object.keys(frontier).length > 0){
            var best_node = null
            var best_improvement = this.params['min_impurity_decrease']
            // select node with highest improvement of impurity
            for(var k in frontier){
                var node = frontier[k]
                if(node === null){
                    continue
                }
                if(node['improvement'] >= best_improvement){
                    best_improvement = node['improvement']
                    best_node = node
                }
            }

            if(best_node === null){ // no improvement above minimal threshold. Terminate
                break
            }

            // remove node from frontier
            frontier[best_node['id']] = null

            if(best_node['depth'] > this.params['max_depth']){ // max depth reached
                continue
            }

            if(best_node['y'].length < this.params['min_samples_split'] * N){
                continue
            }

            // no need to do this. -inf improvement already does this.
            //if(best_node['split']['boundary'] === null){ 
            //    continue
            //}

            // no excuses to not split. Therefore split
            var children = splitter.enact_split(best_node['X'], best_node['y'], best_node['split'])

            for(var key in children){
                var [Xc, yc] = children[key]
                id += 1
                var child = this.make_node(Xc, yc, splitter, criterion, id, best_node['depth']+1, N)
                best_node[key] = child
                frontier[child['id']] = child
            }
        }

        root = this.clear(root)

        // remove unnecessary elements here
        return root
    }
}


/**
 * Converts a decision tree, given by the root of the tree, into
 * a text format that can be rendered to the user.
 * @param {Object} root Root node of a decision tree.
 * @param {Array} features Array of dictionaries, which contain information
 * on names of the features and type of the features. 
 * @param {Array} definitions Is only used for recurrent calls inside 
 * @param {Array} links Similarily, stores the links between elements
 * in graph.
 */
function make_murmaid(root, features=null, definitions=null, links=null, info=null){
    var isroot = (definitions === null)
    if(isroot){
        definitions = []
        links = []
        shared = {'counter': 0}
    }

    // add definition of the node
    var name = "Node" + shared['counter']
    shared['counter'] += 1
    var info = name

    if(root['left'] === null){ // is leaf
        info += '("' + root['output'] + '")'
    }else{
        // add definition of the node
        info += '["'
        var split = root['split']
        var feat = null

        if(features === null){
            feat = {
                'name': "Feature #" + split['feature'],
                'type': 'number'
            }
        }else{
            feat = features[split['feature']]
        }

        info += "<b>" + feat['name'] + "</b>"

        if(feat['type'] == 'number'){
            info += " > " + split['boundary'] + "?"
        }else{
            info += "?"
        }

        info += '"]'

        var lname = make_murmaid(root['left'], features, definitions, links, shared)
        var rname = make_murmaid(root['right'], features, definitions, links, shared)

        links.push(name + ' -->|false| ' + lname)
        links.push(name + ' -->|true| ' + rname)
    }

    definitions.push(info)

    if(isroot){ // return complete source of the graph
        var total_results = ['graph TD']
        total_results = total_results.concat(definitions, links)
        total_results = total_results.join('\n')
        return total_results
    }
  
    return name
}

/**
 * Generic Decision Tree model, mimicing the implementation in
 * scikit-learn Python ML package.
 */
class DecisionTreeBase extends base.BaseEstimator{
    /**
     * Instantiates a generic implementation of the decision tree.
     * For specific implementations, such as the ones for 
     * classification or regression, see other imports in this module.
     * @param {Object} criterion Class that implements a splitting
     * criterion. For example implementation, see GiniCriterion or
     * MSECriterion.
     * @param {Oject} params Configuration of the training procedure
     * and decision tree architecture. 
     * @param {Number} [params.min_impurity_decrease] Minimal decrease
     * in criterion value (e.g. impurity) to form a split. Non - zero
     * split criterion could lead to problems on some of the datasets.
     * @param {Number} [params.min_samples_split] Minimal number of
     * samples from the training dataset, with which a split can be 
     * formed. If the value is in the range of [0.0, 1.0], then the 
     * value is assumed to denote the fraction of the training dataset.
     * If the value is greater equal 2, this is assumed to be the total
     * number of samples.
     * @param {Number} [params.max_depth] Maximum depth of the decision
     * tree allowed.
     */
    constructor(criterion, params){
        super(params, {
            min_impurity_decrease: 0.0,
            min_samples_split: 2,
            max_depth: Number.POSITIVE_INFINITY

        })
        this.criterion = criterion
    }
    async fit(X, y){
        var builder = new BestFirstTreeBuilder(
            this.criterion, this.params
        )
        var root = builder.build(X, y)
        this.state = {
            'tree_': root
        }
    }
    async predict(X){
        utils.check_is_fitted(this, ['tree_'])
        
        var y_pred = []

        for(var x of X){
            var root = this.state['tree_']

            while(true){
                if(root['left'] === null){
                    break // at leaf now
                }

                var feat = root['split']['feature']
                var boundary = root['split']['boundary']

                if(x[feat] < boundary){
                    root = root['left']
                }else{
                    root = root['right']
                }
            }

            y_pred.push(root['output'])
        }

        return y_pred
    }
    /**
     * Creates a mermaid graph representation of
     * this tree. See more here:
     * https://mermaidjs.github.io/
     */
    async to_mermaid(features=null){
        utils.check_is_fitted(this, ['tree_'])
        return make_murmaid(this.state['tree_'], features)
    }
}

module.exports.DecisionTreeBase = DecisionTreeBase

class DecisionTreeClassifier extends base.ClassifierMixin(DecisionTreeBase){
    /**
     * Instantiates a DecisionTreeClassifier.
     * @param {Oject} params Configuration of the training procedure
     * and decision tree architecture. 
     * @param {Number} [params.min_impurity_decrease] Minimal decrease
     * in criterion value (e.g. impurity) to form a split. Non - zero
     * split criterion could lead to problems on some of the datasets.
     * @param {Number} [params.min_samples_split] Minimal number of
     * samples from the training dataset, with which a split can be 
     * formed. If the value is in the range of [0.0, 1.0], then the 
     * value is assumed to denote the fraction of the training dataset.
     * If the value is greater equal 2, this is assumed to be the total
     * number of samples.
     * @param {Number} [params.max_depth] Maximum depth of the decision
     * tree allowed.
     */
    constructor(params){
        super(GiniCriterion, params)
    }
}

module.exports.DecisionTreeClassifier = DecisionTreeClassifier

class DecisionTreeRegressor extends base.RegressorMixin(DecisionTreeBase){
    /**
     * Instantiates a DecisionTreeClassifier.
     * @param {Oject} params Configuration of the training procedure
     * and decision tree architecture. 
     * @param {Number} [params.min_impurity_decrease] Minimal decrease
     * in criterion value (e.g. impurity) to form a split. Non - zero
     * split criterion could lead to problems on some of the datasets.
     * @param {Number} [params.min_samples_split] Minimal number of
     * samples from the training dataset, with which a split can be 
     * formed. If the value is in the range of [0.0, 1.0], then the 
     * value is assumed to denote the fraction of the training dataset.
     * If the value is greater equal 2, this is assumed to be the total
     * number of samples.
     * @param {Number} [params.max_depth] Maximum depth of the decision
     * tree allowed.
     */
    constructor(params){
        super(MSECriterion, params)
    }
}

module.exports.DecisionTreeRegressor = DecisionTreeRegressor
