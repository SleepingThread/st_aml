"""
dynamic block - block with undefined number of outputs.
Dynamic block requires fit method to use, when applied.

dynamic_build in BaseNetwork allows you to build and apply
network on the fly.

BaseNetwork - allows you to build network on the fly
and use builded network as sklearn estimator.

Base principles:
    Block class needs to store algorithm itself.



Warning
-------
I want the above to be true.

Todo
----
GDT - Generalized Decision Tree
MGUALinearRegressionBlock - + process_tensors
SubsetReduce - + process_tensors

Note
----
Tell something about this module.
"""

from sklearn.utils.metaestimators import _BaseComposition

import numpy as np

import six
from abc import ABCMeta, abstractmethod

from sklearn.base import clone
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate

from st_aml.alg.mgua import MGUA


class DataBlock(dict):
    pass


class _NetworkNode(object):
    """
    Note
    ----
    This class created to store two types of blocks:
        Data Block under self.data field and
        Data Manipulation Block under self.block field.
        
    Data Block can have only one input and any number of outputs.
    Because of that Data Block outputs can be added, other inputs
    and outputs of any node could only be setted.
    Data Manipulation Block can have fixed number of inputs and outputs - 
    only set_inputs and set_outputs are available.
    """
    def __init__(self,data = None,
                 block = None,manipulations = None, 
                 inputs = None, outputs = None):
        """
        Parameters
        ----------
        data - DataBlock
        block - int
        
        Note
        ----
        self.mark - for service purposes
        """
        
        self.data = data
        self.block = block
        self.manipulations = manipulations
        
        # lists of next and previous nodes
        self.inputs = inputs or []
        self.outputs = outputs or []
        
        # mark - for service purposes
        self.mark = False
        
        return
   
    def __str__(self):
        return str(self.block) if self.is_block() else "Data"

    def is_data(self):
        """
        Returns
        -------
        Whether node contain Data Block.
        """
        return self.block is None
    
    def is_block(self):
        """
        Returns
        -------
        Whether node contain Data Manipulation Block.
        """
        return self.block is not None
    
    def is_empty(self):
        """
        Note
        ----
        Works only with Data Block nodes.
        
        Returns
        -------
        Where self.data is empty.
        """
        assert self.is_data()
        return self.data == {}
    
    def add_outputs(self,outputs):
        """
        Note
        ----
        Only for data nodes.
        """
        
        if self.is_block():
            raise Exception("Method is applicable only with block nodes.")
        
        if isinstance(outputs,list):
            self.outputs.extend(outputs)
        else:
            self.outputs.append(outputs)
        
        return self
    
    def set_inputs(self,inputs):
        """
        Note
        ----
        Both for Data Block and Data Manipulation Block nodes.
        """
        if self.is_data():
            assert len(inputs) == 1
        else:
            assert len(inputs) == self.block.n_inputs
            
        self.inputs = inputs
        return
    
    def set_outputs(self,outputs):
        """
        Note
        ----
        Both for Data Block and Data Manipulation Block nodes.        
        """
        self.outputs = outputs
        return
    
    def clear_data(self):
        self.data = DataBlock()
        return
    
    def reset_mark(self):
        self.mark = False
        return
    
    def set_mark(self):
        self.mark = True
        return
    
    def delete(self):
        """
        Warning
        -------
        Recursive algorithm. Is this is bad? Maybe yes.
        
        Note
        ----
        Deletes node from graph it enters.
        """
        for _inp in self.inputs:
            _inp.outputs.remove(self)
            
        for _out in self.outputs:
            _out.delete()
        
        return
    
    def process(self,network,fit=False):
        """
        Note
        ----
        Works only with Data Manipulation nodes.
        """
        assert self.is_block()
        _block = network.blocks[self.block]
        
        _out = _block.process([_inp.data for _inp in self.inputs],
                              manipulations = self.manipulations,
                              fit = fit)
        
        for _out_node, _out_val in six.itertools.izip(self.outputs,_out):
            # _out_node, _out_val
            _out_node.data.update(_out_val)
        
        return 
    
    def process_tensors(self,network,tens_dict):
        """
        Parameters
        ----------
        network : child of BaseNetwork

        tens_dict : dict
            maps _NetworkNode to tensor or 
            list or tuple of tensors.

        Returns
        -------
        list of tensors or list or tuple of tensors
        """

        assert self.is_block()
        _block = network.blocks[self.block]
        _out_tens = _block.process_tensors([tens_dict[_el] for _el in self.inputs],
                                          manipulations = self.manipulations)
        for _tens,_out_node in six.itertools.izip(_out_tens,self.outputs):
            tens_dict[_out_node] = _tens
            
        return _out_tens


class MetaBlock(six.with_metaclass(ABCMeta,object)):
    @abstractmethod
    def __call__(network,inputs,manipulations = None):
        return
    

class Block(six.with_metaclass(ABCMeta, BaseEstimator,MetaEstimatorMixin)):
    """
    Note
    ----
    Two types of blocks:
        1. with specified outputs ( inputs can be specified 
            in time of __call__ method ).
        2. with unspecified outputs - dynamic blocks.
    """
    
    def __init__(self,n_inputs=1,n_outputs=1):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        return
    
    def __call__(self,network,inputs,
                 manipulations=None):
        """
        Parameters
        ----------
        network : BaseNetwork object or None
            If None: block will be processed in online mode 
            (dynamic execution).
        
        inputs : list of _NetworkNode objects or _NetworkNode object
            
        manipulations : [(<method_name>,<dict of method params>), ... ]
            Specify methods to execute witin block
        Note
        ----
        Generate outputs nodes from inputs, add new blocks into network.
        """
        
        if not isinstance(inputs,list):
            inputs = [inputs]
        
        manipulations = manipulations or [("fit_block",{}),("process_block",{})]
        
        # check: if self.is_dynamic() => network.dynamic_build
        assert network is not None and ( not self.is_dynamic() or network.dynamic_build )
        
        if self.n_inputs:
            assert self.n_inputs == len(inputs)
        else:
            self.n_inputs = len(inputs)
        
        if network is None or network.dynamic_build:
            _out_val = self.process([_inp.data for _inp in inputs],
                         manipulations=manipulations, fit = True)
        else:
            _out_val = [DataBlock() for _i in range(self.n_outputs)]
        
        if network is None:
            ind = -1
        elif self in network.blocks:
            ind = network.blocks.index(self)
        else:
            ind = len(network.blocks)
            network.blocks.append(self)
            
        _block_node = _NetworkNode(block=ind,manipulations=manipulations,inputs=inputs)
        _out_nodes = [_NetworkNode(data=_out_val[_i],inputs=[_block_node]) for _i in range(self.n_outputs)]
        _block_node.set_outputs(_out_nodes)
        
        if network is not None:
            for _inp in inputs:
                _inp.add_outputs(_block_node)
        
        return _block_node, _out_nodes
    
    def is_dynamic(self):
        """ 
        """
        if self.n_outputs is None:
            return True
        
        return False
    
    def process(self, data,
                manipulations = None,
                fit=False) :
        """
        Parameters
        ----------
        data : list of dicts
        
        Returns
        -------
        Result of the last manipulation.
        
        Warning
        -------
        ONLY THE LAST MANIPULATION RESULT IS RETURNED
        
        Note
        ----
        This method must support dynamic blocks - where n_outputs can
        be unknown.
        """
        
        manipulations = manipulations or [("fit_block",{}),("process_block",{})]
        
        _manips = set([_el[0] for _el in manipulations])
        assert len(_manips)==len(manipulations)
        
        if not fit and "fit_block" in _manips:
            _manips.remove("fit_block")
            
        for _man in manipulations:
            if _man[0] not in _manips:
                continue
            
            _method = getattr(self,_man[0])
            _args = _man[1]
            
            _out = _method(data,**_args)
        
        assert self.n_outputs is not None
        
        return _out
    
    @abstractmethod
    def process_tensors(self, tensors, 
                       manipulations = None):
        """
        Parameters
        ----------
        tensors : list of ( tensor | list of tensors |
            tuple of tensors )
            Each element corresponds to the block input.
        
        manipulations : list of ( str method_name, dict params )

        Examples
        --------
        No examples.
        """
        
        return []
    
    @abstractmethod
    def fit_block(self,data,**fit_params):
        """
        Parameters
        ----------
        data : list of DataBlock

        fit_params : dict
            Params for process block alg.

        Returns
        -------
        None
        """
        return
    
    @abstractmethod
    def process_block(self, data, **predict_params):
        """
        Parameters
        ----------
        data : list of DataBlock

        predict_params : dict 
            Parameters for process block alg.
        
        Returns
        -------
        list of DataBlock
        """
        return


class BaseNetwork(_BaseComposition, Block):
    """
    Todo
    ----
    Process standard methods for networks with single input and output.
    Theese are: fit, predict, and so on.
    
    Write process_tensors method.
   
    Note
    ----
    Network class supports two types of build methods - 
    dynamic and static.
    dynamic_build is True - block.__call__ method create nodes and 
    execute block.process.
    Inputs MetaBlock can set up dynamic_build value, but cannot change
    value that set once.
    """

    def __init__(self,blocks,inputs,outputs,n_inputs=0,n_outputs=0,
                 dynamic_build = None):
        self.blocks = blocks
        
        Block.__init__(self,n_inputs=n_inputs,n_outputs=n_outputs)
        
        # list of sorted nodes ( execution order sorting )
        self.steps_ = None
        # list of data nodes
        self.data_nodes_ = None
        
        # graph structure
        self.inputs = inputs
        self.outputs = outputs
        
        # build type
        self.dynamic_build = dynamic_build
        
        return
    
    def get_params(self, deep=True):
        return self._get_params('blocks',deep=deep)
    
    def set_params(self, **kwargs):
        return self._set_params('blocks',**kwargs)
    
    def set_dynamic_build(self, dynamic_build):
        """
        Note
        ----
        Function to set dynamic_build only once.
        """
        
        if self.dynamic_build is None:
            self.dynamic_build = dynamic_build
        else:
            assert self.dynamic_build == dynamic_build
            
        return
    
    def set_inputs(self,inputs):
        self.inputs = inputs
        return
    
    def set_outputs(self,outputs):
        self.outputs = outputs
        return
   
    def _read_nodes(self):
        """
        Note
        ----
        Sort NetworkNodes in execution order and creates list of data_nodes.
        """
        _wave = set(self.inputs)
        
        if len(_wave) != len(self.inputs):
            raise Exception("Network inputs have equal inputs.")
        
        self.data_nodes_ = data_nodes_ = []
        self.steps_ = steps_ = []
        
        while len(_wave)>0:
            _visited = []
            _new_wave = set()
            for _node in _wave:
                if _node.mark:
                    raise Exception("Cycle found: "+str(_node))
                    
                if _node.is_data():
                    _visited.append(_node)
                    data_nodes_.append(_node)
                    
                    if _node.outputs is not None:
                        _new_wave = _new_wave | set(_node.outputs)
                    
                else:
                    # _node - is block
                    if reduce(lambda a,x: x.mark and a,_node.inputs,True):
                        _visited.append(_node)
                        steps_.append(_node)
                        
                        if _node.outputs is not None:
                            _new_wave = _new_wave | set(_node.outputs)
                        
                    else:
                        _new_wave.add(_node)
            
            for _node in _visited:
                _node.set_mark()
                        
            _wave = _new_wave
                        
        for _node in data_nodes_:
            _node.reset_mark()
            
        for _node in steps_:
            _node.reset_mark()
        
        return
   
    def _clear_data_nodes(self):
        for _node in self.data_nodes_:
            _node.clear_data()
        return
    
    def _forward_network(self,data,fit=False):
        """
        Parameters
        ----------
        input_data : list of DataBlock
        """
        
        self._clear_data_nodes()
        
        for _inp_node,_inp_val in six.itertools.izip(self.inputs,data):
            _inp_node.data = DataBlock(_inp_val)
            
        for _node in self.steps_:
            _node.process(self,fit=fit)
            
        return
    
    def fit(self,X,y,**fit_params):
        """
        """
        assert self.n_inputs==1 and self.n_outputs==1
        self.fit_block([{"X":X,"y":y}],**fit_params)
        return
    
    def predict(self,X,**predict_params):
        """ 
        """
        assert self.n_inputs==1 and self.n_outputs==1
        _res = self.process_block([{"X":X}],**predict_params)
        return _res[0]["prediction"]

    def _build_network(self,data,**build_params):
        """
        Method provides algorithm for dynamic network build.
        """
        return

    def fit_block(self,data, **fit_params):
        """
        """
        self._read_nodes()
        self._forward_network(data,fit=True)
        self._clear_data_nodes()
        return
    
    def process_block(self,data, **process_params):
        self._read_nodes()
        self._clear_data_nodes()
        self._forward_network(data,fit=False)
        _result = [_el.data for _el in self.outputs]
        self._clear_data_nodes()
        return _result

    def process_tensors(self,tensors,manipulations=None):
        """
        """
        raise NotImplementedError
        return
    
    def create_keras_io_tensors(self,input_shape):
        """
        Parameters
        ----------
        inputs_shape : list of <keras shape>
            <keras shape> - tuple or list of ints.
        """
        from keras import layers
        
        tens_dict = {}
        for _inp,_inp_shape in six.itertools.izip(self.inputs,input_shape):
            tens_dict[_inp] = layers.Input(shape=_inp_shape)
        
        for _node in self.steps_:
            _node.process_tensors(self,tens_dict)
            
        inputs=[tens_dict[_el] for _el in self.inputs]
        outputs=[tens_dict[_el] for _el in self.outputs]

        return inputs,outputs

    def create_svg(self,verbose=1):
        """
        Parameters
        ----------
        verbose : int
            Set labels form.

        Returns
        -------
        str
            string with SVG image data.
        """
        
        import networkx

        def get_node_label(_node):
            if _node.is_block():
                if verbose==0:
                    return str(_node)
                elif verbose==1:
                    return str(self.blocks[_node.block].__class__.__name__)
                elif verbose==2:
                    return str(self.blocks[_node.block])
            else:
                return "Data"
            
            return

        gr = networkx.DiGraph()
        wave = self.inputs
        while len(wave):
            _new_wave = []
            for _el in wave:
                gr.add_node(repr(_el),label=get_node_label(_el))
                for _el2 in _el.outputs:
                    gr.add_node(repr(_el2),label=get_node_label(_el2))
                    gr.add_edge(repr(_el),repr(_el2))
                _new_wave.extend(_el.outputs)
            
            wave = _new_wave
            
        return networkx.nx_pydot.to_pydot(gr).create_svg()


class Input(MetaBlock):
    def __init__(self,n_inputs=1):
        assert n_inputs >= 0            
        self.n_inputs = n_inputs
        return
    
    def __call__(self,network,data=None):
        network.set_dynamic_build(data is not None)
        if data is None:
            data = [DataBlock() for _i in range(self.n_inputs)]
        elif not isinstance(data,list):
            data = [data]
            
        _inputs = [_NetworkNode(data=data[_i],inputs=[]) for _i in range(self.n_inputs)]
        network.inputs.extend(_inputs)
        network.n_inputs += self.n_inputs
        return _inputs


class Output(MetaBlock):
    def __init__(self):
        return
    
    def __call__(self,network,outputs):
        if not isinstance(outputs,list):
            network.outputs.append(outputs)
            network.n_outputs += 1
        else:
            network.outputs.extend(outputs)
            network.n_outputs += len(outputs)
        
        return


class EstimatorBlock(Block):
    """
    Todo
    ----
    Add _estimator_type property
    """
    def __init__(self,estimator):
        super(EstimatorBlock,self).__init__(n_inputs=1,n_outputs=1)
        self.estimator = estimator
        
        self.estimator_ = None
        return
    
    @property
    def _estimator_type(self):
        return self.estimator._estimator_type
    
    def fit(self,X,y=None,**params):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X,y,**params)
        return self
    
    def predict(self,X,**params):
        return self.estimator_.predict(X,**params)
    
    def fit_block(self,data,**fit_params):
        data = data[0]
        estim = self.estimator_ = clone(self.estimator)
        estim.fit(data["X"],data["y"],**fit_params)
        return self
    
    def process_block(self,data,**process_params):
        out = dict(data[0])
        if data[0]["X"].shape[0] != 0:
            _pred = self.estimator_.predict(data[0]["X"],**process_params)
            out["prediction"] = _pred
        else:
            out["prediction"] = np.array([])

        return [out]

    def process_tensors(self,inputs,manipulations=None):
        """
        """
        raise NotImplementedError
        return


class Split(Block):
    """
    Todo
    ----
        
    """
    def __init__(self,estimator,n_feature_steps,
                 min_samples_split = 2,
                 min_samples_leaf = 1,
                 min_score_increase = 0.0,
                 cv_n_splits=4):
        """
        Parameters
        ----------
        estimator : sklearn estimator
        n_feature_steps : int
            Number of threshold to process for each feature.
        min_samples_split : int
            Minimum amount of samples to split data.
        min_samples_leaf : int
            Minimum amount of samples in leaf.
        min_score_increase : int
            Minimum quality increase to process split.
        cv_n_splits : int
            Amount of folds in cross-validation.
        """
        
        super(Split,self).__init__(n_inputs=1,n_outputs=None)
        self.estimator = estimator
        self.n_feature_steps = n_feature_steps
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_score_increase = min_score_increase
        self.cv_n_splits = cv_n_splits
        
        
        self._best_feature = None
        self._best_threshold = None
        self._best = {}
        return
    
    @property
    def _estimator_type(self):
        return self.estimator._estimator_type
    
    def fit_block(self,data,**fit_params):
        data = data[0]
        
        estimator = self.estimator
        n_feature_steps = self.n_feature_steps
        
        min_samples_split = self.min_samples_split
        min_samples_leaf = self.min_samples_leaf
        min_score_increase = self.min_score_increase
        
        cv_n_splits = self.cv_n_splits
        
        # search on CrossValidation - the best split
        _best_feature = None
        _best_threshold = None
        _best_quality = float('-inf')
        _best_score_1 = None
        _best_score_2 = None
        
        X = data["X"]
        y = data["y"]
        if "score" in data:
            score = data["score"]
        else:
            score = float('-inf')
        
        if X.shape[0] >= min_samples_split:
            for _cur_feature in six.moves.xrange(X.shape[1]):
                _feature_max = np.max(X[:,_cur_feature])
                _feature_min = np.min(X[:,_cur_feature])
                _step = (_feature_max - _feature_min) / ( n_feature_steps + 1 )
                for _cur_threshold in np.arange(_feature_min+_step,_feature_max,_step):
                    # check this split
                    _s1 = X[:,_cur_feature] >= _cur_threshold
                    _s2 = ~_s1
                    
                    _min_leaf_size = min(np.sum(_s2),np.sum(_s1))
                    if _min_leaf_size<cv_n_splits or \
                        _min_leaf_size<min_samples_leaf:
                        continue
                    
                    kf = KFold(n_splits = self.cv_n_splits)
                    
                    _res1 = cross_validate(self.estimator,X[_s1],y[_s1],cv=kf,return_train_score=True, error_score='raise')
                    _res2 = cross_validate(self.estimator,X[_s2],y[_s2],cv=kf,return_train_score=True, error_score='raise')
                    _qual = np.average(_s1)*np.average(_res1["test_score"])+\
                        np.average(_s2)*np.average(_res2["test_score"])
                    
                    if _qual > _best_quality:
                        _best_quality = _qual
                        _best_feature = _cur_feature
                        _best_threshold = _cur_threshold
                        _best_score_1 = np.average(_res1["test_score"])
                        _best_score_2 = np.average(_res2["test_score"])
            
            if _best_quality - score < min_score_increase:
                _best_feature = None
                _best_threshold = None
        else:
            pass
        
        self._best_feature = _best_feature
        self._best_threshold = _best_threshold
        self._best["score_1"] = _best_score_1
        self._best["score_2"] = _best_score_2
        
        if _best_feature is None or _best_threshold is None:
            self.n_outputs = 1
        else:
            self.n_outputs = 2
        
        return self
    
    def process_tensors(self,tensors,manipulations=None):
        """
        Parameters:
        tensors : list of tuples or tensors
        """
        import keras.backend as K
        
        assert len(tensors) == 1
        
        out = []
        if self._best_feature is not None and self._best_threshold is not None:
            inp = tensors[0]
            if isinstance(inp,(tuple,list)):
                mask = inp[1]
                inp = inp[0]
                cond1 = K.greater_equal(inp[:,self._best_feature],self._best_threshold)
                cond2 = K.less(inp[:,self._best_feature],self._best_threshold)
                mask1 = mask*K.switch(cond1,K.ones_like(cond1,dtype='float32'),K.zeros_like(cond1,dtype='float32'))
                mask2 = mask*K.switch(cond2,K.ones_like(cond1,dtype='float32'),K.zeros_like(cond1,dtype='float32'))
                out = [(inp,mask1),(inp,mask2)]                
            else:
                cond1 = K.greater_equal(inp[:,self._best_feature],self._best_threshold)
                cond2 = K.less(inp[:,self._best_feature],self._best_threshold)
                mask1 = K.switch(cond1,K.ones_like(cond1,dtype='float32'),K.zeros_like(cond1,dtype='float32'))
                mask2 = K.switch(cond2,K.ones_like(cond1,dtype='float32'),K.zeros_like(cond1,dtype='float32'))
                out = [(inp,mask1),(inp,mask2)]
        else:
            out.append(tensors[0])
        
        return out
    
    def process_block(self,data,**process_params):
        """
        Note
        ----
        X key required. Works with subset key.
        Keys: score, subset, X, y, prediction.
        """
        
        data = data[0]
        
        subset = data.get("subset",np.arange(data["X"].shape[0]))
        
        # supported keys
        _sup_keys = ["X","y","prediction"]
        
        output = []
                
        # split all supported keys
        if self._best_feature is not None and self._best_threshold is not None:
            X = data["X"]
            _s1 = X[:,self._best_feature] >= self._best_threshold
            _s2 = ~_s1
            _best = self._best
            
            # copy all fields to outputs
            _o1 = dict(data)
            _o2 = dict(data)
            output.extend([_o1,_o2])
            
            _o1["score"] = self._best["score_1"]
            _o2["score"] = self._best["score_2"]
            
            _o1["subset"] = subset[_s1]
            _o2["subset"] = subset[_s2]
            
            # update processed fields
            for _key in _sup_keys:
                if _key in data:
                    _o1[_key] = data[_key][_s1]
                    _o2[_key] = data[_key][_s2]
        
        else:
            output.append(dict(data))
        
        return output


class SubsetReduce(Block):
    def __init__(self):
        super(SubsetReduce,self).__init__(n_inputs=None,n_outputs=1)
        return

    def fit_block(self, data, **fit_params):
        # empty method
        return
    
    def process_block(self, data, **process_params):
        """
        subset key required.
        Keys: X,y, prediction, subset.
        """
        
        _sup_keys = ["X","y","prediction","subset"]
        
        output = [dict(data[0])]
        _o = output[0]

        subset = np.concatenate([_el["subset"] for _el in data],axis=0)
        _ord = np.argsort(subset)
        
        for _key in _sup_keys:
            if _key in data[0]:
                _o[_key] = np.concatenate([_el[_key] for _el in data],axis=0)[_ord]
        
        return output
    
    def process_tensors(self,tensors,manipulations=None):
        import keras.backend as K
        return [K.sum([_el[0]*_el[1] for _el in tensors],axis=0)]


class MGUALinearRegressionBlock(EstimatorBlock):
    def __init__(self,complexity,max_buf_size,max_corr_coef):
        super(MGUALinearRegressionBlock,self).__init__(
            MGUA(
                LinearRegression(),complexity,
                max_buf_size = max_buf_size,
                max_corr_coef = max_corr_coef))
        
        self.complexity = complexity
        self.max_buf_size = max_buf_size
        self.max_corr_coef = max_corr_coef
        
        return
    
    def process_tensors(self,tensors,manipulations=None):
        """
        """
        import keras.backend as K
        
        assert len(tensors)==1
        tensors = tensors[0]
        
        params = np.array(self.estimator_._buffer["buffer"][0]["params"],dtype=np.int32)
        coef = self.estimator_.best_estimator_.coef_.reshape((-1,1))
        intercept = self.estimator_.best_estimator_.intercept_
        
        if isinstance(tensors,(list,tuple)):
            inp = tensors[0]
            inp = K.stack([inp[:,_param] for _param in params],axis=1)
            var = K.variable(coef,dtype='float32')
            out = K.dot(inp,var)
            out = K.reshape(out,(-1,))
            out = out + intercept*K.ones_like(out,dtype='float32')
            out = [out]
            out.extend(tensors[1:])
        else:
            inp = tensors
            inp = K.stack([inp[:,_param] for _param in params],axis=1)
            out = K.dot(inp,K.variable(coef,dtype='float32'))
            out = K.reshape(out,(-1,))
            out = out + intercept*K.ones_like(out,dtype='float32')
        
        return [out]


class GDT(BaseNetwork):
    """
    """
    
    def __init__(self,split_block,estimator_block,reduce_block,
                abs_prune_loss = 0.05):
        """
        """
        
        super(GDT,self).__init__([],[],[],n_inputs=1,n_outputs=1,
                                dynamic_build=True)
        
        self.split_block = split_block
        self.estimator_block = estimator_block
        self.reduce_block = reduce_block
        
        self.abs_prune_loss = abs_prune_loss
        
        return
    
    def _build_network(self,data,**build_params):
        """
        """        
        
        self.n_inputs = 0
        self.n_outputs = 0
        
        split_block = self.split_block
        estimator_block = self.estimator_block
        
        # build network
        inp = Input()(self,data)
        wave = list(inp)
        
        leafs = []
        
        while len(wave)>0:
            _new_wave = []
            for _el in wave:
                _split = clone(split_block)
                _split_blk, _outs = _split(self,_el)
                
                if len(_outs)>1:
                    # we have split
                    _new_wave.extend(_outs)
                else:
                    # we have no split
                    _split_blk.delete()
                    leafs.append(_el)
                    
            wave = _new_wave
        
        # calculate score and real score
        # do not recalculate score
        def _calc_real_score(node):
            if "real_score" in node.data:
                return
            if len(node.outputs)==0:
                node.data["real_score"] = node.data["score"]
            else:
                _dn1 = node.outputs[0].outputs[0]
                _dn2 = node.outputs[0].outputs[1]
                _d1 = _dn1.data
                _d2 = _dn2.data
                _calc_real_score(_dn1)
                _calc_real_score(_dn2)
                node.data["real_score"] = _d1["real_score"]*np.average(_d1["subset"])+\
                    _d2["real_score"]*np.average(_d1["subset"])
            return
        
        _calc_real_score(inp[0])
        
        # prune network with respect to score
        leafs = []
        wave = list(inp)
        while len(wave)>0:
            _new_wave = []
            for _el in wave:
                if "score" in _el.data and \
                    (_el.data["real_score"]-_el.data["score"] < self.abs_prune_loss):
                    # delete block manipulation node
                    if len(_el.outputs)>0:
                        _el.outputs[0].delete()
                    leafs.append(_el)
                else:
                    _new_wave.extend(_el.outputs[0].outputs)
            
            wave = _new_wave
        # apply estimator_block to leafs
        est_outputs = []
        for _el in leafs:
            _est = clone(estimator_block)
            _est_blk, _outs = _est(self,_el)
            est_outputs.extend(_outs)
        
        # apply reduce_block 
        _red = clone(self.reduce_block)
        _red_blk,_out = _red(self,est_outputs)
        
        Output()(self,_out)
        
        return
    
    def fit_block(self,data, **fit_params):
        self._build_network(data)
        self._read_nodes()
        self._clear_data_nodes()
        return



