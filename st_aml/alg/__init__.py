import copy
import sys
import six
from joblib import Parallel, delayed


def StepWise(initializer,step,stop_condition,verbose=0):
    """ 
    Parameters
    ----------
    initializer : class with initializer method or function
        Function to initialize state.
    step : class with step method or function 
        Function to make algorithm step.
        Changes state of algorithm.
    stop_condition : class with stop_condition method \
        or function 
        Function to make algorithm stop.

    Returns
    -------
    state : dict
        Some description

    Note
    ----
    Function implements step-wise state changing. \
    On each step it checks stop_condition for state.

    """
    if hasattr(initializer,"initializer"):
        initializer = initializer.initializer
    if hasattr(step,"step"):
        step = step.step
    if hasattr(stop_condition,"stop_condition"):
        stop_condition = stop_condition.stop_condition
    
    _counter = 0
    state = initializer()
    if verbose>0:
        sys.stdout.write("\rStep "+str(_counter))
    while not stop_condition(state):
        state = step(state)
        _counter += 1
        if verbose>0:
            sys.stdout.write("\rStep "+str(_counter))
    
    return state


class Buffer(object):
    """
    Note
    ----
    State initializer.\n
    Class for creating 'state' according with template:\
        {"buffer":list({ "params":list() })}.\n
    This template is used with greedy buffered algorithms.\n 
    """
    def __init__(self,initial_buffer=None):
        """
        Parameters
        ----------
        initial_buffer : dict
            Specify if you want to start algorithm \
            with initial_buffer.
        """
        if initial_buffer is None:
            self.initial_buffer = {"buffer":[{"params":[]}]}
        else:
            self.initial_buffer = initial_buffer
        return
    def initializer(self):
        return self.initial_buffer


class BufferSequenceSize(object):
    """
    Note
    ----
    Stop condition. \
    Check size of state["buffer"][0]["params"].
    """
    def __init__(self,max_sequence_size):
        """
        """
        self.max_sequence_size = max_sequence_size
        return
    def stop_condition(self,state):
        if len(state["buffer"][0]["params"])>=self.max_sequence_size:
            return True
        return False


class GreedySequenceSearch(object):
    """
    Note
    ----
    Construct sequences with integers from segment\
    [0 .. sequence_size-1]. Increment size of sequence.
    """
    def __init__(self,func,select,sequence_size):
        """
        Parameters
        ----------
        func: class with method func or function
            Take dict with "params" key and returns 
            the same dict with changed fields.
            Only "params" field must not be changed.
            Input template: { "params": list() }

        select: class with method select or function
            Input - list of {"params": list()}.
            Select appropriate sequences from generated one.

        sequence_size: int
            Determines segment right bond to
            search sequence from.
            I.e. sequence elements are
            from [0...sequence_size] segment.
        Note
        ----
        Generated sequences stored under "params" key.
        Each step add one element into "params".
        Params func and select must be consistent.
        """
        if hasattr(func,"func"):
            func = func.func
        if hasattr(select,"select"):
            select = select.select
        
        self.func = func
        self.select = select
        self.sequence_size = sequence_size
        return

    def step(self,state):
        """
        Parameters
        ----------
        state : dict
            Parameter must have structure like:
            {"buffer" : list( {"params":[]} )}.
        Note
        ----
        Function constructs new "buffer" with new "params".
        Each step add one element into "params".
        """
        buf = state["buffer"]
        func = self.func
        sequence_size = self.sequence_size
        _new_buffer = []
        for _el in buf:
            _param = _el["params"]
            for i in range(sequence_size):
                if i in _param:
                    continue
                #_new_el = copy.deepcopy(_el)
                _new_el = dict(_el)
                _new_el["params"] = list(_param)
                _new_el["params"].append(i)
                _new_buffer.append(func(_new_el))
        
        _new_buffer = self.select(_new_buffer)
        
        return {"buffer": _new_buffer}


class GreedySequenceSearchDownward(object):
    """
    Note
    ----
    Construct sequences with integers from segment\
    [0 .. sequence_size-1]. Decrement size of sequence.
    """
    def __init__(self,func,select,sequence_size):
        """
        Parameters
        ----------
        func: class with method func or function
            Take dict with "params" key and returns \
            the same dict with changed fields. \n
            Only "params" field must not be changed. \n
            Input template: { "params": list() }
        select: class with method select or function
            Input - list of {"params": list()}.\
            Select appropriate sequences from generated one.
        sequence_size: int
            Maximum size of sequence to search.
        Note
        ----
        Generated sequences stored under "params" key.\n
        Each step remove one element from "params".\n
        Params func and select must be consistent.
        """
        if hasattr(func,"func"):
            func = func.func
        if hasattr(select,"select"):
            select = select.select
        
        self.func = func
        self.select = select
        self.sequence_size = sequence_size
        return

    def step(self,state):
        """
        Parameters
        ----------
        state : dict
            Parameter must have structure like:\
            {"buffer" : list( {"params":[]} )}.
        Note
        ----
        Function constructs new "buffer" with new "params".\
        Each step add one element into "params".
        """
        buf = state["buffer"]
        func = self.func
        sequence_size = self.sequence_size
        _new_buffer = []
        for _el in buf:
            _param = _el["params"]
            for i in _param:
                #_new_el = copy.deepcopy(_el)
                _new_el = dict(_el)
                _new_el["params"] = list(_param)
                _new_el["params"].remove(i)
                _new_buffer.append(func(_new_el))
        
        _new_buffer = self.select(_new_buffer)
        
        return {"buffer": _new_buffer}


class GreedySetSearch(object):
    """
    Note
    ----
    Construct sequences with integers from segment\
    [0 .. right_bond-1]. Increment size of sequence.
    """
    def __init__(self,func,select,right_bond,add_new_el = False,n_jobs=1):
        """
        Parameters
        ----------
        func: class with method func or function
            Take dict with "params" key and returns 
            the same dict with changed fields.
            Only "params" field must not be changed.
            Input template: { "params": list() }

        select: class with method select or function
            Input - list of {"params": list()}.
            Select appropriate sequences from generated one.

        right_bond int
            Determines segment right bond to
            search sequence from.
            I.e. sequence elements are
            from [0...right_bond] segment.
        Note
        ----
        Generated sequences stored under "params" key.
        Each step add one element into "params".
        Params func and select must be consistent.
        """
        if hasattr(func,"func"):
            func = func.func
        if hasattr(select,"select"):
            select = select.select
        
        self.func = func
        self.select = select
        self.right_bond = right_bond
        self.add_new_el = add_new_el
        self.n_jobs = n_jobs
        return

    def step(self,state):
        """
        Parameters
        ----------
        state : dict
            Parameter must have structure like:
            {"buffer" : list( {"params":[]} )}.
        Note
        ----
        Function constructs new "buffer" with new "params".
        Each step add one element into "params".
        """
        buf = state["buffer"]
        func = self.func
        right_bond = self.right_bond
        add_new_el = self.add_new_el
        
        _new_buffer = []
        
        _args_dict = {}
        
        for _el in buf:
            # _param - set
            _param = _el["params"]
            for i in range(right_bond):
                if i in _param:
                    continue
                #_new_el = copy.deepcopy(_el)
                _new_el = dict(_el)
                _new_el["params"] = set(_param)
                _new_el["params"].add(i)
                
                if add_new_el:
                    _new_el["new_el"] = i
                
                _args_dict[frozenset(_new_el["params"])] = _new_el
                
        _new_buffer = Parallel(n_jobs=self.n_jobs)(delayed(func)(_el[1]) for _el in six.iteritems(_args_dict))
        
        _new_buffer = self.select(_new_buffer)
        
        return {"buffer": _new_buffer}
