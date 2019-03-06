import copy
import sys


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
        Each step add one element into "params".\n
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
