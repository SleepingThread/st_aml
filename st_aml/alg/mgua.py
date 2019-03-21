"""
Examples
--------
To select 3 MGUA descriptors use the following code:

.. code-block:: python
    :linenos:

    MGUASelectDescriptors(mat,[_el[0] for _el in ds.targets],3,verbose=1)

End of examples.

"""

from . import *

import numpy as np
from scipy.sparse import issparse
from sklearn.base import clone
from sklearn.utils.validation import check_X_y, check_array
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate

class MGUACVQualPredic(object):
    """
    Note
    ----
    qual_predict call-class for MGUAFunc class. \

    Returns
    -------
    dict
        { "quality" : <averaged over folds quality>, \
    "prediction" : <concatenated predictions of all cv-folds>}
    """

    def __init__(self,ml_alg):
        """
        Parameters
        ----------
        ml_alg : sklearn - trainer
            Trainer to use with KFold cross validation.
        """
        self.ml_alg = ml_alg
        return
    
    def __call__(self,mat,targets):
        """
        """
        n_splits = 4
        kf = KFold(n_splits = n_splits)
        _res = cross_validate(self.ml_alg,mat,targets,cv=kf,return_estimator=True,return_train_score=True, error_score='raise')
        quals = []
        preds = []
        for _ind in range(n_splits):
            quals.append(_res["test_score"][_ind])
        for _ind,(train_mat,test_mat) in enumerate(kf.split(mat)):
            preds.append(_res["estimator"][_ind].predict(mat[test_mat]))
        return {"quality":np.mean(quals),"prediction":np.concatenate(preds)}


class MGUASelect(object):
    """
    Note
    ----
    Class implements method select to use with methods like \
    st_aml.alg.GreedySequenceSearch and \
    st_aml.alg.GreedySequenceSearchDownward.
    """
    def __init__(self,max_buf_size,max_corr_coef):
        """
        Parameters
        ----------
        max_buf_size : int
            Maximum number of buffer elemenets to select.
        max_corr_coef : float
            Maximum correlation threhsold between predictions.
        """
        self.max_buf_size = max_buf_size
        self.max_corr_coef = max_corr_coef
        return
    
    def select(self,buf):
        new_buf = []
        max_buf_size = self.max_buf_size
        max_corr_coef = self.max_corr_coef
        buf.sort(cmp=lambda x,y: int(x["quality"]<y["quality"])-int(x["quality"]>y["quality"]))
        
        # normalize predictions
        for el in buf:
            _pred = el["prediction"]
            _pred /= np.linalg.norm(_pred-np.average(_pred))
        
        if len(buf)>0:
            sel_predictions = [buf[0]["prediction"]]
            new_buf.append(buf[0])
        
        for el in buf[1:]:
            if len(new_buf)>=max_buf_size:
                break
                
            if np.max(np.dot(np.asarray(sel_predictions),el["prediction"])) < max_corr_coef:
                new_buf.append(el)
                sel_predictions.append(el["prediction"])
            
        return new_buf


class MGUAFunc(object):
    """
    Note
    ----
    Class implements method func for usage with methods like \
    st_aml.alg.GreedySequenceSearch and \
    st_aml.alg.GreedySequenceSearchDownward.
    """

    def __init__(self,qual_predic,mat,targets):
        """
        Parameters
        ----------
        qual_predict : function
            Inputs - mat, targets. Function must return\
            dict:\
            {"quality":<quality>, "prediction": <prediction>}.
        mat : numpy.ndarray
            Feature matrix. Rows corresponds to objects.
        targets : numpy.ndarray
            Target vector.
        """
        self.mat = mat
        self.targets = targets
        self.qual_predic = qual_predic
        return
    
    def func(self,params):
        """
        """

        params = list(params["params"])
        mat = self.mat[:,np.asarray(params)]
        out = self.qual_predic(mat,self.targets)
        return {"params": params,
                "quality":out["quality"],
                "prediction":out["prediction"]}


def MGUASelectDescriptors(mat,targets,
    complexity,ml_alg=LinearRegression(),
    max_buf_size=20,max_corr_coef=0.8,
    n_jobs = 1, verbose=0):    
    """
    Parameters
    ----------
    mat : numpy.ndarray
        Feature matrix.
    targets : numpy.ndarray
        Target matrix.
    complexity : int
        Maximum amount of features to select.
    ml_alg : sklearn-trainer
        Trainer to select features.
    n_jobs : int
        Amount of threads to use.
    """

    n_descr = mat.shape[1]

    qual_predic = MGUACVQualPredic(ml_alg)

    mgua_func = MGUAFunc(qual_predic,mat,targets)
    mgua_select = MGUASelect(max_buf_size,max_corr_coef)
    
    mgua_step = GreedySetSearch(
        func = mgua_func,
        select = mgua_select,
        right_bond = n_descr,
        n_jobs = n_jobs
    )

    return StepWise(
        step = mgua_step,
        initializer = Buffer(),
        stop_condition = BufferSequenceSize(min(n_descr,complexity)),
        verbose = verbose
    )

import six
from abc import ABCMeta
from sklearn.base import BaseEstimator, MetaEstimatorMixin

class MGUA(six.with_metaclass(ABCMeta, BaseEstimator,
                                      MetaEstimatorMixin)):
    """
    Note
    ----
    Implements MGUA as sklearn-trainer.
    """
    def __init__(self,estimator,complexity,
        max_buf_size=20,max_corr_coef=0.8):
        """
        """
        self.estimator = estimator
        self.complexity = complexity
        self.max_buf_size = max_buf_size
        self.max_corr_coef = max_corr_coef
        
        # other parameters
        self._buffer = None
        self.best_estimator_ = None
        return

    def fit(self,X,y=None,verbose=0,n_jobs=1):
        """
        Note
        ----
        MGUA fit method
        """

        X,y = check_X_y(X,y,accept_large_sparse=False)

        # initialize _buffer
        self._buffer = MGUASelectDescriptors(X,y,
            self.complexity,ml_alg=self.estimator,
            max_buf_size=self.max_buf_size,
            max_corr_coef=self.max_corr_coef,
            n_jobs = n_jobs,
            verbose=verbose)

        _param = self._buffer["buffer"][0]["params"]
        self.best_estimator_ = clone(self.estimator)
        self.best_estimator_.fit(X[:,_param],y)

        return self 

    def predict(self,X):
        """
        """

        _param = self._buffer["buffer"][0]["params"]
        
        X = check_array(X,accept_large_sparse=False)
        
        return self.best_estimator_.predict(X[:,_param])

    @property
    def _estimator_type(self):
        return self.best_estimator_._estimator_type

    def score(self,X,y=None):
        if self._buffer is None:
            raise Exception("Trainer not fitted.")
        _param = self._buffer["buffer"][0]["params"]
        return self.best_estimator_.score(X[:,_param],y)

