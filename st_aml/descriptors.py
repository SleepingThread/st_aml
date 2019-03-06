import numpy as np
import sys

from . import __version__

_aml_descriptors_types_list = []
_descriptors_types_list = list(_aml_descriptors_types_list)

class Descriptors(object):
    def __init__(self):
        self.__version__ = __version__
        self.params = {}
        self.names = []
        self.types = []
        return
    
    def __len__(self):
        return len(self.names)
    
    @staticmethod
    def getTypes():
        return _descriptors_types_list
    
    def calculate(self,objects,verbose=0):
        """
            Returns:    if objects - list: list of lists
                        else: list
        """
        if isinstance(objects,list):
            val_list = []
            if verbose>0:
                sys.stdout.write("\r0 from "+str(len(objects)))
            for _ind,obj in enumerate(objects):
                val_list.append(self.calculate(obj))
                if verbose>0:
                    sys.stdout.write("\r"+str(_ind+1)+" from "+str(len(objects)))

            return val_list
        else:
            return self._calculate(objects)
        
    def _calculate(self,obj):
        raise Exception("Method not implemented")
        return None


class Description(Descriptors):
    def __init__(self,descriptors):
        super(Description,self).__init__()
        self.descriptors = list(descriptors)
        self.bonds = []
        
        # fill types, names, bonds
        for _descr in self.descriptors:
            self.names.extend(_descr.names)
            self.types.extend(_descr.types)
            self.bonds.append(len(_descr))
            
        self.mask = np.ones((len(self),),dtype=np.bool)
        
        return
    
    def _calculate(self,obj):
        _values = []
        for _descr in self.descriptors:
            _values.extend(_descr.calculate(obj))
        return _values
   
    def _build_masked_description(self):
        descrn = Description(self.descriptors)
        descrn.types = np.asarray(list(self.types))[self.mask]
        descrn.names = np.asarray(list(self.names))[self.mask]

        _prev_ind = 0
        _n_del = []
        for _el in self.bonds:
            _n_del.append(np.sum(~self.mask[_prev_ind:_el]))
        
        descrn.bonds = np.asarray(self.bonds)-np.asarray(_n_del)
        
        return descrn
    
    def remove_nan_descriptors(self,descr):
        arr = np.asarray(descr,dtype=np.float)
        for i in range(arr.shape[1]):
            if not np.all(~np.isnan(arr[:,i])):
                # remove i-th column
                self.mask[i] = False

        return arr[:,self.mask],self._build_masked_description()
    
    # add many feature engineering things
    def check_normal(self):
        return
    
    def analyze_descriptors(self):
        return
    
    def process_descriptor(self):
        """
            Apply function or something like this
        """
        return
