from sklearn import model_selection 
import copy

class Dataset(object):
    def __init__(self):
        self.params = {}
        self.objects = []
        self.targets = []
        self.target_types = []
        return
    
    def __len__(self):
        return len(self.objects)
    
    def train_test_split(self,**options):
        options["test_size"] = 0.25
        options["shuffle"] = False
        options["random_state"] = 0
        
        train_objs, test_objs, \
        train_targets, test_targets = \
            model_selection.train_test_split(self.objects,self.targets,**options)

        train_dataset = self.__class__()
        train_dataset.objects = train_objs
        train_dataset.targets = train_targets
        train_dataset.target_types = list(self.target_types)
        train_dataset.params = copy.deepcopy(self.params)
        
        test_dataset = self.__class__()
        test_dataset.objects = test_objs
        test_dataset.targets = test_targets
        test_dataset.target_types = list(self.target_types)
        test_dataset.params = copy.deepcopy(self.params)
        
        return train_dataset, test_dataset
    
    def train_test_split_balanced(self,**options):
        options["test_size"] = 0.25
        options["shuffle"] = False
        options["random_state"] = 0
        
        unique_targets = set(self.targets)
        targets = np.asarray(self.targets)
        objects = np.asarray(self.objects)
        
        _train_objects = []
        _train_targets = []
        _test_objects = []
        _test_targets = []
        
        
        for _el in unique_targets:
            _class_objects = objects[targets==_el]
            _class_targets = targets[targets==_el]
            
            _class_train_objects, _class_test_objects, \
            _class_train_targets, _class_test_targets = \
                model_selection.train_test_split(_class_objects,_class_targets,**options)
            
            if len(_class_test_objects)==0 or len(_class_train_objects)==0:
                raise Exception("Class "+str(_el)+" not presented in one of the splits")
            
            _train_objects.extend(_class_train_objects)
            _train_targets.extend(_class_train_targets)
            
            _test_objects.extend(_class_test_objects)
            _test_targets.extend(_class_test_targets) 
            
        train_dataset = self.__class__()
        train_dataset.objects = _train_objects
        train_dataset.targets = _train_targets
        train_dataset.target_types = list(self.target_types)
        train_dataset.params = copy.deepcopy(self.params)
        
        test_dataset = self.__class__()
        test_dataset.objects = _test_objects
        test_dataset.targets = _test_targets
        test_dataset.target_types = list(self.target_types) 
        test_dataset.params = copy.deepcopy(self.params)
            
        return train_dataset, test_dataset
    
    def load(self):
        raise Exception("Method not implemented")
        
class TestDataset(Dataset):
    def __init__(self,size=10):
        super(TestDataset,self).__init__()
        self.objects = range(size)
        self.targets = np.zeros((size,),dtype=np.int32)
        self.targets[size/2:] = 1
        self.size = size
        return
