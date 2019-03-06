from . import __version__

_aml_object_types_list = []
_object_types_list = list(_aml_object_types_list)


class Object(object):
    def __init__(self):
        self.type = None
        #self.target = target
        return
    
    @staticmethod
    def getTypes():
        return _object_types_list
