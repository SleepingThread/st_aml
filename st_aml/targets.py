from collections import OrderedDict

_target_types_list = \
[
    
]

class Target(object):
    def __init__(self):
        self.type = None
        return
    
    @staticmethod
    def getTypes():
        return _target_types_list


class GroupTarget(Target):
    def __init__(self):
        super(GroupTarget,self).__init__()
        self.type = None
        self.group_dict = {}
        self.rgroup_dict = {}
        return

    def fit(self,targets):
        unique_targets = list(set(targets))
        unique_targets.sort()
        self.group_dict = \
            OrderedDict([(_el,_ind) \
                for _ind,_el in enumerate(unique_targets)])

        self.rgroup_dict = \
            OrderedDict([(_ind,_el) \
                for _el,_ind in self.group_dict.iteritems()])

        return

    def transform(self,targets):
        return [self.group_dict[_el] for _el in targets]

    def fit_transform(self,targets):
        self.fit(targets)
        return self.transform(targets)

    def rtransform(self,targets):
        return [self.rgroup_dict[_el] for _el in targets]


class IntervalTarget(Target):
    def __init__(self):
        super(IntervalTarget,self).__init__()
        self.type = None
        return

    def fit(self,targets):
        return

    @staticmethod
    def from_hr(val):
        """
            From human-readable format
        """
        if isinstance(val,str):
            if ">" in val or ">=" in val:
                val = val.strip(" ><=")
                res = (float(val),float("+inf"))
            elif "<" in val or "<=" in val:
                val = val.strip(" ><=")
                res = (float("-inf"),float(val))
            else:
                res = (float(val),float(val))
            return res  
        else:
            return val
        return

    @staticmethod
    def to_hr(val):
        """
            To human-readable format
        """
        if isinstance(val,str):
            return val
        else:
            if val[1]==float("+inf"):
                return ">"+str(val[0])
            elif val[0]==float("-inf"):
                return "<"+str(val[1])
        return

    def transform(self,targets):
        return [self.from_hr(_el) for _el in targets]

    def fit_transform(self,targets):
        self.fit(targets)
        return self.transform(targets)

    def rtransform(self,targets):
        return [self.to_hr(_el) for _el in targets]

# class TimeSeries(Target)
# class GroupTarget - classification
# class RealTarget - regression
# class RealVectorTarget - multiple regressions
