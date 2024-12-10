
from __future__ import absolute_import
from types import MethodType

#http://stackoverflow.com/questions/17486104/python-lazy-loading-of-class-attributes
lazy_property = cached_property

class JsObject(dict):
    def __init__(self, *args, **kwargs):
        super(JsObject, self).__init__(*args, **kwargs)
        self.__dict__ = self


#http://www.ianlewis.org/en/dynamically-adding-method-classes-or-class-instanc

def add_method_to_instance(obj, meth, name=None):
    if name is None:
        name = meth.__name__
    x = MethodType(meth, obj, obj.__class__)
    setattr(obj, name, x)
    return x

def add_method_to_class(cls, meth, name=None):
    if name is None:
        name = meth.__name__
    x = MethodType(meth, None, cls)
    setattr(cls, name, x)
    return x

def extend_class(base, ext, before):
    if not ext in base.__bases__:
        if before:
            base.__bases__ = (ext,) + base.__bases__
        else:
            base.__bases__ = base.__bases__ + (ext,)
    return base

