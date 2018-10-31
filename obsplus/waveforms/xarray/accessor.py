"""
Create and register seismology specific DataArray accessors.
"""
import functools

import xarray as xr


OPS_METHODS = {}  # a dict for storing dtx attributes


def ops_method(attr_name):
    """
    Decorator for adding a function to data array accessor.

    The first argument must always be the data array
    """

    def decorator(func):
        OPS_METHODS[attr_name] = func
        return func

    return decorator


def _add_wrapped_methods(cls):
    """ add the methods marked with the ops decorator to class def """

    for name, func in OPS_METHODS.items():

        def deco(func):
            """ This is needed to bind the correct function to scope """

            @functools.wraps(func)
            def _func(self, *args, **kwargs):
                return func(self._obj, *args, **kwargs)

            return _func

        setattr(cls, name, deco(func))
    return cls


@xr.register_dataarray_accessor("ops")
class ObsplusAccessor:
    """
    Class for registering obsplus specific functionality on the data array
    """

    def __init__(self, xarray_object):
        self._obj = xarray_object
        _add_wrapped_methods(self.__class__)

    # def __getattr__(self, item):
    #     """ If the data Accessor doesn't yet have a method try to get it. """
    #     try:
    #         return getattr(self, item)
    #     except AttributeError:
    #         raise
