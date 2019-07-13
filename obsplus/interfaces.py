"""
Some common interfaces for event/client types.

Note: These are used instead of the ones in obspy.clients.base so the subclass
hooks can be used.
"""
from abc import ABC, abstractmethod
import obspy


def add_func_name(my_set: set):
    """ add the name of a function to a set """

    def _wrap(func):
        my_set.add(func.__name__)
        return func

    return _wrap


class _MethodChecker(type):
    """ class for checking if methods exist """

    required_methods: set = set()

    def __subclasshook__(cls, C):
        methods = {x for B in C.__mro__ for x in B.__dict__}
        if cls.required_methods.issubset(methods):
            return True
        return NotImplemented

    # def __instancecheck__(cls, instance):
    #     if set(cls.required_methods).issubset(dir(instance)):
    #         return True
    #     return super().__instancecheck__(cls, instance)
    #     # return NotImplemented


class EventClient(metaclass=_MethodChecker):
    """ The event client interface """

    required_methods = {"get_events"}


class WaveformClient(metaclass=_MethodChecker):
    """ The waveform client interface"""

    required_methods = {"get_waveforms"}


class StationClient(metaclass=_MethodChecker):
    """ The station client interface """

    required_methods = {"get_stations"}


class BankType(metaclass=_MethodChecker):
    """ an object that looks like a bank """

    required_methods = {"read_index"}


class ProgressBar(metaclass=_MethodChecker):
    """
    A class that behaves like the progressbar2.ProgressBar class.
    """

    required_methods: set = {"update", "finish"}

    @abstractmethod
    def update(self, value=None, force=False, **kwargs):
        """ Called when updating the progress bar. """

    @abstractmethod
    def finish(self, **kwargs):
        """ Puts the progress bar in the finished state. """


# register virtual subclasses
# WaveformClient.register(obspy.Stream)
# EventClient.register(obspy.Catalog)
# StationClient.register(obspy.Inventory)
