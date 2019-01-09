"""
Some common interfaces for event/client types.

Note: These are used instead of the ones in obspy.clients.base so the subclass
hooks can be used.
"""
from abc import ABC
import obspy


def add_func_name(my_set: set):
    """ add the name of a function to a set """

    def _wrap(func):
        my_set.add(func.__name__)
        return func

    return _wrap


class _MethodChecker:
    """ class for checking if methods exist """

    required_methods: set = set()

    @classmethod
    def __subclasshook__(cls, C):
        methods = {x for B in C.__mro__ for x in B.__dict__}
        if cls.required_methods.issubset(methods):
            return True
        else:
            return NotImplemented


class EventClient(ABC, _MethodChecker):
    """ The event client interface """

    required_methods = {"get_events"}


class WaveformClient(ABC, _MethodChecker):
    """ The waveform client interface"""

    required_methods = {"get_waveforms"}


class StationClient(ABC, _MethodChecker):
    """ The station client interface """

    required_methods = {"get_stations"}


class BankType(ABC, _MethodChecker):
    """ an object that looks like a bank """

    required_methods = {"read_index"}


# register virtual subclasses
WaveformClient.register(obspy.Stream)
EventClient.register(obspy.Catalog)
StationClient.register(obspy.Inventory)
