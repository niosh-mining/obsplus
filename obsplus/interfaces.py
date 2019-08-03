"""
Some common interfaces for event/client types.

Note: These are used instead of the ones in obspy.clients.base so the subclass
hooks can be used.
"""
import inspect
from abc import abstractmethod

import obspy
import pandas as pd


def add_func_name(my_set: set):
    """ add the name of a function to a set """

    def _wrap(func):
        my_set.add(func.__name__)
        return func

    return _wrap


class _MethodChecker(type):
    """ class for checking if methods exist """

    def __init__(cls, name, bases, arg_dict):
        # get all methods defined by class
        methods = {
            i for i, v in arg_dict.items() if not i.startswith("_") and callable(v)
        }
        cls._required_methods = methods

    def __subclasscheck__(cls, subclass):
        if not inspect.isclass(subclass):
            return False
        methods = {x for B in subclass.__mro__ for x in B.__dict__}
        if cls._required_methods.issubset(methods):
            return True
        return False

    def __instancecheck__(cls, instance):
        is_instance = not inspect.isclass(instance)
        if is_instance and set(cls._required_methods).issubset(dir(instance)):
            return True
        return False


class EventClient(metaclass=_MethodChecker):
    """ The event client interface """

    @abstractmethod
    def get_events(self, *args, **kwargs) -> obspy.Catalog:
        """ A method which must return events as obspy.Catalog object. """


class WaveformClient(metaclass=_MethodChecker):
    """ The waveform client interface. """

    @abstractmethod
    def get_waveforms(
        self, network, station, location, channel, starttime, endtime
    ) -> obspy.Stream:
        """ A method which must return waveforms as an obspy.Stream. """


class StationClient(metaclass=_MethodChecker):
    """ The station client interface """

    @abstractmethod
    def get_stations(self, *args, **kwargs) -> obspy.Inventory:
        """ A method which must return an inventory object. """


class BankType(metaclass=_MethodChecker):
    """ an object that looks like a bank """

    @abstractmethod
    def read_index(self, *args, **kwargs) -> pd.DataFrame:
        """ A method which must return a dataframe of index contents. """


class ProgressBar(metaclass=_MethodChecker):
    """
    A class that behaves like the progressbar2.ProgressBar class.
    """

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
