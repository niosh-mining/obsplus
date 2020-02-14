"""
Some common interfaces for event/client types.

Note: These are used instead of the ones in obspy.clients.base so the subclass
hooks can be used.
"""
import inspect
from abc import abstractmethod

import obspy
import pandas as pd


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
        # look for defined methods
        if is_instance and set(cls._required_methods).issubset(dir(instance)):
            return True
        # if they do not exist probe for dynamic methods  that meet criteria
        has_meths = [getattr(instance, x, False) for x in cls._required_methods]
        return all(has_meths) and is_instance


class EventClient(metaclass=_MethodChecker):
    """
    Abstract Base Class defining the event client interface.

    Any object which has a `get_events` method implements the interface
    and is either an instance or subclass of EventClient.

    Examples
    --------
    >>> import obsplus
    >>> import obspy
    >>> # EventBank is a subclass of EventClient
    >>> assert issubclass(obsplus.EventBank, EventClient)
    >>> assert issubclass(obspy.Catalog, EventClient)
    >>> # a catalog is an instance of EventClient
    >>> assert isinstance(obspy.read_events(), EventClient)
    >>> # A string has no `get_events` so it is not a subclass/instance
    >>> assert not issubclass(str, EventClient)
    >>> assert not isinstance('string', EventClient)
    >>> # this works on any class with a get_events method
    >>> class NewEventClientThing:
    ...     def get_events(self, *args, **kwargs):
    ...         pass
    >>> assert issubclass(NewEventClientThing, EventClient)
    """

    @abstractmethod
    def get_events(self, *args, **kwargs) -> obspy.Catalog:
        """ A method which must return events as obspy.Catalog object. """


class WaveformClient(metaclass=_MethodChecker):
    """
    Abstract Base Class defining the waveform client interface.

    Any object which has a `get_waveforms` method implements the interface
    and is either an instance or subclass of WaveformClient.

    Examples
    --------
    >>> import obsplus
    >>> import obspy
    >>> # WaveBank/Stream are subclasses of WaveformClient
    >>> assert issubclass(obsplus.WaveBank, WaveformClient)
    >>> assert issubclass(obspy.Stream, WaveformClient)
    >>> # A string has no `get_waveforms` so it is not a subclass/instance
    >>> assert not issubclass(str, WaveformClient)
    >>> assert not isinstance('string', WaveformClient)
    >>> # A stream is a subclass of WaveformClient
    >>> assert isinstance(obspy.read(), WaveformClient)
    >>> # this works on any class with a get_waveforms method
    >>> class NewWaveformClientThing:
    ...     def get_waveforms(self, *args, **kwargs):
    ...         pass
    >>> assert issubclass(NewWaveformClientThing, WaveformClient)
    """

    @abstractmethod
    def get_waveforms(
        self, network, station, location, channel, starttime, endtime
    ) -> obspy.Stream:
        """ A method which must return waveforms as an obspy.Stream. """


class StationClient(metaclass=_MethodChecker):
    """
    Abstract Base Class defining the station client interface.

    Any object which has a `get_stations` method implements the interface and is
    either an instance or subclass of StationClient.

    Examples
    --------
    >>> import obsplus
    >>> import obspy
    >>> # Inventory is a subclass of StationClient
    >>> assert issubclass(obspy.Inventory, StationClient)
    >>> assert isinstance(obspy.read_inventory(), StationClient)
    >>> # A string has no `get_stations` so it is not a subclass/instance
    >>> assert not issubclass(str, StationClient)
    >>> assert not isinstance('string', StationClient)
    >>> # this works on any class with a get_waveforms method
    >>> class NewStationClientThing:
    ...     def get_stations(self, *args, **kwargs):
    ...         pass
    >>> assert issubclass(NewStationClientThing, StationClient)
    """

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
