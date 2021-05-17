"""
Some common interfaces for event/client types.

Note: These are used instead of the ones in obspy.clients.base so the subclass
hooks can be used.
"""
from abc import abstractmethod
from typing_extensions import Protocol, runtime_checkable

import obspy
import pandas as pd


@runtime_checkable
class EventClient(Protocol):
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
        """A method which must return events as obspy.Catalog object."""


@runtime_checkable
class WaveformClient(Protocol):
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
    >>> # A string has no `get_waveforms` so it is not a subclass/instance
    >>> assert not issubclass(str, WaveformClient)
    >>> assert not isinstance('string', WaveformClient)
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
        """A method which must return waveforms as an obspy.Stream."""


@runtime_checkable
class StationClient(Protocol):
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
        """A method which must return an inventory object."""


@runtime_checkable
class BankType(Protocol):
    """an object that looks like a bank"""

    @abstractmethod
    def read_index(self, *args, **kwargs) -> pd.DataFrame:
        """A method which must return a dataframe of index contents."""


@runtime_checkable
class ProgressBar(Protocol):
    """
    A class that behaves like the progressbar2.ProgressBar class.
    """

    @abstractmethod
    def update(self, value=None, force=False, **kwargs):
        """Called when updating the progress bar."""

    @abstractmethod
    def finish(self, **kwargs):
        """Puts the progress bar in the finished state."""


if __name__ == "__main__":
    import doctest

    doctest.testmod()
