"""
Module for management of seismic events.
"""
import obsplus.events.schema as eschema
from obsplus.structures.mill import Mill, DataFramer


class EventMill(Mill):
    """
    A class for managing Seismic events.
    """

    def __init__(self, data):
        self._data = self._input_to_model(data)

    def _input_to_model(self, data):
        """Coerce the input to json."""
        if isinstance(data, dict):  # assume json if top-level is dict.
            return eschema.Catalog.from_orm(data)
        else:
            return eschema.Catalog.from_orm(data)


@EventMill.register_data_framer("events")
class EventFramer(DataFramer):
    """
    Class for extracting dataframes from event info from EventMill.
    """

    _base = eschema.Event
    # breakpoint()
    time = _base.time
    latitude = _base.latitude
    longitude = _base.longitude
    depth = _base.depth

    _pref_mag = _base.get_preferred_magnitude()

    magnitude = _pref_mag.magnitude
