"""
Class for chaining clients together
"""
from typing import Sequence

import obspy
from obspy import Stream

from obsplus.constants import waveform_request_type
from obsplus.utils.misc import iterate


def empty_inventory():
    return obspy.Inventory([], "unknown")


class ChainClient:
    """
    ClientChain is a collection of clients with prioritized interfaces.

    It is similar to the ChainMap from the collection module, but for
    obspy clients.

    Parameters
    -------------
    waveform_clients
        A single clients or collections of Clients that are used only
        for getting waveforms. Must have get_waveforms method.
    event_clients
        A single clients or collections of Clients that are used only
        for getting events. Must have get_events method.
    station_clients
        A single clients or collections of Clients that are used only
        for getting stations. Must have get_stations method.
    """

    def __init__(self, waveform_clients=None, event_clients=None, station_clients=None):
        self.waveform_clients = iterate(waveform_clients)
        self.event_clients = iterate(event_clients)
        self.station_clients = iterate(station_clients)

    # --- get functions

    def get_events(self, *args, **kwargs) -> obspy.Catalog:
        """
        Attempt to get a events from clients.

        First try all event_clients then resort to general clients.

        See obspy.clients.fdsn.Client.get_events for supported arguments.
        """
        return self._get_object("event", *args, **kwargs)

    def get_waveforms(self, *args, **kwargs) -> obspy.Stream:
        """
        Attempt to get a waveforms from the clients clients.

        First try all waveform_clients then resort to general clients.

        See obspy.clients.fdsn.Client.get_waveforms for supported arguments.
        """
        return self._get_object("waveform", *args, **kwargs)

    def get_waveforms_bulk(self, bulk: Sequence[waveform_request_type]):
        """
        Make a bulk request for waveforms.

        Any exceptions raised on individual data missing will be ignored
        to allow the other data acquisitions to proceed.

        Parameters
        ----------
        bulk
            A sequence of sequences containing network, station, location,
            channel, starttime, endtime. Eg:
            time = UTCDateTime('2017-01-01')
            bulk = [
                ('UU', 'TMU', '*', 'HHZ', time, time + 10),
                ('UU', 'COY', '*', 'HHZ', time, time + 10),
            ]

        Returns
        -------
        A waveforms
        """
        return self._get_object_bulk("waveform", bulk)

    def get_stations(self, *args, **kwargs):
        """
        Attempt to get an stations from clients.

        First try all station_clients then resort to general clients.

        See obspy.clients.fdsn.Client.get_stations for supported arguments.
        """
        return self._get_object("station", *args, **kwargs)

    def get_stations_bulk(self, bulk: Sequence[waveform_request_type]):
        """
        Make a bulk request for station information.

        Any exceptions raised on individual data missing will be ignored
        to allow the other data acquisitions to proceed.

        Parameters
        ----------
        bulk
            A sequence of sequences containing network, station, location,
            channel, starttime, endtime. Eg:
            time = UTCDateTime('2017-01-01')
            bulk = [
                ('UU', 'TMU', '*', 'HHZ', time, time + 10),
                ('UU', 'COY', '*', 'HHZ', time, time + 10),
            ]

        Returns
        -------
        A waveforms
        """
        return self._get_object_bulk("waveform", bulk)

    # --- misc

    def _iter_client(self, clients, method_name, *args, **kwargs):
        """
        Iterate clients, return any values that aren't None.

        Return None if no data is found.
        """
        for cli in iterate(clients):
            try:
                out = getattr(cli, method_name)(*args, **kwargs)
            except AttributeError:  # client doesn't have required method.
                continue
            # if a non-empty object was obtained return it
            if out is not None and len(out):
                return out

    def _get_object(self, client_type, *args, **kwargs):
        """
        Iterate client type and try methods until a non-empty is returned.
        """
        assert client_type in {"waveform", "event", "station"}
        method = "get_" + client_type + "s"
        # iterate specialized clients first
        specials = getattr(self, client_type + "_clients")
        out = self._iter_client(specials, method, *args, **kwargs)
        if out is not None and len(out):
            return out
        # else try normal clients
        if out is None or not len(out):
            raise ValueError("No data returned")
        return out

    def _iter_client_bulk(self, clients, method, bulk_method, out, bulk):
        """iterate over clients trying to get bulk requests."""
        for client in clients:
            if hasattr(client, bulk_method):
                out += getattr(client, bulk_method)(bulk)
            elif hasattr(client, method):
                for b in bulk:
                    try:
                        out += getattr(client, method)(*b)
                    except:
                        pass
            else:
                continue
        if out is not None and len(out):
            return out

    def _get_object_bulk(self, client_type, bulk):
        """
        Iterate clients and get bulk requests. If client doesn't have a bulk
        request option iterate over bulk arguments and aggregate.
        """
        assert client_type in {"waveform", "station"}
        method = "get_" + client_type + "s"
        bulk_method = method + "_bulk"
        out = Stream() if "waveform" in method else empty_inventory()
        # iterate specialized clients first
        specials = getattr(self, client_type + "_clients")
        self._iter_client_bulk(specials, method, bulk_method, out, bulk)
        # then try normal clients
        if out is None or not len(out):
            raise ValueError("No data returned")
        return out

    def __getattr__(self, item):
        """
        Search for special attrs on clients
        """
        for client_type in ("waveform_clients", "event_clients", "station_clients"):
            clients = self.__dict__.get(client_type, [])  # avoids recursion issue
            for cli in clients:
                if hasattr(cli, item):
                    return getattr(cli, item)
        raise AttributeError(f"No client have attribute {item}")
