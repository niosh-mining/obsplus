"""
A Client.get_stations like interface for obspy stations objects
"""
import copy
import fnmatch
import inspect
import operator as op

import obspy
from obspy.clients.fdsn import Client
from obspy.core.inventory import Network, Channel, Station

from obsplus.utils.misc import get_instances_from_tree

UNSUPPORTED = {"latitude", "longitude", "matchtimeseries", "minradius", "maxradius"}

CLIENT_ARGS = set(inspect.signature(Client.get_stations).parameters)
SUPPORTED_ARGS = CLIENT_ARGS - UNSUPPORTED


def match(name, patern):
    """ perform case in-sensitive unix style matches with fnmatch """
    return fnmatch.fnmatch(name.upper(), patern.upper())


# dict of attrs effected by various params
PARAMS = dict(
    startbefore="start_date",
    startafter="start_date",
    starttime="start_date",
    endtime="end_date",
    endbefore="end_date",
    endafter="end_date",
    minlatitude="latitude",
    maxlatitude="latitude",
    minlongitude="longitude",
    maxlongitude="longitude",
    network="network_code",
    station="station_code",
    location="location_code",
    channel="channel_code",
)

OPERATORS = dict(
    startbefore=op.lt,
    startafter=op.gt,
    starttime=op.ge,
    endtime=op.le,
    endbefore=op.lt,
    endafter=op.gt,
    minlatitude=op.gt,
    maxlatitude=op.lt,
    minlongitude=op.gt,
    maxlongitude=op.lt,
    network=match,
    station=match,
    location=match,
    channel=match,
)


def _add_codes(inv):
    """ add network, station, location, channel codes where applicable """
    for network in inv:
        net_code = network.code
        network.network_code = net_code
        for station in network:
            sta_code = station.code
            station.station_code = sta_code
            station.network_code = net_code
            for channel in station:
                channel.station_code = sta_code
                channel.network_code = net_code
                channel.channel_code = channel.code


def _keep_obj(obj, **kwargs) -> bool:
    """
    Apply filters and return bool if object have at least one of
    the required attrs and meet the requirements.
    """
    assert set(PARAMS).issuperset(set(kwargs))
    met_requirement = False  # switch if any requirements have been met
    for parameter, requirement in kwargs.items():
        attr, oper = PARAMS[parameter], OPERATORS[parameter]
        # add network, station, channel codes if applicable
        if hasattr(obj, attr):
            value = getattr(obj, attr)
            if not oper(value, requirement):
                return False
            else:
                met_requirement = True
    return met_requirement


def _filter(obj, cls, **kwargs):
    out = (x for x in get_instances_from_tree(obj, cls=cls) if _keep_obj(x, **kwargs))
    return out


def _get_keep_ids(inv, **kwargs):
    """Return the id of objects that meet the filter requirements."""
    _add_codes(inv)
    nets = {id(x) for x in _filter(inv, Network, **kwargs)}
    stas = {id(x) for x in _filter(inv, Station, **kwargs)}
    chans = {id(x) for x in _filter(inv, Channel, **kwargs)}
    return nets | stas | chans


def get_stations(inv: obspy.Inventory, **kwargs) -> obspy.Inventory:
    """
    Return new stations whose channels meet the filter parameters.

    See obspy.clients.fdsn.Client for supported parameters.
    """
    if not kwargs:  # no filter requested, return original stations
        return inv
    if set(kwargs) - SUPPORTED_ARGS:
        unsupported = set(kwargs) - SUPPORTED_ARGS
        msg = f"{unsupported} are not supported by stations get_stations"
        raise TypeError(msg)
    inv = copy.deepcopy(inv)
    keep_ids = _get_keep_ids(inv, **kwargs)
    # iterate over inv and remove channels/stations that dont meet reqs.
    for net in inv:
        for sta in net:
            # only keep channels that meet reqs.
            sta.channels = [x for x in sta.channels if id(x) in keep_ids]
        # only keep stations that meet reqs or have channels that do
        net.stations = [x for x in net.stations if id(x) in keep_ids or len(x.channels)]
    # only keep networks that have some stations
    inv.networks = [x for x in inv.networks if len(x.stations)]
    return inv


def get_stations_bulk(inv: obspy.Inventory, bulk_args) -> obspy.Inventory:
    """ return bulk station request """
    raise NotImplementedError("working on it")


# ----------------- monkey patch get stations onto stations


obspy.Inventory.get_stations = get_stations
obspy.Inventory.get_stations_bulk = get_stations_bulk
