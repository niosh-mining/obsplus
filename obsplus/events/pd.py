"""
Module for generating a pandas dataframe from an obspy events.
"""
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import obspy
import obspy.core.event as ev
import pandas as pd

import obsplus
import obsplus.utils.time
from obsplus.constants import (
    EVENT_COLUMNS,
    EVENT_DTYPES,
    PICK_COLUMNS,
    PICK_DTYPES,
    AMPLITUDE_COLUMNS,
    AMPLITUDE_DTYPES,
    STATION_MAGNITUDE_COLUMNS,
    STATION_MAGNITUDE_DTYPES,
    MAGNITUDE_COLUMNS,
    MAGNITUDE_DTYPES,
    ARRIVAL_COLUMNS,
    ARRIVAL_DTYPES,
    NSLC,
    MAGNITUDE_COLUMN_TYPES,
)
from obsplus.interfaces import BankType, EventClient
from obsplus.structures.dfextractor import (
    DataFrameExtractor,
    standard_column_transforms,
)
from obsplus.utils.events import get_preferred
from obsplus.utils.events import get_seed_id
from obsplus.utils.misc import get_instances_from_tree, read_file, getattrs
from obsplus.utils.time import get_reference_time

# -------------------- init extractors
events_to_df = DataFrameExtractor(
    ev.Event,
    required_columns=EVENT_COLUMNS,
    column_funcs=standard_column_transforms,
    dtypes=EVENT_DTYPES,
)

picks_to_df = DataFrameExtractor(
    ev.Pick, PICK_COLUMNS, column_funcs=standard_column_transforms
)

arrivals_to_df = DataFrameExtractor(
    ev.Arrival, ARRIVAL_COLUMNS, column_funcs=standard_column_transforms
)

amplitudes_to_df = DataFrameExtractor(
    ev.Amplitude, AMPLITUDE_COLUMNS, column_funcs=standard_column_transforms
)

station_magnitudes_to_df = DataFrameExtractor(
    ev.StationMagnitude,
    STATION_MAGNITUDE_COLUMNS,
    column_funcs=standard_column_transforms,
)

magnitudes_to_df = DataFrameExtractor(
    ev.Magnitude, MAGNITUDE_COLUMNS, column_funcs=standard_column_transforms
)


class _OriginQualityExtractor:
    """
    A class encapsulating logic for getting information about origin quality.
    """

    def __init__(self, event: ev.Event):
        self.event = event

    def _get_picks_linked_to_amps(self, eve, arrival_set):
        """ Make sure the picks exists the amplitudes point to. """
        for pick in eve.picks:
            pick.resource_id.set_referred_object(pick)
        pick_dict = {str(p.resource_id): p for p in eve.picks}
        assert set(arrival_set).issubset(
            set(pick_dict)
        ), "arrivals link to non-existent picks"
        return pick_dict

    def _get_pick_count(self, phase, pict_dict):
        """ count the number of non-rejected picks with given phase """
        out = {
            i
            for i, v in pict_dict.items()
            if v.phase_hint == phase and v.evaluation_status != "rejected"
        }
        return len(out)

    def _get_phase_count(self, ori: ev.Origin, phase_type: str):
        """ return the number of phases with phase_type found in origin """
        count = 0
        for ar in ori.arrivals:
            # if phase is not specified dont count it
            if ar.phase is None or not len(ar.phase):
                continue
            if ar.phase == phase_type:
                count += 1
        return count

    def _get_origin_quality_info(self, origin, out):
        """ Get information from quality info."""
        quality_attrs = (
            ("standard_error", np.NaN),
            ("associated_phase_count", out.get("associated_phase_count", 0)),
            ("azimuthal_gap", np.NaN),
            ("used_phase_count", out.get("used_phase_count", 0)),
        )
        quality = getattr(origin, "quality", ev.OriginQuality())
        for (attr, default) in quality_attrs:
            out[attr] = getattr(quality, attr, None) or default

    def _get_origin_uncertainty(self, origin, out):
        """ Get information from uncertainty. """
        uncert_attrs = (("horizontal_uncertainty", np.NaN),)
        uncert = getattr(origin, "origin_uncertainty", ev.OriginUncertainty())
        for (attr, default) in uncert_attrs:
            out[attr] = getattr(uncert, attr, default) or default

    def _get_depth_uncertainty_info(self, origin, out):
        """ Get info from depth info. """
        depth_uncert = origin.depth_errors
        out["vertical_uncertainty"] = getattr(depth_uncert, "uncertainty", np.NaN)

    def _get_phase_and_pick_counts(self, origin, out):
        # get a dict of picks
        arrivals = {str(x.pick_id) for x in origin.arrivals}
        pick_dict = self._get_picks_linked_to_amps(self.event, arrivals)
        used_picks = [p for pid, p in pick_dict.items() if pid in arrivals]
        # get counts of picks
        out["p_phase_count"] = self._get_phase_count(origin, "P")
        out["s_phase_count"] = self._get_phase_count(origin, "S")
        out["p_pick_count"] = self._get_pick_count("P", pick_dict)
        out["s_pick_count"] = self._get_pick_count("S", pick_dict)
        out["used_phase_count"] = out["p_phase_count"] + out["s_phase_count"]
        # get names of station and station count
        assert all(used_picks)
        pset = {p.waveform_id.station_code for p in used_picks}
        sl = sorted(list(pset))
        out["stations"] = ", ".join(sl)
        out["station_count"] = len(sl)

    def __call__(self):
        """ Return a dict of origin quality attributes. """
        out = {}
        origin = get_preferred(self.event, "origin", init_empty=True)
        # get phase and pick count
        self._get_phase_and_pick_counts(origin, out)
        # now extract information
        self._get_origin_quality_info(origin, out)
        self._get_depth_uncertainty_info(origin, out)
        self._get_origin_uncertainty(origin, out)
        return out


def _get_last_magnitude(mags: Sequence[ev.Magnitude], mag_type: Optional[str] = None):
    """ Get the value of the last magnitude, optionally of a given type. """
    out = np.NaN
    for mag in mags:
        if mag_type is not None:
            mtype = (mag.magnitude_type or "").upper()
            if not mag_type == mtype:
                continue
        out = mag.mag
    return out


def _path_or_event_bank(path):
    """
    Return either a path (str) to a function, or an EventBank if path is a directory.
    """
    if Path(path).is_dir():
        return obsplus.EventBank(path).update_index()
    return str(path)


@events_to_df.extractor
def _get_event_description(event):
    """ return a string of the first event description. """
    try:
        return event.event_descriptions[0].text
    except (AttributeError, IndexError, TypeError):
        return None


@events_to_df.extractor
def _get_event_id(event):
    return str(event.resource_id)


@events_to_df.extractor
def _get_author(event):
    """ get the name of the analyst as a string, or return empty str """
    ci = getattr(event, "creation_info", None)
    return getattr(ci, "author", "")


@events_to_df.extractor
def _get_eve_creation_info(event):
    """ pull out information from the event level creation_info """
    keys = ("author", "agency_id", "creation_time", "version")
    out = {}
    cinfo = getattr(event, "creation_info", None)
    if hasattr(cinfo, "__dict__"):
        out.update(cinfo.__dict__)
    return {x: out.get(x) for x in keys}


@events_to_df.extractor()
def _get_update_time(eve):
    """ return the most recent time anything was updated in event """
    creations = get_instances_from_tree(eve, cls=ev.CreationInfo)
    timestamps = [getattr(x.creation_time, "timestamp", None) or 0 for x in creations]
    return {"updated": max(timestamps) if timestamps else np.NaN}


origin_dtypes = {x: float for x in ["latitude", "longitude", "depth"]}


@events_to_df.extractor(dtypes=origin_dtypes)
def _get_origin_basic(eve):
    """ extract basic info from origin. """
    ori = get_preferred(eve, "origin", init_empty=True)
    return getattrs(ori, set(origin_dtypes))


@events_to_df.extractor
def _get_time(event):
    try:
        return {"time": obsplus.utils.time.get_reference_time(event)}
    except ValueError:  # no valid starttime
        return {"time": np.nan}


@events_to_df.extractor
def _get_origin_quality(eve: ev.Event):
    """ get information from origin quality """
    return _OriginQualityExtractor(eve)()


@events_to_df.extractor
def _get_magnitude_info(eve: ev.Event):
    """
    Extract magnitude information. Get base magnitude, as well as various
    other magnitude types (where applicable).
    """
    out = {}
    magnitude = get_preferred(eve, "magnitude", init_empty=True)
    out["magnitude"] = magnitude.mag
    out["magnitude_type"] = magnitude.magnitude_type or ""
    for col_name, mag_type in MAGNITUDE_COLUMN_TYPES.items():
        out[col_name] = _get_last_magnitude(eve.magnitudes, mag_type)
    return out


# ----------------- Alternative constructors


@events_to_df.register(str)
@events_to_df.register(Path)
def _str_catalog_to_df(path):
    path = _path_or_event_bank(path)
    if isinstance(path, obsplus.EventBank):
        return events_to_df(path)
    funcs = (obspy.read_events, pd.read_csv)
    return events_to_df(read_file(path, funcs=funcs))


@events_to_df.register(BankType)
def _bank_to_catalog(bank):
    assert isinstance(bank, EventClient), "wrong bank buddy"
    df = bank.read_index()
    # make sure event_id is a column
    if df.index.name == "event_id":
        df = df.reset_index()
    return events_to_df(df)


# -------------- Picks to dataframe


@picks_to_df.register(str)
@picks_to_df.register(Path)
def _file_to_picks_df(path):
    path = _path_or_event_bank(path)
    if isinstance(path, obsplus.EventBank):
        return picks_to_df(path)
    return _file_to_df(path, picks_to_df)


@picks_to_df.register(ev.Event)
@picks_to_df.register(ev.Catalog)
def _picks_from_event(event: ev.Event):
    """ return a dataframe of picks from a pick list """
    return _objs_from_event(event, "picks", picks_to_df)


@picks_to_df.register(BankType)
def _picks_from_event_bank(event_bank):
    return _objs_from_event_bank(event_bank, picks_to_df)


@picks_to_df.extractor(dtypes=PICK_DTYPES)
def _pick_extractor(pick):
    return _obj_extractor(pick, PICK_DTYPES, error_obj="time_errors")


# -------------- Arrivals to dataframe


# still thinking about the best way to go about combining these...


@arrivals_to_df.register(str)
@arrivals_to_df.register(Path)
def _file_to_arrivals_df(path):
    path = _path_or_event_bank(path)
    if isinstance(path, obsplus.EventBank):
        return arrivals_to_df(path)
    return _file_to_df(path, arrivals_to_df)


@arrivals_to_df.register(ev.Event)
@arrivals_to_df.register(ev.Catalog)
def _arrivals_from_event(event: ev.Event):
    """ return a dataframe of arrivals from an event """
    cat = [event] if isinstance(event, ev.Event) else event
    origins = [e.preferred_origin() for e in cat if e.preferred_origin()]
    arr_list = []
    for o in origins:
        extras = {}
        arrivals = o.arrivals
        event_dict = dict(
            origin_id=str(o.resource_id), origin_time=get_reference_time(o)
        )
        extras.update({id(arr): event_dict for arr in arrivals})
        arrivals = arrivals_to_df(o.arrivals, extras=extras)
        arr_list.append(arrivals)
    if not len(arr_list):
        return pd.DataFrame(columns=ARRIVAL_COLUMNS)
    else:
        return pd.concat(arr_list).reset_index(drop=True)


@arrivals_to_df.register(ev.Origin)
def _arrivals_from_origin(origin: ev.Origin):
    extras = {}
    arrivals = origin.arrivals
    event_dict = dict(
        origin_id=str(origin.resource_id), origin_time=get_reference_time(origin)
    )
    extras.update({id(arr): event_dict for arr in arrivals})
    return arrivals_to_df(origin.arrivals, extras=extras)


@arrivals_to_df.register(BankType)
def _arrivals_from_event_bank(event_bank):
    return _objs_from_event_bank(event_bank, arrivals_to_df)


@arrivals_to_df.extractor(dtypes=ARRIVAL_DTYPES)
def _arrivals_extractor(arr):
    return _obj_extractor(arr, ARRIVAL_DTYPES)


# -------------- Amplitudes to dataframe


@amplitudes_to_df.register(str)
@amplitudes_to_df.register(Path)
def _file_to_amplitudes_df(path):
    path = _path_or_event_bank(path)
    if isinstance(path, obsplus.EventBank):
        return amplitudes_to_df(path)
    return _file_to_df(path, picks_to_df)


@amplitudes_to_df.register(ev.Event)
@amplitudes_to_df.register(ev.Catalog)
def _amplitudes_from_event(event: ev.Event):
    """ return a dataframe of amplitudes from an amplitude list """
    return _objs_from_event(event, "amplitudes", amplitudes_to_df)


@amplitudes_to_df.register(BankType)
def _amplitudes_from_event_bank(event_bank):
    return _objs_from_event_bank(event_bank, amplitudes_to_df)


@amplitudes_to_df.extractor(dtypes=AMPLITUDE_DTYPES)
def _amplitudes_extractor(amp):
    base = _obj_extractor(amp, AMPLITUDE_DTYPES, error_obj="generic_amplitude_errors")
    # get other amplitude info
    if amp.time_window:
        base["reference"] = amp.time_window.reference.timestamp
        base["time_begin"] = amp.time_window.begin
        base["time_end"] = amp.time_window.end
    if amp.scaling_time:
        base["scaling_time"] = amp.scaling_time.timestamp
    return base


# -------------- StationMagnitudes to dataframe


@station_magnitudes_to_df.register(str)
@station_magnitudes_to_df.register(Path)
def _file_to_station_magnitudes_df(path):
    path = _path_or_event_bank(path)
    if isinstance(path, obsplus.EventBank):
        return station_magnitudes_to_df(path)
    return _file_to_df(path, station_magnitudes_to_df)


@station_magnitudes_to_df.register(ev.Event)
@station_magnitudes_to_df.register(ev.Catalog)
def _station_magnitudes_from_event(event: ev.Event):
    """ return a dataframe of station_magnitudes from a station_magnitude list """
    return _objs_from_event(event, "station_magnitudes", station_magnitudes_to_df)


@station_magnitudes_to_df.register(ev.Magnitude)  # This may not work nicely...
def _station_magnitudes_from_magnitude(mag: ev.Magnitude):
    """ return a dataframe of station magnitudes from a Magnitude """
    sms = []
    for smc in mag.station_magnitude_contributions:
        if smc.station_magnitude_id:
            sms.append(smc.station_magnitude_id.get_referred_object())
    extras = {}
    extras.update({id(sm): {"magnitude_id": mag.resource_id.id} for sm in sms})
    return station_magnitudes_to_df(sms, extras=extras)


@station_magnitudes_to_df.register(BankType)
def _station_magnitudes_from_event_bank(event_bank):
    return _objs_from_event_bank(event_bank, station_magnitudes_to_df)


@station_magnitudes_to_df.extractor(dtypes=STATION_MAGNITUDE_DTYPES)
def _station_magnitudes_extractor(sm):
    return _obj_extractor(sm, STATION_MAGNITUDE_DTYPES, error_obj="mag_errors")


# -------------- Magnitudes to dataframe


@magnitudes_to_df.register(str)
@magnitudes_to_df.register(Path)
def _file_to_magnitudes_df(path):
    path = _path_or_event_bank(path)
    if isinstance(path, obsplus.EventBank):
        return magnitudes_to_df(path)
    return _file_to_df(path, magnitudes_to_df)


@magnitudes_to_df.register(ev.Event)
@magnitudes_to_df.register(ev.Catalog)
def _magnitudes_from_event(event: ev.Event):
    """ return a dataframe of magnitudes from a magnitude list """
    return _objs_from_event(event, "magnitudes", magnitudes_to_df)


@magnitudes_to_df.register(BankType)
def _magnitudes_from_event_bank(event_bank):
    return _objs_from_event_bank(event_bank, magnitudes_to_df)


@magnitudes_to_df.extractor(dtypes=MAGNITUDE_DTYPES)
def _magnitudes_extractor(mag):
    return _obj_extractor(mag, MAGNITUDE_DTYPES, seed_id=False, error_obj="mag_errors")


# -------------- Internal functions for extracting event info


def _file_to_df(path, extractor):
    """Extract info from a file"""
    path = str(path)
    try:
        return extractor(obspy.read_events(path))
    except TypeError:  # obspy failed to read file, try csv
        return extractor(pd.read_csv(path))


def _objs_from_event(event, attr, extractor):
    """ return a dataframe of an obj type from an event """
    # ensure we have an iterable and flatten station_magnitudes
    cat = [event] if isinstance(event, ev.Event) else event
    objs = [obj for e in cat for obj in getattr(e, attr)]
    return extractor(objs, extras=_get_event_info(cat, attr))


def _objs_from_event_bank(event_bank, extractor):
    """ return a dataframe of a set obj type from an event bank """
    assert isinstance(event_bank, EventClient)
    return extractor(event_bank.get_events())


def _obj_extractor(obj, dtypes, seed_id=True, error_obj=None):
    """ extract common information from event object """
    # extract attributes that are floats/str
    overlap = set(obj.__dict__) & set(dtypes)
    base = {i: getattr(obj, i) for i in overlap}
    # get waveform_id stuff (seed_id, network, station, location, channel)
    if seed_id:
        base.update(_get_seed_id(obj))
    # extract error info, if applicable
    if error_obj:
        errors = obj.__dict__[error_obj]
        if errors:
            base.update(_get_uncertainty(errors))
    # get creation info
    cio = obj.creation_info or ev.CreationInfo()
    base.update(_get_creation_info(cio))
    return base


def _get_event_info(cat, attr):
    """Extract event_id and time for an extractor"""
    extras = {}
    for event in cat:
        iterable = event.__dict__[attr]
        if not len(iterable):
            continue
        event_dict = dict(
            event_id=str(event.resource_id), event_time=get_reference_time(event)
        )
        extras.update({id(item): event_dict for item in iterable})
    return extras


def _get_creation_info(cio):
    """ Strip the creation info for an extractor """
    return {
        "creation_time": cio.creation_time,
        "author": cio.author,
        "agency_id": cio.agency_id,
    }


def _get_uncertainty(errors):
    """ Strip uncertainty info for an extractor """
    return {
        "uncertainty": errors.uncertainty,
        "lower_uncertainty": errors.lower_uncertainty,
        "upper_uncertainty": errors.upper_uncertainty,
        "confidence_level": errors.confidence_level,
    }


def _get_seed_id(obj):
    """ Strip nslc info for an extractor """
    seed_id = get_seed_id(obj)
    dd = {x: y for x, y in zip(NSLC, seed_id.split("."))}
    return {"seed_id": seed_id, **dd}


# --- monkey patch events/event classes to have to_df methods.

# event_to_dataframe
def event_to_dataframe(cat_or_event):
    """
    Given a catalog or event, return a DataFrame summary.

    Notes
    -----
    Because of the complexity of obspy catalog and event objects, this extractor
    makes some assumptions to determine the appropriate information to include.

    These assumptions are as follows:

    The origin information is based on the preferred origin, or the last origin
    attached to the event if the preferred origin is not set.

    The event description uses the first object in the list of event descriptions.

    The magnitude information is based on the preferred magnitude. Additionally,
    it will look for the latest magnitudes with a magnitude of "MD", "ML", and
    "MW" magnitudes and report those magnitudes.

    For the origin quality information, specifically the associated and used
    phase counts, the information attached to origin's OriginQuality object.
    If this information is not available, it will attempt to estimate this
    by counting the number of P and S picks and arrivals attached to the
    event/origin.

    The updated time is the latest of all creation times attached to an
    object, while the reported creation time, author, and agency id are all
    based on the Event object itself.
    """
    return events_to_df(cat_or_event)


obspy.core.event.Catalog.to_df = event_to_dataframe
obspy.core.event.Event.to_df = event_to_dataframe


# picks_to_dataframe
def picks_to_dataframe(cat_or_event):
    """ Given a catalog or event, return a dataframe of picks """
    return picks_to_df(cat_or_event)


obspy.core.event.Catalog.picks_to_df = picks_to_dataframe
obspy.core.event.Event.picks_to_df = picks_to_dataframe


# arrivals_to_dataframe
def arrivals_to_dataframe(cat_or_event):
    """ Given a catalog or event, return a dataframe of arrivals """
    return arrivals_to_df(cat_or_event)


obspy.core.event.Catalog.arrivals_to_df = arrivals_to_dataframe
obspy.core.event.Event.arrivals_to_df = arrivals_to_dataframe
obspy.core.event.Origin.arrivals_to_df = arrivals_to_dataframe


# amplitudes_to_dataframe
def amplitudes_to_dataframe(cat_or_event):
    """ Given a catalog or event, return a dataframe of amplitudes """
    return amplitudes_to_df(cat_or_event)


obspy.core.event.Catalog.amplitudes_to_df = amplitudes_to_dataframe
obspy.core.event.Event.amplitudes_to_df = amplitudes_to_dataframe


# station_magnitudes_to_dataframe
def station_magnitudes_to_dataframe(cat_or_event):
    """ Given a catalog or event, return a dataframe of station magnitudes """
    return station_magnitudes_to_df(cat_or_event)


obspy.core.event.Catalog.station_magnitudes_to_df = station_magnitudes_to_dataframe
obspy.core.event.Event.station_magnitudes_to_df = station_magnitudes_to_dataframe
obspy.core.event.Magnitude.station_magnitudes_to_df = station_magnitudes_to_dataframe


# magnitudes_to_dataframe
def magnitudes_to_dataframe(cat_or_event):
    """ Given a catalog or event, return a dataframe of magnitudes """
    return magnitudes_to_df(cat_or_event)


obspy.core.event.Catalog.magnitudes_to_df = magnitudes_to_dataframe
obspy.core.event.Event.magnitudes_to_df = magnitudes_to_dataframe

# save the default events converter for use by other code (eg EventBank).
_default_cat_to_df = events_to_df.copy()
_default_pick_to_df = picks_to_df.copy()
