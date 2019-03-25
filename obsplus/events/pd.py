"""
Module for generating a pandas dataframe from an obspy events.
"""

from os.path import isdir
from pathlib import Path

import numpy as np
import obspy
import obspy.core.event as ev
import pandas as pd

import obsplus
from obsplus.constants import (
    EVENT_COLUMNS,
    PICK_COLUMNS,
    AMPLITUDE_COLUMNS,
    STATION_MAGNITUDE_COLUMNS,
    NSLC,
    PICK_DTYPES
)
from obsplus.events.utils import get_reference_time, get_seed_id
from obsplus.interfaces import BankType, EventClient
from obsplus.structures.dfextractor import DataFrameExtractor
from obsplus.utils import read_file, apply_to_files_or_skip, get_instances, getattrs
from obsplus import get_preferred

# -------------------- event extractors

events_to_df = DataFrameExtractor(
    ev.Event, required_columns=EVENT_COLUMNS, utc_columns=("time",)
)


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
def _get_creation_info(event):
    """ pull out information from the event level creation_info """
    keys = ("author", "agency_id", "creation_time", "version")
    out = {}
    cinfo = getattr(event, "creation_info", None)
    if hasattr(cinfo, "__dict__"):
        out.update(cinfo.__dict__)
    return {x: out.get(x) for x in keys}


@events_to_df.extractor(dtypes={"updated": float})
def _get_update_time(eve):
    """ return the most recent time anything was updated in event """
    creations = get_instances(eve, ev.CreationInfo)
    timestamps = [getattr(x.creation_time, "timestamp", None) or 0 for x in creations]
    return {"updated": max(timestamps) if timestamps else np.NaN}


origin_dtypes = {x: float for x in ["latitude", "longitude", "depth"]}


@events_to_df.extractor(dtypes=origin_dtypes)
def _get_origin_basic(eve):
    """ extract basic info from origin. """
    ori = get_preferred(eve, "origin", init_empty=True)
    return getattrs(ori, set(origin_dtypes))


@events_to_df.extractor(dtypes={"time": float})
def _get_time(event):
    try:
        return {"time": obsplus.get_reference_time(event)}
    except ValueError:  # no valid starttime
        return {"time": np.nan}


def _get_used_stations(origin: ev.Origin, pid):
    """ get the stations used in the origin, return str """
    arrivals = origin.arrivals
    picks = [pid.get(arr.pick_id.id) for arr in arrivals]
    assert all(picks)
    pset = {p.waveform_id.station_code for p in picks}
    sl = sorted(list(pset))
    return {"stations": ", ".join(sl), "station_count": len(sl)}


@events_to_df.extractor
def _get_origin_quality(eve: ev.Event):
    """ get information from origin quality """
    # ensure resource_ids in arrivals don't point to picks that dont exist
    ori = get_preferred(eve, "origin", init_empty=True)

    for pick in eve.picks:
        pick.resource_id.set_referred_object(pick)
    pick_dict = {str(p.resource_id): p for p in eve.picks}
    apid = {str(ar.pick_id): ar for ar in ori.arrivals}
    assert set(apid).issubset(set(pick_dict)), "arrivals link to non-existent picks"

    # desired attrs
    qual_set = {
        "standard_error",
        "associated_phase_count",
        "azimuthal_gap",
        "used_phase_count",
    }
    uncert_set = {"horizontal_uncertainty"}
    # objects to pull from
    qual = ori.quality
    uncert = ori.origin_uncertainty
    duncert = ori.depth_errors
    # out dict to populate
    out = {}
    for obsject, attrs in ((qual, qual_set), (uncert, uncert_set)):
        obsject = obsject or {}
        out.update(getattrs(obsject, attrs))

    if duncert is not None:
        out["vertical_uncertainty"] = duncert.get("uncertainty", np.nan)
    else:
        out["vertical_uncertainty"] = np.nan

    out["p_phase_count"] = _get_phase_count(ori, "P")
    out["s_phase_count"] = _get_phase_count(ori, "S")
    out["p_pick_count"] = _get_pick_count("P", pick_dict)
    out["s_pick_count"] = _get_pick_count("S", pick_dict)

    # get station count and concat'ed string of stations
    arrivals = ori.arrivals
    picks = [pick_dict.get(arr.pick_id.id) for arr in arrivals]
    assert all(picks)
    pset = {p.waveform_id.station_code for p in picks}
    sl = sorted(list(pset))
    out.update({"stations": ", ".join(sl), "station_count": len(sl)})

    return out


def _get_pick_count(phase, pict_dict):
    """ count the number of non-rejected picks with given phase """
    out = {
        i
        for i, v in pict_dict.items()
        if v.phase_hint == phase and v.evaluation_status != "rejected"
    }
    return len(out)


def _get_phase_count(ori: ev.Origin, phase_type: str):
    """ return the number of phases with phase_type found in origin """
    count = 0
    for ar in ori.arrivals:
        # if phase is not specified dont count it
        if ar.phase is None or not len(ar.phase):
            continue
        if ar.phase == phase_type:
            count += 1
    return count


@events_to_df.extractor
def _get_magnitude_info(eve: ev.Event):
    """ extract magnitude information. Get base magnitude, as well as various
     other magnitude types (where applicable). """
    out = {}
    magnitude = get_preferred(eve, "magnitude", init_empty=True)
    out["magnitude"] = magnitude.mag
    out["magnitude_type"] = magnitude.magnitude_type
    mw = [
        x.mag
        for x in eve.magnitudes
        if x.magnitude_type and x.magnitude_type.upper() == "MW"
    ]
    ml = [
        x.mag
        for x in eve.magnitudes
        if x.magnitude_type and x.magnitude_type.upper() == "ML"
    ]
    md = [
        x.mag
        for x in eve.magnitudes
        if x.magnitude_type and x.magnitude_type.upper() == "MD"
    ]

    out["moment_magnitude"] = mw[-1] if mw else np.nan
    out["local_magnitude"] = ml[-1] if ml else np.nan
    out["duration_magnitude"] = md[-1] if md else np.nan
    return out


# ----------------- Alternative constructors


@events_to_df.register(str)
@events_to_df.register(Path)
def _str_catalog_to_df(path):
    # if applied to directory, recurse
    path = str(path)  # convert possible path object to str
    if isdir(path):
        df = pd.concat(list(apply_to_files_or_skip(_str_catalog_to_df, path)))
        df.reset_index(drop=True, inplace=True)
        return df
    # else try to read single file
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


picks_to_df = DataFrameExtractor(
    ev.Pick, PICK_COLUMNS, utc_columns=("time", "event_time")
)


@picks_to_df.register(str)
@picks_to_df.register(Path)
def _file_to_picks_df(path):
    path = str(path)
    try:
        return picks_to_df(obspy.read_events(path))
    except TypeError:  # obspy failed to read file, try csv
        return picks_to_df(pd.read_csv(path))


@picks_to_df.register(ev.Event)
@picks_to_df.register(ev.Catalog)
def _picks_from_event(event: ev.Event):
    """ return a dataframe of picks from a pick list """
    # ensure we have an iterable and flatten picks
    cat = [event] if isinstance(event, ev.Event) else event
    picks = [p for e in cat for p in e.picks]
    # iterate events and create extras for inject event info to pick level
    extras = {}
    for event in cat:
        if not len(event.picks):
            continue  # skip events with no picks
        event_dict = dict(
            event_id=str(event.resource_id), event_time=get_reference_time(event)
        )
        extras.update({id(p): event_dict for p in event.picks})

    return picks_to_df(picks, extras=extras)


@picks_to_df.register(BankType)
def _picks_from_event_bank(event_bank):
    assert isinstance(event_bank, EventClient)
    return picks_to_df(event_bank.get_events())


@picks_to_df.extractor(dtypes=PICK_DTYPES)
def _pick_extractor(pick):
    # extract attributes that are floats/str
    overlap = set(pick.__dict__) & set(PICK_DTYPES)
    base = {i: getattr(pick, i) for i in overlap}
    # get waveform_id stuff (seed_id, network, station, location, channel)
    # Should it fail if it can't find the seed id, or just return one that's blank? A pick is pretty useless without it
    seed_id = get_seed_id(pick)
    # seed_id = (pick.waveform_id or ev.WaveformStreamID()).get_seed_string()
    dd = {x: y for x, y in zip(NSLC, seed_id.split("."))}
    base.update(dd)
    base["seed_id"] = seed_id
    # get creation info
    cio = pick.creation_info or ev.CreationInfo()
    base["creation_time"] = cio.creation_time
    base["author"] = cio.author
    base["agency_id"] = cio.agency_id
    # get time error info
    terrors = pick.time_errors
    if terrors:
        base["uncertainty"] = terrors.uncertainty
        base["lower_uncertainty"] = terrors.lower_uncertainty
        base["upper_uncertainty"] = terrors.upper_uncertainty
        base["confidence_level"] = terrors.confidence_level
    return base


# -------------- Amplitudes to dataframe


# It seems like there is enough similarity between amplitudes_to_df and
# picks_to_df that there should be some way to combine them...
amplitudes_to_df = DataFrameExtractor(
    ev.Amplitude, AMPLITUDE_COLUMNS, utc_columns=("event_time")
)


@amplitudes_to_df.register(str)
@amplitudes_to_df.register(Path)
def _file_to_amplitudes_df(path):
    path = str(path)
    try:
        return amplitudes_to_df(obspy.read_events(path))
    except TypeError:  # obspy failed to read file, try csv
        return amplitudes_to_df(pd.read_csv(path))


amplitude_attrs = {
    "resource_id": str,
    "generic_amplitude": float,
    "seed_id": str,
    "type": str,
    "category": str,
    "unit": str,
    "magnitude_hint": str,
    "filter_id": str,
    "method_id": str,
    "period": float,
    "snr": float,
    "pick_id": str,
    "reference": float,
    "time_begin": float,
    "time_end": float,
    "scaling_time": float,
    "evaluation_mode": str,
    "evaluation_status": str,
    "creation_time": float,
    "author": str,
    "agency_id": str,
    "event_id": str,
    "network": str,
    "station": str,
    "location": str,
    "channel": str,
}


@amplitudes_to_df.register(ev.Event)
@amplitudes_to_df.register(ev.Catalog)
def _amplitudes_from_event(event: ev.Event):
    """ return a dataframe of amplitudes from an amplitude list """
    # ensure we have an iterable and flatten amplitudes
    cat = [event] if isinstance(event, ev.Event) else event
    amps = [a for e in cat for a in e.amplitudes]
    # iterate events and create extras for inject event info to amplitude level
    extras = {}
    for event in cat:
        if not len(event.amplitudes):
            continue  # skip events with no amplitudes
        event_dict = dict(
            event_id=str(event.resource_id), event_time=get_reference_time(event)
        )
        extras.update({id(a): event_dict for a in event.amplitudes})

    return amplitudes_to_df(amps, extras=extras)


@amplitudes_to_df.register(BankType)
def _amplitudes_from_event_bank(event_bank):
    assert isinstance(event_bank, EventClient)
    return amplitudes_to_df(event_bank.get_events())


@amplitudes_to_df.extractor(dtypes=amplitude_attrs)
def _amplitudes_extractor(amp):
    # extract attributes that are floats/str
    overlap = set(amp.__dict__) & set(amplitude_attrs)
    base = {i: getattr(amp, i) for i in overlap}
    # get waveform_id stuff (seed_id, network, station, location, channel)
    seed_id = get_seed_id(amp)
    dd = {x: y for x, y in zip(NSLC, seed_id.split("."))}
    base.update(dd)
    base["seed_id"] = seed_id

    # get more complicated amplitude info
    if amp.time_window:
        base["reference"] = amp.time_window.reference.timestamp
        base["time_begin"] = amp.time_window.begin
        base["time_end"] = amp.time_window.end
    if amp.scaling_time:
        base["scaling_time"] = amp.scaling_time.timestamp

    # get creation info
    cio = amp.creation_info or ev.CreationInfo()
    base["creation_time"] = cio.creation_time
    base["author"] = cio.author
    base["agency_id"] = cio.agency_id
    return base


# -------------- Amplitudes to dataframe


# It seems like there is enough similarity between amplitudes_to_df and
# picks_to_df that there should be some way to combine them...
station_magnitudes_to_df = DataFrameExtractor(
    ev.StationMagnitude, STATION_MAGNITUDE_COLUMNS, utc_columns=("event_time")
)


@station_magnitudes_to_df.register(str)
@station_magnitudes_to_df.register(Path)
def _file_to_station_magnitudes_df(path):
    path = str(path)
    try:
        return station_magnitudes_to_df(obspy.read_events(path))
    except TypeError:  # obspy failed to read file, try csv
        return station_magnitudes_to_df(pd.read_csv(path))


station_magnitude_attrs = {
    "resource_id": str,
    "mag": float,
    "seed_id": str,
    "station_magnitude_type": str,
    "amplitude_id": str,
    "magnitude_id": str,
    "origin_id": str,
    "method_id": str,
    "creation_time": float,
    "author": str,
    "agency_id": str,
    "event_id": str,
    "network": str,
    "station": str,
    "location": str,
    "channel": str,
}


@station_magnitudes_to_df.register(ev.Event)
@station_magnitudes_to_df.register(ev.Catalog)
def _station_magnitudes_from_event(event: ev.Event):
    """ return a dataframe of station_magnitudes from a station_magnitude list """
    # ensure we have an iterable and flatten station_magnitudes
    # is there a relatively simple way to get the ID of the magnitude(s) that use a sm?
    cat = [event] if isinstance(event, ev.Event) else event
    sms = [sm for e in cat for sm in e.station_magnitudes]
    # iterate events and create extras for inject event info to station magnitude level
    extras = {}
    for event in cat:
        if not len(event.station_magnitudes):
            continue  # skip events with no station magnitudes
        event_dict = dict(
            event_id=str(event.resource_id), event_time=get_reference_time(event)
        )
        extras.update({id(sm): event_dict for sm in event.station_magnitudes})
    return station_magnitudes_to_df(sms, extras=extras)


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
    assert isinstance(event_bank, EventClient)
    return station_magnitudes_to_df(event_bank.get_events())


@station_magnitudes_to_df.extractor(dtypes=station_magnitude_attrs)
def _station_magnitudes_extractor(sm):
    # extract attributes that are floats/str
    overlap = set(sm.__dict__) & set(station_magnitude_attrs)
    base = {i: getattr(sm, i) for i in overlap}
    # get waveform_id stuff (seed_id, network, station, location, channel)
    seed_id = get_seed_id(sm)
    dd = {x: y for x, y in zip(NSLC, seed_id.split("."))}
    base.update(dd)
    base["seed_id"] = seed_id

    # get creation info
    cio = sm.creation_info or ev.CreationInfo()
    base["creation_time"] = cio.creation_time
    base["author"] = cio.author
    base["agency_id"] = cio.agency_id
    return base


# --- monkey patch events/event classes to have to_df methods.


def event_to_dataframe(cat_or_event):
    """ Given a catalog or event, return a Dataframe summary. """
    return events_to_df(cat_or_event)


def picks_to_dataframe(cat_or_event):
    """ Given a catalog or event return a dataframe of picks """
    return picks_to_df(cat_or_event)


def amplitudes_to_dataframe(cat_or_event):
    """ Given a catalog or event return a dataframe of amplitudes """
    return amplitudes_to_df(cat_or_event)


def station_magnitudes_to_dataframe(cat_or_event):
    """ Given a catalog or event return a dataframe of station magnitudes """
    return station_magnitudes_to_df(cat_or_event)


obspy.core.event.Catalog.to_df = event_to_dataframe
obspy.core.event.Event.to_df = event_to_dataframe
obspy.core.event.Catalog.picks_to_df = picks_to_dataframe
obspy.core.event.Event.picks_to_df = picks_to_dataframe
obspy.core.event.Catalog.amplitudes_to_df = amplitudes_to_dataframe
obspy.core.event.Event.amplitudes_to_df = amplitudes_to_dataframe
obspy.core.event.Catalog.station_magnitudes_to_df = station_magnitudes_to_dataframe
obspy.core.event.Event.station_magnitudes_to_df = station_magnitudes_to_dataframe
obspy.core.event.Magnitude.station_magnitudes_to_df = station_magnitudes_to_dataframe

# save the default events converter for use by other code (eg EventBank).
_default_cat_to_df = events_to_df.copy()
_default_pick_to_df = picks_to_df.copy()
