"""
Slimmed down bits of obspy's mseed module.

Copyrights to ObsPy developers still apply.
"""
import os

import numpy as np
from obspy.io.mseed.core import DATATYPES, C, clibmseed


def _get_lil(mseed_object):
    """
    Slimmed down functions for indexing miniseed files.

    Contains hacked bits from obspy's mseed module.

    Copyrights to ObsPy developers still apply.
    """
    # Parse the headonly and reclen flags.
    unpack_data = 0
    details = False
    reclen = -1
    header_byteorder = -1
    verbose = 0
    selections = None

    length = os.path.getsize(mseed_object)
    assert 128 < length < 2 ** 31, "data length is outside of the save range"

    # Assume a file was passed
    bfr_np = np.fromfile(mseed_object, dtype=np.int8)

    buflen = len(bfr_np)
    all_data = []

    # Use a callback function to allocate the memory and keep track of the
    # data.
    def allocate_data(samplecount, sampletype):
        # Enhanced sanity checking for libmseed 2.10 can result in the
        # sampletype not being set. Just return an empty array in this case.
        if sampletype == b"\x00":
            data = np.empty(0)
        else:
            data = np.empty(samplecount, dtype=DATATYPES[sampletype])
        all_data.append(data)
        return data.ctypes.data

    alloc_data = C.CFUNCTYPE(C.c_longlong, C.c_int, C.c_char)(allocate_data)
    clibmseed.verbose = bool(verbose)
    # note: normally this would be in a try except, but we just fal back to
    # obspy read if an exception happens (this only needs to work most the time)
    lil = clibmseed.readMSEEDBuffer(
        bfr_np,
        buflen,
        selections,
        C.c_int8(unpack_data),
        reclen,
        C.c_int8(verbose),
        C.c_int8(details),
        header_byteorder,
        alloc_data,
    )
    clibmseed.verbose = True
    return lil


def summarize_mseed(mseed_object):
    """
    get a summary of an mseed file.

    Note: we cannot simply use obspy.io.mseed.get_record_information because it
    returns info only about the first trace.
    """
    lil = _get_lil(mseed_object)
    traces = []
    current_id = lil.contents
    while True:
        # Init header with the essential information.
        header = {
            "network": current_id.network.strip(),
            "station": current_id.station.strip(),
            "location": current_id.location.strip(),
            "channel": current_id.channel.strip(),
            "path": mseed_object,
        }
        # Loop over segments.
        try:
            current_segment = current_id.firstSegment.contents
        except ValueError:
            break
        while True:
            header["starttime"] = current_segment.starttime * 1_000
            header["endtime"] = current_segment.endtime * 1_000
            header["sampling_period"] = current_segment.hpdelta * 1_000
            traces.append(dict(header))
            try:
                current_segment = current_segment.next.contents
            except ValueError:
                break
        try:
            current_id = current_id.next.contents
        except ValueError:
            break

    clibmseed.lil_free(lil)  # NOQA
    del lil  # NOQA
    if not traces:
        raise IOError(f"could not read {mseed_object}")
    return traces
