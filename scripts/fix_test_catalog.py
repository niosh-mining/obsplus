"""
This script is designed to fix catalog 6.

It has two S picks on the same station. We need to remove these and anything
that might refer to them
"""
from pathlib import Path

import obsplus
import obspy
from obspy.core.event import ResourceIdentifier
from obsplus.events.validate import validate_catalog

if __name__ == "__main__":
    cat_path = Path(
        "/media/data/Gits/obsplus/tests/test_data/qml2merge/2016-10-15T02-27-50/2016-10-15T02-27-50_2.xml"
    )
    assert cat_path.exists()

    cat = obspy.read_events(str(cat_path))

    # remove all amplitudes
    picks = cat[0].picks
    cat[0].picks = [x for x in picks if not x.phase_hint == "IAML"]

    # get duplicated event ids
    pdf = obsplus.picks_to_df(cat)
    pdf = pdf[pdf["evaluation_status"] != "rejected"]
    duplicated_ids = set(pdf[pdf["station"].duplicated()]["resource_id"])

    # next mark duplicated as rejected and prune the event
    for duplicated_id in duplicated_ids:
        rid = ResourceIdentifier(duplicated_id)
        obj = rid.get_referred_object()
        obj.evaluation_status = "rejected"

    cat_out = obsplus.events.utils.prune_events(cat)

    validate_catalog(cat_out)

    cat.write(str(cat_path), "quakeml")
