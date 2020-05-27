"""
Tests for the dataframe extractor.
"""
import obspy.core.event as ev

from obsplus import load_dataset
from obsplus.structures.dfextractor import DataFrameExtractor


# create a dataframe extractor for magnitudes

# note: because we are on python > 3.6 dicts are ordered.
dtypes = {
    "magnitude": float,
    "azimuthal_gap": float,
    "station_count": int,
    "resource_id": str,
    "origin_id": str,
    "event_id": str,
}
ml_to_df = DataFrameExtractor(
    ev.Magnitude, required_columns=list(dtypes), dtypes=dtypes
)


# first extractor, get basic info from the magnitude object
@ml_to_df.extractor
def _get_basic(obj: ev.Magnitude):
    # check mag type, if not ML return None to not add a row for this object
    if obj.magnitude_type != "ML":
        return None
    out = dict(
        magnitude=obj.mag,
        resource_id=str(obj.resource_id),
        azimuthal_gap=obj.azimuthal_gap,
        origin_id=obj.origin_id,
    )
    return out


# add another extractor to get the number of stations.
# the column is obtained from the function name.
@ml_to_df.extractor
def _get_station_count(obj):
    if obj.magnitude_type != "ML":
        return None
    return getattr(obj, "station_count") or -10  # we need a default value for ints


# get events and list of magnitudes
cat = load_dataset("bingham_test").event_client.get_events()
magnitudes = [mag for event in cat for mag in event.magnitudes]
