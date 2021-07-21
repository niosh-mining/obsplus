"""
A module for testing the mill.
"""
from functools import partial
from typing_extensions import Annotated

import pandas as pd
import pytest

import obsplus.events.schema as eschema
import numpy as np
from obsplus.structures.mill import _OperationResolver
from obsplus.utils.time import to_datetime64


@pytest.fixture(scope="class")
def event_op_resolver(bing_eventmill):
    """Return a partial of the Operation resolver fixed on event level."""
    kwargs = {"mill": bing_eventmill, "model_name": "Event"}
    op_res = partial(_OperationResolver, **kwargs)
    return op_res


class TestDFMapper:
    """tests for mapping dataframes to tree structures."""


class TestOperationResolver:
    """Tests for resolving query operations."""

    def test_parent(self, event_op_resolver, bingham_events):
        """Test that parents can be obtained."""
        spec = eschema.Event.parent().resource_id
        res = event_op_resolver(spec=spec, dtype="str")
        out = res()
        assert (out == str(bingham_events.resource_id)).all()

    def test_match(self, event_op_resolver, bingham_events):
        """Test that matching works"""

        def _get_mls():
            """Get local mags from catalog."""
            ml_keys = {}
            for event in bingham_events:
                ml = None
                ml_keys[str(event.resource_id)] = np.NaN
                for mag in event.magnitudes:
                    if mag.magnitude_type == "ML":
                        ml = mag
                if ml:
                    ml_keys[str(event.resource_id)] = ml.mag
            return pd.Series(ml_keys)

        spec = eschema.Event.magnitudes.match(magnitude_type="ML").mag.last()
        res = event_op_resolver(spec=spec, dtype=float)
        out1 = res()
        out2 = _get_mls()
        # ensure index is identical and values are close
        assert np.all(out1.index.values == out2.index.values)
        mask = ~(np.isnan(out1.values) | np.isnan(out2.values))
        assert np.allclose(out1.values[mask], out2.values[mask])

    def test_first(self, event_op_resolver, bingham_events):
        """Tests for getting the first pick time."""

        def _get_first_pick_time():
            out = {}
            for event in bingham_events:
                out[str(event.resource_id)] = np.NaN
                if event.picks:
                    pick = event.picks[0]
                    out[str(event.resource_id)] = to_datetime64(pick.time)
            return pd.Series(out)

        spec = eschema.Event.picks.first().time
        dtype = Annotated[np.datetime64, to_datetime64]
        out1 = event_op_resolver(spec=spec, dtype=dtype)()
        out2 = _get_first_pick_time()
        # ensure index is identical and values are close
        assert np.all(out1.index.values == out2.index.values)
        mask = ~(np.isnan(out1.values) | np.isnan(out2.values))
        ar1, ar2 = out1.values[mask].astype(int), out2.values[mask].astype(int)
        assert np.allclose(ar1, ar2)

    def test_antimatch(self, event_op_resolver, bingham_events):
        """Tests for antimatching."""

        def _get_first_non_rejected_pick():
            """returns id of first non-rejected pick for each event."""
            out = {}
            for event in bingham_events:
                out[str(event.resource_id)] = ""
                for pick in event.picks:
                    if pick.evaluation_status != "rejected":
                        out[str(event.resource_id)] = str(pick.resource_id)
                        break
            return pd.Series(out)

        # get id of first non-rejected event
        spec = (
            eschema.Event.picks.antimatch(evaluation_status="rejected")
            .first()
            .resource_id
        )
        out1 = event_op_resolver(spec=spec, dtype=str)()
        out2 = _get_first_non_rejected_pick()
        assert out1.equals(out2)

    def test_aggregate(self, event_op_resolver, bingham_events):
        """Tests for getting minimum values grouped by model"""

        def get_min_mags():
            """Get min mags by event id."""
            out = {}
            for event in bingham_events:
                rid = str(event.resource_id)
                out[rid] = np.NaN
                for mag in event.magnitudes:
                    if not mag.mag > out[rid]:
                        out[rid] = mag.mag
            return pd.Series(out)

        spec = eschema.Event.magnitudes.mag.aggregate(np.min)
        out1 = event_op_resolver(spec=spec, dtype=float)()
        out2 = get_min_mags()
        # ensure index is identical and values are close
        assert np.all(out1.index.values == out2.index.values)
        mask = ~(np.isnan(out1.values) | np.isnan(out2.values))
        assert np.allclose(out1.values[mask], out2.values[mask])


class TestOperationSetter:
    """Tests for setting values to mill"""
