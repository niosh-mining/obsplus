""" tests for wavebank utilities """
import textwrap

import obspy
import obspy.core.event as ev
import pytest
from obspy import UTCDateTime
from obspy.core.event import Catalog, Event, Origin

import obsplus
from obsplus.utils import compose_docstring

append_func_name = pytest.append_func_name


# ------------------------- module level fixtures


class TestGetReferenceTime:
    """ tests for getting reference times from various objects """

    time = obspy.UTCDateTime("2009-04-01")
    fixtures = []

    # fixtures
    @pytest.fixture(scope="class")
    @append_func_name(fixtures)
    def utc_object(self):
        return obspy.UTCDateTime(self.time)

    @pytest.fixture(scope="class")
    @append_func_name(fixtures)
    def timestamp(self):
        return self.time.timestamp

    @pytest.fixture(scope="class")
    @append_func_name(fixtures)
    def event(self):
        origin = Origin(time=self.time, latitude=47, longitude=-111.7)
        return Event(origins=[origin])

    @pytest.fixture(scope="class")
    @append_func_name(fixtures)
    def catalog(self, event):
        return Catalog(events=[event])

    @pytest.fixture(scope="class")
    def picks(self):
        t1, t2 = UTCDateTime("2016-01-01"), UTCDateTime("2015-01-01")
        picks = [ev.Pick(time=t1), ev.Pick(time=t2), ev.Pick()]
        return picks

    @pytest.fixture(scope="class")
    def event_only_picks(self, picks):
        return ev.Event(picks=picks)

    @pytest.fixture(scope="class", params=fixtures)
    def time_outputs(self, request):
        """ meta fixtures to gather up all the input types"""
        fixture_value = request.getfixturevalue(request.param)
        return obsplus.utils.get_reference_time(fixture_value)

    # tests
    def test_is_utc_date(self, time_outputs):
        """ ensure the output is a UTCDateTime """
        assert isinstance(time_outputs, obspy.UTCDateTime)

    def test_time_equals(self, time_outputs):
        """ ensure the outputs are equal to time on self """
        assert time_outputs == self.time

    def test_empty_event_raises(self):
        """ ensure an empty event will raise """
        event = ev.Event()
        with pytest.raises(ValueError):
            obsplus.utils.get_reference_time(event)

    def test_event_with_picks(self, event_only_picks):
        """ test that an event with picks, no origin, uses smallest pick """
        t_expected = UTCDateTime("2015-01-01")
        t_out = obsplus.utils.get_reference_time(event_only_picks)
        assert t_expected == t_out


class TestIterate:
    def test_none(self):
        """ None should return an empty tuple """
        assert obsplus.utils.iterate(None) == tuple()

    def test_object(self):
        """ A single object should be returned in a tuple """
        assert obsplus.utils.iterate(1) == (1,)


class TestMisc:
    """ misc tests for small utilities """

    def test_no_std_out(self, capsys):
        """ ensure print doesn't propagate to std out when suppressed. """
        with obsplus.utils.no_std_out():
            print("whisper")
        # nothing should have made it to stdout to get captured
        assert not capsys.readouterr().out

    def test_to_timestamp(self):
        """ ensure things are properly converted to timestamps. """
        ts1 = obsplus.utils.to_timestamp(10, None)
        ts2 = obspy.UTCDateTime(10).timestamp
        assert ts1 == ts2
        on_none = obsplus.utils.to_timestamp(None, 10)
        assert on_none == ts1 == ts2

    def test_docstring(self):
        """ Ensure docstrings can be composed with the docstring decorator. """
        params = textwrap.dedent(
            """
        Parameters
        ----------
        a: int
            a
        b int
            b
        """
        )

        @compose_docstring(params=params)
        def testfun1():
            """
            {params}
            """

        assert "Parameters" in testfun1.__doc__
        line = [x for x in testfun1.__doc__.split("\n") if "Parameters" in x][0]
        base_spaces = line.split("Parameters")[0]
        assert len(base_spaces) == 12
