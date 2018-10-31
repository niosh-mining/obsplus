"""
Simple script to use obspy to get catalogs for testing
"""
import os

import obspy


def get_cat1(name="cat1.xml"):
    """ the default obspy cat_name """
    if os.path.exists(name):
        return
    cat = obspy.read_events()
    cat.write(name, "quakeml")


def get_cat2(name="cat2.xml"):
    """ a large iris cat_name """
    if os.path.exists(name):
        return
    from obspy.clients.fdsn import Client

    client = Client("IRIS")
    # times
    t1 = obspy.UTCDateTime("2008-01-01")
    t2 = obspy.UTCDateTime("2008-01-08")
    cat = client.get_events(starttime=t1, endtime=t2, minmagnitude=2.5)
    cat.write(name, "quakeml")


def get_cat3(name="cat3.xml"):
    """
    a cat_name with a few events that have multiple origins/magnitudes
    """
    if os.path.exists(name):
        return
    from obspy.clients.fdsn import Client

    client = Client("IRIS")
    # times
    t1 = obspy.UTCDateTime("2016-09-03")
    t2 = obspy.UTCDateTime("2016-09-04")
    cat = client.get_events(
        starttime=t1,
        endtime=t2,
        minmagnitude=5.0,
        includeallmagnitudes=True,
        includeallorigins=True,
    )
    cat.write(name, "quakeml")


class MegaCatalog(object):
    """
    A class to Create a events with a single event that has many features.

    Uses most (maybe all) the event related classes. The events can be
    accessed as the events attribute of a MegaCatalog instance.
    """

    def __init__(self):
        # handle imports in init to avoid circular imports
        import obspy.core.event as ev
        from obspy import UTCDateTime, Catalog

        self.ev = ev
        self.UTCDateTime = UTCDateTime
        self.Catalog = Catalog
        self.ResourceIdentifier = ev.ResourceIdentifier
        # create events and bind to self
        self.time = UTCDateTime("2016-05-04T12:00:01")
        events = [self._create_event()]
        self.catalog = Catalog(events=events)

    def _create_event(self):
        event = self.ev.Event(
            event_type="mining explosion",
            event_descriptions=[self._get_event_description()],
            picks=[self._create_pick()],
            origins=[self._create_origins()],
            station_magnitudes=[self._get_station_mag()],
            magnitudes=[self._create_magnitudes()],
            amplitudes=[self._get_amplitudes()],
            focal_mechanisms=[self._get_focal_mechanisms()],
        )
        # set preferred origin, focal mech, magnitude
        preferred_objects = dict(
            origin=event.origins[-1].resource_id,
            focal_mechanism=event.focal_mechanisms[-1].resource_id,
            magnitude=event.magnitudes[-1].resource_id,
        )
        for item, value in preferred_objects.items():
            setattr(event, "preferred_" + item + "_id", value)

        return event

    def _create_pick(self):
        # setup some of the classes
        creation = self.ev.CreationInfo(
            agency="SwanCo",
            author="Indago",
            creation_time=self.UTCDateTime(),
            version="10.10",
            author_url=self.ResourceIdentifier("smi:local/me.com"),
        )

        pick = self.ev.Pick(
            time=self.time,
            comments=[self.ev.Comment(x) for x in "BOB"],
            evaluation_mode="manual",
            evaluation_status="final",
            creation_info=creation,
            phase_hint="P",
            polarity="positive",
            onset="emergent",
            back_azimith_errors={"uncertainty": 10},
            slowness_method_id=self.ResourceIdentifier("smi:local/slow"),
            backazimuth=122.1,
            horizontal_slowness=12,
            method_id=self.ResourceIdentifier(),
            horizontal_slowness_errors={"uncertainty": 12},
            filter_id=self.ResourceIdentifier(),
            waveform_id=self.ev.WaveformStreamID("UU", "FOO", "--", "HHZ"),
        )
        self.pick_id = pick.resource_id
        return pick

    def _create_origins(self):
        ori = self.ev.Origin(
            resource_id=self.ResourceIdentifier("smi:local/First"),
            time=self.UTCDateTime("2016-05-04T12:00:00"),
            time_errors={"uncertainty": 0.01},
            longitude=-111.12525,
            longitude_errors={"uncertainty": 0.020},
            latitude=47.48589325,
            latitude_errors={"uncertainty": 0.021},
            depth=2.123,
            depth_errors={"uncertainty": 1.22},
            depth_type="from location",
            time_fixed=False,
            epicenter_fixed=False,
            reference_system_id=self.ResourceIdentifier(),
            method_id=self.ResourceIdentifier(),
            earth_model_id=self.ResourceIdentifier(),
            arrivals=[self._get_arrival()],
            composite_times=[self._get_composite_times()],
            quality=self._get_origin_quality(),
            origin_type="hypocenter",
            origin_uncertainty=self._get_origin_uncertainty(),
            region="US",
            evaluation_mode="manual",
            evaluation_status="final",
        )
        self.origin_id = ori.resource_id
        return ori

    def _get_arrival(self):
        return self.ev.Arrival(
            resource_id=self.ResourceIdentifier("smi:local/Ar1"),
            pick_id=self.pick_id,
            phase="P",
            time_correction=0.2,
            azimuth=12,
            distance=10,
            takeoff_angle=15,
            takeoff_angle_errors={"uncertainty": 10.2},
            time_residual=0.02,
            horizontal_slowness_residual=12.2,
            backazimuth_residual=12.2,
            time_weight=0.23,
            horizontal_slowness_weight=12,
            backazimuth_weight=12,
            earth_model_id=self.ResourceIdentifier(),
            commens=[self.ev.Comment(x) for x in "Nothing"],
        )

    def _get_composite_times(self):
        return self.ev.CompositeTime(
            year=2016,
            year_errors={"uncertainty": 0},
            month=5,
            month_errors={"uncertainty": 0},
            day=4,
            day_errors={"uncertainty": 0},
            hour=0,
            hour_errors={"uncertainty": 0},
            minute=0,
            minute_errors={"uncertainty": 0},
            second=0,
            second_errors={"uncertainty": 0.01},
        )

    def _get_origin_quality(self):
        return self.ev.OriginQuality(
            associate_phase_count=1,
            used_phase_count=1,
            associated_station_count=1,
            used_station_count=1,
            depth_phase_count=1,
            standard_error=0.02,
            azimuthal_gap=0.12,
            ground_truth_level="GT0",
        )

    def _get_origin_uncertainty(self):
        return self.ev.OriginUncertainty(
            horizontal_uncertainty=1.2,
            min_horizontal_uncertainty=0.12,
            max_horizontal_uncertainty=2.2,
            confidence_ellipsoid=self._get_confidence_ellipsoid(),
            preferred_description="uncertainty ellipse",
        )

    def _get_confidence_ellipsoid(self):
        return self.ev.ConfidenceEllipsoid(
            semi_major_axis_length=12,
            semi_minor_axis_length=12,
            major_axis_plunge=12,
            major_axis_rotation=12,
        )

    def _create_magnitudes(self):
        return self.ev.Magnitude(
            resource_id=self.ResourceIdentifier(),
            mag=5.5,
            mag_errors={"uncertainty": 0.01},
            magnitude_type="Mw",
            origin_id=self.origin_id,
            station_count=1,
            station_magnitude_contributions=[self._get_station_mag_contrib()],
        )

    def _get_station_mag(self):
        station_mag = self.ev.StationMagnitude(mag=2.24)
        self.station_mag_id = station_mag.resource_id
        return station_mag

    def _get_station_mag_contrib(self):
        return self.ev.StationMagnitudeContribution(
            station_magnitude_id=self.station_mag_id
        )

    def _get_event_description(self):
        return self.ev.EventDescription(
            text="some text about the EQ", type="earthquake name"
        )

    def _get_amplitudes(self):
        return self.ev.Amplitude(
            generic_amplitude=0.0012,
            type="A",
            unit="m",
            period=1,
            time_window=self._get_timewindow(),
            pick_id=self.pick_id,
            scalling_time=self.time,
            mangitude_hint="ML",
            scaling_time_errors=self.ev.QuantityError(uncertainty=42.0),
        )

    def _get_timewindow(self):
        return self.ev.TimeWindow(
            begin=1.2, end=2.2, reference=self.UTCDateTime("2016-05-04T12:00:00")
        )

    def _get_focal_mechanisms(self):
        return self.ev.FocalMechanism(
            nodal_planes=self._get_nodal_planes(),
            principal_axis=self._get_principal_axis(),
            azimuthal_gap=12,
            station_polarity_count=12,
            misfit=0.12,
            station_distribution_ratio=0.12,
            moment_tensor=self._get_moment_tensor(),
        )

    def _get_nodal_planes(self):
        return self.ev.NodalPlanes(
            nodal_plane_1=self.ev.NodalPlane(strike=12, dip=2, rake=12),
            nodal_plane_2=self.ev.NodalPlane(strike=12, dip=2, rake=12),
            preferred_plane=2,
        )

    def _get_principal_axis(self):
        return self.ev.PrincipalAxes(t_axis=15, p_axis=15, n_axis=15)

    def _get_moment_tensor(self):
        return self.ev.MomentTensor(
            scalar_moment=12213,
            tensor=self._get_tensor(),
            variance=12.23,
            variance_reduction=98,
            double_couple=0.22,
            clvd=0.55,
            iso=0.33,
            source_time_function=self._get_source_time_function(),
            data_used=[self._get_data_used()],
            method_id=self.ResourceIdentifier(),
            inversion_type="general",
        )

    def _get_tensor(self):
        return self.ev.Tensor(
            m_rr=12,
            m_rr_errors={"uncertainty": 0.01},
            m_tt=12,
            m_pp=12,
            m_rt=12,
            m_rp=12,
            m_tp=12,
        )

    def _get_source_time_function(self):
        return self.ev.SourceTimeFunction(
            type="triangle", duration=0.12, rise_time=0.33, decay_time=0.23
        )

    def _get_data_used(self):
        return self.ev.DataUsed(
            wave_type="body waves",
            station_count=12,
            component_count=12,
            shortest_period=1,
            longest_period=20,
        )


def get_cat4(name="cat4.xml"):
    """ A manually created single-event cat_name with many features """

    cat = MegaCatalog().catalog
    cat.write(name, "quakeml")


if __name__ == "__main__":
    get_cat1()
    get_cat2()
    get_cat3()
    get_cat4()
